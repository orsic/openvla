"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero.seg_utils import SAMSegHandler, get_libero_seg_image
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    # Task-range parameters for parallel execution (None = use full suite range)
    task_id_start: Optional[int] = None             # First task index to evaluate (inclusive)
    task_id_end: Optional[int] = None               # Last task index to evaluate (exclusive)

    # GPU assignment for parallel execution (None = use CUDA_VISIBLE_DEVICES as-is)
    cuda_device_index: Optional[int] = None         # CUDA device index for model inference (0, 1, ...)

    # Input modality: rgb (default), gt_seg (GT segmentation), sam_seg (SAM 2 predicted masks)
    input_type: str = "rgb"                         # "rgb" | "gt_seg" | "sam_seg"
    sam_checkpoint: Optional[str] = None            # Path to SAM 2 checkpoint (.pt) for sam_seg mode
    sam_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"  # SAM 2 model config path

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Detect torchrun launch (sets LOCAL_RANK / LOCAL_WORLD_SIZE per process).
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    is_torchrun = local_rank >= 0
    if is_torchrun and cfg.cuda_device_index is None:
        cfg.cuda_device_index = local_rank

    # Configure GPU assignment for parallel execution.
    # MUJOCO_EGL_DEVICE_ID must be set before env creation (EGL device is always 0 on this system).
    # For subprocess-based parallelism we also pin CUDA_VISIBLE_DEVICES so GPU 0 stays visible for EGL.
    # Under torchrun all GPUs are already visible, so we skip that override.
    if cfg.cuda_device_index is not None:
        if not is_torchrun:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize LIBERO task suite early so we can resolve task range for the run_id.
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_start = cfg.task_id_start if cfg.task_id_start is not None else 0
    task_end = cfg.task_id_end if cfg.task_id_end is not None else num_tasks_in_suite

    # Under torchrun with no explicit task range, partition tasks evenly across ranks.
    if is_torchrun and cfg.task_id_start is None and cfg.task_id_end is None:
        base, rem = divmod(num_tasks_in_suite, world_size)
        task_start = local_rank * base + min(local_rank, rem)
        task_end = task_start + base + (1 if local_rank < rem else 0)

    # Initialize SAM handler lazily (only when needed)
    sam_handler = None
    if cfg.input_type == "sam_seg":
        assert cfg.sam_checkpoint is not None, "--sam_checkpoint must be set when using --input_type sam_seg"
        sam_handler = SAMSegHandler(cfg.sam_checkpoint, model_cfg=cfg.sam_model_cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if is_torchrun:
        run_id += f"--rank{local_rank}"
    if task_start != 0 or task_end != num_tasks_in_suite:
        run_id += f"--tasks{task_start}-{task_end}"
    if cfg.input_type != "rgb":
        run_id += f"--{cfg.input_type}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            config={"input_type": cfg.input_type, "task_suite": cfg.task_suite_name},
        )

    print(f"Task suite: {cfg.task_suite_name} (tasks {task_start}–{task_end-1})")
    log_file.write(f"Task suite: {cfg.task_suite_name} (tasks {task_start}–{task_end-1})\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(task_start, task_end)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(
            task, cfg.model_family, resolution=256, use_segmentation=(cfg.input_type != "rgb")
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        t_img_total, t_infer_total, t_env_total, t_steps_counted = 0.0, 0.0, 0.0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image (RGB, GT segmentation, or SAM segmentation)
                    _t0 = time.perf_counter()
                    if cfg.input_type == "rgb":
                        img = get_libero_image(obs, resize_size)
                    elif cfg.input_type == "gt_seg":
                        img = get_libero_seg_image(obs, resize_size)
                    else:  # sam_seg
                        img = sam_handler.predict(obs["agentview_image"], resize_size)
                    _t1 = time.perf_counter()

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )
                    _t2 = time.perf_counter()

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    _t3 = time.perf_counter()

                    t_img_total += _t1 - _t0
                    t_infer_total += _t2 - _t1
                    t_env_total += _t3 - _t2
                    t_steps_counted += 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Print per-step timing breakdown for the first episode of the first assigned task
            if task_id == task_start and episode_idx == 0 and t_steps_counted > 0:
                avg_img = t_img_total / t_steps_counted * 1000
                avg_infer = t_infer_total / t_steps_counted * 1000
                avg_env = t_env_total / t_steps_counted * 1000
                avg_total = avg_img + avg_infer + avg_env
                print(f"\n[TIMING] Per-step averages over {t_steps_counted} steps:")
                print(f"  Image preprocess : {avg_img:6.1f} ms ({avg_img/avg_total*100:.0f}%)")
                print(f"  Model inference  : {avg_infer:6.1f} ms ({avg_infer/avg_total*100:.0f}%)")
                print(f"  Env step         : {avg_env:6.1f} ms ({avg_env/avg_total*100:.0f}%)")
                print(f"  Total per step   : {avg_total:6.1f} ms  =>  {1000/avg_total:.1f} steps/s")

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{cfg.input_type}/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
