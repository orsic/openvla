"""collect_libero_seg_data.py

Re-renders existing LIBERO demonstration trajectories with segmentation masks and
saves them as HDF5 files for LoRA fine-tuning.

Algorithm (per task):
  1. Open the task's demo HDF5 (robomimic format stored under --data_root).
  2. For each demonstration: reset the env to the stored initial state,
     replay the recorded actions, capture colorized segmentation at each step.
  3. Write images + actions + language to an output HDF5 under --output_dir.

Output layout:
  {output_dir}/{suite_name}/task_{task_id:02d}.h5
  HDF5 groups: demos/demo_{i}/images (uint8), demos/demo_{i}/actions (float32)
  HDF5 attr:   language (str)

Usage:
  python experiments/robot/libero/collect_libero_seg_data.py \
      --suite_name libero_spatial \
      --data_root /opt/LIBERO/datasets \
      --output_dir datasets/libero_seg \
      --resize 224
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path

sys.path.insert(0, str(Path(__file__).parents[3]))  # repo root

from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.libero.seg_utils import get_libero_seg_image


def find_demo_hdf5(data_root: Path, suite_name: str, task) -> Path:
    """Locate the HDF5 demo file for a given task.

    LIBERO stores one HDF5 per task, named after the BDDL file without the .bddl extension.
    Tries several path conventions used across LIBERO versions.
    """
    stem = Path(task.bddl_file).stem  # e.g. "LIVING_ROOM_SCENE1_put_the_..."
    candidates = [
        data_root / suite_name / f"{stem}_demo.hdf5",
        data_root / suite_name / f"{stem}.hdf5",
        data_root / f"{suite_name}_demo.hdf5",
        Path(get_libero_path("datasets")) / suite_name / f"{stem}_demo.hdf5",
        Path(get_libero_path("datasets")) / f"{suite_name}_demo.hdf5",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find demo HDF5 for task '{task.problem_folder}/{task.bddl_file}'.\n"
        f"Searched:\n" + "\n".join(f"  {p}" for p in candidates) + "\n"
        f"Set --data_root to the directory containing your LIBERO demo HDF5 files."
    )


def collect_task(
    task_id: int,
    task,
    task_suite,
    data_root: Path,
    suite_name: str,
    output_path: Path,
    resize: int,
    num_demos: int,
) -> None:
    hdf5_path = find_demo_hdf5(data_root, suite_name, task)
    initial_states = task_suite.get_task_init_states(task_id)
    language = task.language

    env, _ = get_libero_env(task, model_family="openvla", resolution=256, use_segmentation=True)

    with h5py.File(hdf5_path, "r") as src, h5py.File(output_path, "w") as dst:
        dst.attrs["language"] = language
        dst.attrs["task_id"] = task_id
        dst.attrs["source_hdf5"] = str(hdf5_path)

        demo_keys = sorted(src["data"].keys())[:num_demos]
        demos_grp = dst.create_group("demos")

        for demo_idx, demo_key in enumerate(tqdm.tqdm(demo_keys, desc=f"  task {task_id}", leave=False)):
            actions = src[f"data/{demo_key}/actions"][:]  # (T, 7)

            env.reset()
            obs = env.set_init_state(initial_states[demo_idx])

            images, saved_actions = [], []
            for action in actions:
                img = get_libero_seg_image(obs, resize_size=resize)
                images.append(img)
                saved_actions.append(action.copy())
                obs, _reward, done, _info = env.step(action.tolist())
                if done:
                    break

            if not images:
                continue

            grp = demos_grp.create_group(f"demo_{demo_idx}")
            grp.create_dataset("images", data=np.stack(images), dtype=np.uint8, compression="lzf")
            grp.create_dataset("actions", data=np.stack(saved_actions), dtype=np.float32)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LIBERO segmentation dataset")
    parser.add_argument("--suite_name", default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--data_root", type=Path, default=Path("/opt/LIBERO/datasets"),
                        help="Root directory containing LIBERO demo HDF5 files")
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/libero_seg"),
                        help="Directory to write output HDF5 files")
    parser.add_argument("--resize", type=int, default=224, help="Output image size (square)")
    parser.add_argument("--num_demos", type=int, default=50,
                        help="Max demos per task (default: 50 = full LIBERO suite)")
    parser.add_argument("--task_id_start", type=int, default=0)
    parser.add_argument("--task_id_end", type=int, default=None)
    args = parser.parse_args()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite_name]()
    num_tasks = task_suite.n_tasks

    task_end = args.task_id_end if args.task_id_end is not None else num_tasks
    out_suite_dir = args.output_dir / args.suite_name
    out_suite_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting segmentation data for {args.suite_name} (tasks {args.task_id_start}–{task_end - 1})")
    print(f"Output: {out_suite_dir}")

    for task_id in tqdm.tqdm(range(args.task_id_start, task_end), desc="Tasks"):
        task = task_suite.get_task(task_id)
        output_path = out_suite_dir / f"task_{task_id:02d}.h5"
        if output_path.exists():
            print(f"  Skipping task {task_id} (already exists: {output_path})")
            continue
        collect_task(
            task_id=task_id,
            task=task,
            task_suite=task_suite,
            data_root=args.data_root,
            suite_name=args.suite_name,
            output_path=output_path,
            resize=args.resize,
            num_demos=args.num_demos,
        )
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
