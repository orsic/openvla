"""eval_open_loop.py

Open-loop (teacher-forced) evaluation on the training HDF5 dataset.

Replicates exactly what finetune.py measures as "training accuracy":
  - feeds ground-truth observations from the HDF5 files
  - runs a teacher-forced forward pass (model sees GT action tokens as context)
  - compares predicted vs GT action tokens

This isolates model accuracy from closed-loop drift.

Usage:
    python experiments/robot/libero/eval_open_loop.py \
        --pretrained_checkpoint runs/<your_checkpoint> \
        --libero_seg_data_dir datasets/libero_seg \
        --libero_suite libero_spatial \
        --num_demos 50
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[3]))

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--libero_seg_data_dir", type=Path, default=Path("datasets/libero_seg"))
    parser.add_argument("--libero_suite", type=str, default="libero_spatial")
    parser.add_argument("--num_demos", type=int, default=50)
    args = parser.parse_args()

    class Cfg:
        model_family = "openvla"
        pretrained_checkpoint = args.pretrained_checkpoint
        load_in_8bit = False
        load_in_4bit = False
        unnorm_key = args.libero_suite
        center_crop = False

    cfg = Cfg()
    model = get_model(cfg)
    model.eval()
    processor = get_processor(cfg)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in args.pretrained_checkpoint else VicunaV15ChatPromptBuilder,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    device = next(model.parameters()).device
    num_patches = model.vision_backbone.featurizer.patch_embed.num_patches

    data_dir = args.libero_seg_data_dir / args.libero_suite
    hdf5_paths = sorted(data_dir.glob("task_*.h5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"No task_*.h5 files found under {data_dir}")

    total_tokens = correct_tokens = 0

    with tqdm.tqdm(hdf5_paths, desc="Tasks") as task_bar:
        for hdf5_path in task_bar:
            with h5py.File(hdf5_path, "r") as f:
                language = f.attrs["language"]
                demo_keys = sorted(f["demos"].keys())[: args.num_demos]
                has_depth = f"demos/{demo_keys[0]}/depths" in f if demo_keys else False

                with tqdm.tqdm(demo_keys, desc=f"  {hdf5_path.stem}", leave=False) as demo_bar:
                    for demo_key in demo_bar:
                        images = f[f"demos/{demo_key}/images"][:]
                        actions = f[f"demos/{demo_key}/actions"][:]
                        depths = f[f"demos/{demo_key}/depths"][:] if has_depth else None

                        for t in range(len(images)):
                            seg_img = Image.fromarray(images[t])

                            # Build batch exactly as LiberoSegDataset does during training
                            rlds_batch = {
                                "dataset_name": args.libero_suite.encode(),
                                "action": actions[t][None, :],
                                "observation": {"image_primary": [np.array(seg_img)]},
                                "task": {"language_instruction": language.encode()},
                            }
                            item = batch_transform(rlds_batch)

                            if has_depth and depths is not None:
                                depth_img = Image.fromarray(depths[t])
                                item["pixel_values"] = processor.image_processor.apply_transform_dual(seg_img, depth_img)

                            batch = collator([item])

                            with torch.no_grad():
                                output = model(
                                    input_ids=batch["input_ids"].to(device),
                                    attention_mask=batch["attention_mask"].to(device),
                                    pixel_values=batch["pixel_values"].to(device, dtype=torch.bfloat16),
                                    labels=batch["labels"],
                                )

                            # Replicate finetune.py accuracy computation exactly
                            action_logits = output.logits[:, num_patches:-1]
                            action_preds = action_logits.argmax(dim=2)
                            action_gt = batch["labels"][:, 1:].to(action_preds.device)
                            mask = action_gt > action_tokenizer.action_token_begin_idx

                            correct_tokens += ((action_preds == action_gt) & mask).sum().item()
                            total_tokens += mask.sum().item()

                        acc = correct_tokens / total_tokens if total_tokens else 0.0
                        demo_bar.set_postfix(acc=f"{acc*100:.1f}%")
                        task_bar.set_postfix(acc=f"{acc*100:.1f}%")

    print(f"\nOpen-loop accuracy: {correct_tokens}/{total_tokens} = {correct_tokens/total_tokens*100:.1f}%")


if __name__ == "__main__":
    main()
