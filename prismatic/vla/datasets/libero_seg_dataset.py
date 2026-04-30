"""libero_seg_dataset.py

PyTorch IterableDataset that reads pre-rendered LIBERO segmentation HDF5 files
produced by experiments/robot/libero/collect_libero_seg_data.py.

Each HDF5 file covers one task and contains:
  demos/demo_{i}/images  — uint8 (T, H, W, 3) colorized segmentation frames
  demos/demo_{i}/actions — float32 (T, 7) robot actions
  attrs: language (str)

The dataset cycles through all (task, demo, step) triples indefinitely,
matching the infinite-iteration behaviour of RLDSDataset so that finetune.py
needs no changes to its training loop.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Type

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import IGNORE_INDEX, RLDSBatchTransform


class LiberoSegDataset(IterableDataset):
    """Infinite iterable dataset over LIBERO segmentation demonstrations.

    Compatible with the finetune.py training loop that expects:
      - An IterableDataset (no __len__ required for infinite loops)
      - A .dataset_statistics attribute with q01/q99 action bounds
      - Items of the form dict(pixel_values, input_ids, labels, dataset_name)

    When the HDF5 files contain a 'depths' dataset (produced by collect_libero_seg_data.py
    with use_depth=True), depth maps are loaded and combined with segmentation images via
    apply_transform_dual so that each backbone sees a different modality.
    """

    def __init__(
        self,
        data_dir: Path,
        suite_name: str,
        batch_transform: RLDSBatchTransform,
        image_aug: bool = False,
        shuffle: bool = True,
        image_transform=None,
    ) -> None:
        self.data_dir = Path(data_dir) / suite_name
        self.suite_name = suite_name
        self.batch_transform = batch_transform
        self.image_aug = image_aug
        self.shuffle = shuffle
        self.image_transform = image_transform  # if set, used instead of batch_transform's image path

        hdf5_paths = sorted(self.data_dir.glob("task_*.h5"))
        if not hdf5_paths:
            raise FileNotFoundError(
                f"No task_*.h5 files found under {self.data_dir}.\n"
                "Run collect_libero_seg_data.py first to generate the dataset."
            )

        # Build index: list of (hdf5_path, demo_key, language)
        self._index: list = []
        all_actions: list = []

        for path in hdf5_paths:
            with h5py.File(path, "r") as f:
                language = f.attrs["language"]
                for demo_key in f["demos"].keys():
                    self._index.append((str(path), demo_key, language))
                    all_actions.append(f[f"demos/{demo_key}/actions"][:])

        all_actions_np = np.concatenate(all_actions, axis=0)  # (N_total_steps, 7)
        q01 = np.quantile(all_actions_np, 0.01, axis=0).astype(np.float32)
        q99 = np.quantile(all_actions_np, 0.99, axis=0).astype(np.float32)

        self.dataset_statistics = {
            suite_name: {"action": {"q01": q01, "q99": q99}}
        }

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        order = list(range(len(self._index)))
        while True:
            if self.shuffle:
                random.shuffle(order)
            for idx in order:
                path, demo_key, language = self._index[idx]
                with h5py.File(path, "r") as f:
                    images = f[f"demos/{demo_key}/images"][:]   # (T, H, W, 3) uint8
                    actions = f[f"demos/{demo_key}/actions"][:] # (T, 7) float32
                    has_depth = f"demos/{demo_key}/depths" in f
                    depths = f[f"demos/{demo_key}/depths"][:] if has_depth else None  # (T, H, W, 3) uint8 or None

                step_order = list(range(len(images)))
                if self.shuffle:
                    random.shuffle(step_order)

                for t in step_order:
                    seg_img = Image.fromarray(images[t])
                    action = actions[t]

                    # Reuse RLDSBatchTransform to keep tokenization consistent.
                    # Build a minimal rlds_batch dict matching its expected format.
                    rlds_batch = {
                        "dataset_name": self.suite_name.encode(),
                        "action": action[None, :],              # (1, 7) — batch_transform indexes [0]
                        "observation": {"image_primary": [np.array(seg_img)]},
                        "task": {"language_instruction": language.encode()},
                    }
                    item = self.batch_transform(rlds_batch)

                    # When depth is available and an image_transform supporting dual inputs is
                    # provided, replace pixel_values with the seg+depth fused representation.
                    if has_depth and self.image_transform is not None:
                        depth_img = Image.fromarray(depths[t])
                        item["pixel_values"] = self.image_transform(seg_img, depth_img)

                    yield item
