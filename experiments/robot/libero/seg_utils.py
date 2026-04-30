"""Segmentation utilities for OpenVLA LIBERO evaluation and data collection.

Provides three components:
1. Palette + colorization helpers (GT segmentation from MuJoCo)
2. get_libero_seg_image() — drop-in replacement for get_libero_image()
3. SAMSegHandler — wraps SAM 2 for predicted segmentation at eval time
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# 256-entry LUT: index 0 = background (dark gray), indices 1-255 get unique hues.
# Golden-ratio hue spacing maximises perceptual separation between adjacent IDs.
def _make_lut(n: int = 256) -> np.ndarray:
    import colorsys
    lut = np.empty((n, 3), dtype=np.uint8)
    lut[0] = (50, 50, 50)  # background
    h = 0.0
    for i in range(1, n):
        h = (h + 0.618033988749895) % 1.0   # golden ratio
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.90)
        lut[i] = (int(r * 255), int(g * 255), int(b * 255))
    return lut


_LUT = _make_lut()


# ---------------------------------------------------------------------------
# Core colorization
# ---------------------------------------------------------------------------


def colorize_segmentation(seg: np.ndarray) -> np.ndarray:
    """Map an integer segmentation ID array to a uint8 RGB image.

    Args:
        seg: int/uint array of shape (H, W) or (H, W, 1) — per-pixel geom IDs.

    Returns:
        uint8 RGB array of shape (H, W, 3).
    """
    if seg.ndim == 3:
        seg = seg[..., 0]
    # Clip to LUT range; any ID ≥ 256 wraps via modulo first.
    idx = (seg % 256).astype(np.uint8)
    return _LUT[idx]  # (H, W, 3)


def _get_seg_obs(obs: dict) -> np.ndarray:
    """Extract segmentation array from obs dict, robust to robosuite version differences."""
    for key in (
        "agentview_segmentation",
        "agentview_segmentation_instance",
        "agentview_segmentation_class",
        "agentview_segmentation_element",
    ):
        if key in obs:
            return obs[key]
    seg_keys = [k for k in obs if "segmentation" in k]
    raise KeyError(
        f"No segmentation key found in obs. Available segmentation keys: {seg_keys}. "
        "Make sure get_libero_env() was called with use_segmentation=True."
    )


def get_libero_depth_image(obs: dict, resize_size) -> np.ndarray:
    """Extract and resize a depth map from a LIBERO observation.

    Requires the env to be created with use_depth=True. Robosuite's `agentview_depth`
    is already in normalized device coordinates [0, 1] (0 = near plane, 1 = far plane),
    so we map it directly to uint8 without per-frame rescaling. This keeps the same
    physical depth at the same pixel value across timesteps — a precondition for the
    model to learn absolute depth cues.

    Args:
        obs: observation dict from env.step() / env.set_init_state()
        resize_size: int or (H, W) tuple — target image size

    Returns:
        uint8 numpy array of shape (*resize_size, 3) — grayscale depth as 3-channel image
    """
    from experiments.robot.libero.libero_utils import resize_image

    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    depth = obs["agentview_depth"]  # (H, W, 1) float32 in [0, 1] NDC
    if depth.ndim == 3:
        depth = depth[..., 0]  # (H, W)

    depth_u8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)  # (H, W)
    img = np.stack([depth_u8, depth_u8, depth_u8], axis=2)  # (H, W, 3)
    img = img[::-1, ::-1]  # rotate 180° to match seg preprocessing
    img = resize_image(img, resize_size)
    return img


def get_libero_seg_image(obs: dict, resize_size) -> np.ndarray:
    """Extract, colorize, and resize a segmentation mask from a LIBERO observation.

    Drop-in replacement for get_libero_image() when using segmentation inputs.
    Requires the env to be created with use_segmentation=True.

    Args:
        obs: observation dict from env.step() / env.set_init_state()
        resize_size: int or (H, W) tuple — target image size

    Returns:
        uint8 numpy array of shape (*resize_size, 3)
    """
    from experiments.robot.libero.libero_utils import resize_image

    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    seg = _get_seg_obs(obs)
    img = colorize_segmentation(seg)
    img = img[::-1, ::-1]  # rotate 180° to match training preprocessing
    img = resize_image(img, resize_size)
    return img


# ---------------------------------------------------------------------------
# SAM 2 handler (predicted segmentation)
# ---------------------------------------------------------------------------


def colorize_sam_masks(masks: List[dict], shape: Tuple[int, int]) -> np.ndarray:
    """Assign palette colors to SAM 2 predicted masks (largest area = background).

    Args:
        masks: list of mask dicts from SAM2AutomaticMaskGenerator.generate()
        shape: (H, W) of the original image

    Returns:
        uint8 RGB array of shape (H, W, 3)
    """
    out = np.full((*shape, 3), _LUT[0].tolist(), dtype=np.uint8)
    # Sort by area descending; skip first (largest = background, already set).
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    for i, m in enumerate(sorted_masks[1:], start=1):
        out[m["segmentation"]] = _LUT[i % len(_LUT)]
    return out


class SAMSegHandler:
    """Wraps SAM 2 AutomaticMaskGenerator for per-frame segmentation at eval time.

    Install SAM 2 before use:
        pip install git+https://github.com/facebookresearch/sam2.git
    Download a checkpoint, e.g.:
        wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    """

    def __init__(
        self,
        checkpoint: str,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda",
    ) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "SAM 2 is not installed.\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                "Checkpoint download:\n"
                "  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
            )
        sam2 = build_sam2(model_cfg, checkpoint, device=device)
        self.generator = SAM2AutomaticMaskGenerator(sam2)

    def predict(self, rgb_image: np.ndarray, resize_size) -> np.ndarray:
        """Run SAM 2 on an RGB image and return a colorized segmentation image.

        Args:
            rgb_image: uint8 (H, W, 3) RGB image (raw agentview, NOT rotated)
            resize_size: int or (H, W) target size

        Returns:
            uint8 (resize_size, resize_size, 3) colorized mask, rotated 180°
        """
        from experiments.robot.libero.libero_utils import resize_image

        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)

        masks = self.generator.generate(rgb_image)
        colored = colorize_sam_masks(masks, rgb_image.shape[:2])
        colored = colored[::-1, ::-1]  # rotate 180° to match training preprocessing
        return resize_image(colored, resize_size)
