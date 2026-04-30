"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Determine target device. cuda_device_index allows parallel workers to place
    # models on different GPUs without restricting CUDA_VISIBLE_DEVICES (which would
    # break EGL rendering, which requires GPU 0 to remain visible on this system).
    cuda_device_index = getattr(cfg, "cuda_device_index", None)
    if cuda_device_index is not None:
        target_device = torch.device(f"cuda:{cuda_device_index}")
        device_map = {"": cuda_device_index}
    else:
        target_device = DEVICE
        device_map = None

    from_pretrained_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if device_map is not None:
        from_pretrained_kwargs["device_map"] = device_map

    # Detect LoRA adapter checkpoint (has adapter_config.json but no full model weights).
    adapter_config_path = os.path.join(cfg.pretrained_checkpoint, "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        from peft import PeftModel
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config["base_model_name_or_path"]
        print(f"Loading base model from: {base_model_path}")
        vla = AutoModelForVision2Seq.from_pretrained(base_model_path, **from_pretrained_kwargs)
        if device_map is None and not cfg.load_in_8bit and not cfg.load_in_4bit:
            vla = vla.to(target_device)
        print(f"Applying LoRA adapter from: {cfg.pretrained_checkpoint}")
        vla = PeftModel.from_pretrained(vla, cfg.pretrained_checkpoint)
        vla = vla.merge_and_unload()
    else:
        vla = AutoModelForVision2Seq.from_pretrained(cfg.pretrained_checkpoint, **from_pretrained_kwargs)
        if device_map is None and not cfg.load_in_8bit and not cfg.load_in_4bit:
            vla = vla.to(target_device)

    # Load dataset stats used during finetuning (for action un-normalization).
    # For LoRA adapter checkpoints, stats live in the parent run directory.
    from pathlib import Path as _Path
    ckpt_path = _Path(cfg.pretrained_checkpoint)
    dataset_statistics_path = ckpt_path / "dataset_statistics.json"
    if not dataset_statistics_path.is_file():
        dataset_statistics_path = ckpt_path.parent / "dataset_statistics.json"
    if dataset_statistics_path.is_file():
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
    # For LoRA adapter checkpoints, the processor is saved in the parent run directory.
    from pathlib import Path
    processor_path = Path(cfg.pretrained_checkpoint)
    if (processor_path / "adapter_config.json").is_file():
        processor_path = processor_path.parent
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    # Ensure our local apply_transform_dual method is available regardless of which cached
    # processing_prismatic.py was loaded by trust_remote_code.
    if not hasattr(processor.image_processor, "apply_transform_dual"):
        type(processor.image_processor).apply_transform_dual = PrismaticImageProcessor.apply_transform_dual
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs. Use the model's actual device so parallel workers on non-default GPUs work.
    model_device = next(vla.parameters()).device

    # If pre-computed pixel_values are provided (e.g. for seg+depth dual inputs), inject them
    # directly instead of going through the processor's single-image transform path.
    if "pixel_values" in obs:
        text_inputs = processor.tokenizer(prompt, return_tensors="pt")
        inputs = {
            "input_ids": text_inputs["input_ids"].to(model_device, dtype=torch.long),
            "attention_mask": text_inputs["attention_mask"].to(model_device),
            "pixel_values": obs["pixel_values"].unsqueeze(0).to(model_device, dtype=torch.bfloat16),
        }
    else:
        inputs = processor(prompt, image).to(model_device, dtype=torch.bfloat16)

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
