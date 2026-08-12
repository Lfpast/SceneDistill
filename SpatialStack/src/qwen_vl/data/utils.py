"""Shared frame preparation for the SceneDistill native-video dataflow."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize


def _load_rgb_image(image):
    if isinstance(image, str):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported video frame type: {type(image)}")

    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def load_and_preprocess_video_frames(
    frames: Sequence,
    video_processor,
    min_pixels: int,
    max_pixels: int,
) -> torch.Tensor:
    """Resize one video's real frames once and return uint8 ``[S,3,H,W]`` pixels."""
    if not frames:
        raise ValueError("SceneDistill requires at least one real frame per video.")

    rgb_frames = [_load_rgb_image(frame) for frame in frames]
    source_size = rgb_frames[0].size
    if any(frame.size != source_size for frame in rgb_frames[1:]):
        raise ValueError(
            "All frames in one logical video must share the same source resolution; "
            f"got {[frame.size for frame in rgb_frames]}."
        )

    patch_factor = int(video_processor.patch_size) * int(video_processor.merge_size)
    source_width, source_height = source_size
    target_height, target_width = smart_resize(
        source_height,
        source_width,
        factor=patch_factor,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )
    resized = [
        np.asarray(
            frame.resize((target_width, target_height), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        ).copy()
        for frame in rgb_frames
    ]
    video = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).contiguous()

    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError(f"Video frames must have shape [S,3,H,W], got {tuple(video.shape)}.")
    if video.shape[-2] % patch_factor or video.shape[-1] % patch_factor:
        raise ValueError(
            f"Video height/width must be divisible by {patch_factor}, got {tuple(video.shape[-2:])}."
        )
    return video


def build_geometry_video_inputs(video_frames: torch.Tensor) -> torch.Tensor:
    """Create the VGGT input from the exact uint8 frames consumed by Qwen."""
    if video_frames.ndim != 4 or video_frames.shape[1] != 3 or video_frames.shape[0] == 0:
        raise ValueError(
            "geometry video input must originate from non-empty [S,3,H,W] RGB frames, "
            f"got {tuple(video_frames.shape)}."
        )
    geometry_inputs = video_frames.to(dtype=torch.float32).div_(255.0)
    if not torch.isfinite(geometry_inputs).all():
        raise FloatingPointError("SceneDistill geometry video input contains non-finite values.")
    return geometry_inputs


def prepare_video_inputs(
    frames: Sequence,
    video_processor,
    min_pixels: int,
    max_pixels: int,
) -> dict[str, torch.Tensor]:
    """Prepare the one shared resized frame sequence used by Qwen and VGGT."""
    video_frames = load_and_preprocess_video_frames(
        frames,
        video_processor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return {
        "video_frames": video_frames,
        "geometry_encoder_inputs": build_geometry_video_inputs(video_frames),
    }
