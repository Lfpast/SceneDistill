#!/usr/bin/env python3
"""Extract VGGT-Omega geometry-encoder features for probe training.

The output contract matches the other DL3DV feature extractors:

    <out_dir>/feature_layer{layer_id}.sft

Each file stores one tensor under key ``"feat"`` with shape ``(T, H, W, C)``.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from safetensors.torch import load_file, save_file


logger = logging.getLogger(__name__)
_SUPPORTED_LAYERS = tuple(range(1, 25))


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _add_runtime_paths() -> None:
    root = _workspace_root()
    for path in (root / "SpatialStack" / "src", root / "vggt-omega"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_encoder(model_path: str, device: str, reference_frame: str):
    _add_runtime_paths()
    from qwen_vl.model.geometry_encoders.base import GeometryEncoderConfig
    from qwen_vl.model.geometry_encoders.vggt_omega_encoder import VGGTOmegaEncoder

    encoder = VGGTOmegaEncoder(
        GeometryEncoderConfig(
            encoder_type="vggt_omega",
            model_path=model_path,
            reference_frame=reference_frame,
            freeze_encoder=True,
        )
    )
    encoder.load_model(model_path)
    return encoder.eval().to(torch.device(device))


def _load_preprocess_fn():
    _add_runtime_paths()
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images


def _frame_paths(frame_dir: str, frame_ext: str) -> List[str]:
    return sorted(glob.glob(os.path.join(frame_dir, f"frame_*.{frame_ext}")))


def _query_frame_indices(context_len: int = 76, query_idx_divisor: int = 4) -> List[int]:
    indices = [0, 1]
    idx = 1 + query_idx_divisor
    while idx < context_len:
        indices.append(idx)
        idx += query_idx_divisor
    return indices


def _select_paths(
    frame_dir: str,
    frame_ext: str,
    num_frames: int,
    start_idx: int,
    gt_num_frames: int | None,
    use_query_frame_indices: bool,
    context_len: int,
    query_idx_divisor: int,
) -> List[str]:
    frames = _frame_paths(frame_dir, frame_ext)
    if not frames:
        raise ValueError(f"No frame_*.{frame_ext} files found in {frame_dir}")

    if gt_num_frames is not None:
        clip = frames[start_idx : min(start_idx + gt_num_frames, len(frames))]
    else:
        clip = frames[start_idx:]
    if not clip:
        raise ValueError(
            f"No frames selected from {frame_dir} with start_idx={start_idx}, "
            f"gt_num_frames={gt_num_frames}"
        )

    if use_query_frame_indices:
        query_indices = _query_frame_indices(context_len, query_idx_divisor)
        if len(clip) < context_len:
            mapped = []
            for idx in query_indices:
                mapped_idx = 0 if idx == 0 else int(np.floor(idx / (context_len - 1) * (len(clip) - 1)))
                if mapped_idx < len(clip):
                    mapped.append(mapped_idx)
            selected = list(dict.fromkeys(mapped))
        else:
            selected = [idx for idx in query_indices if idx < len(clip)]
        return [clip[idx] for idx in selected]

    if len(clip) <= num_frames:
        return clip
    selected = np.linspace(0, len(clip) - 1, num_frames).round().astype(int)
    return [clip[idx] for idx in selected]


def _metadata(data_sft: str | None) -> tuple[int, int | None]:
    if not data_sft or not os.path.exists(data_sft):
        logger.warning("No data_sft found; using start_idx=0 and full scene frames.")
        return 0, None

    meta = load_file(data_sft)
    start_idx = int(meta["start_idx"].item())
    if "gt_num_frames" in meta:
        gt_num_frames = int(meta["gt_num_frames"].item())
    elif "images" in meta:
        gt_num_frames = int(meta["images"].shape[0])
    else:
        raise KeyError(f"{data_sft} has neither 'gt_num_frames' nor 'images'")
    return start_idx, gt_num_frames


def _reshape_patch_tokens(feat: torch.Tensor, images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if feat.ndim != 3:
        raise ValueError(f"Expected VGGT-Omega feature shape (T,N,C), got {tuple(feat.shape)}")
    n_frames, _, height, width = images.shape
    h_patch, w_patch = height // patch_size, width // patch_size
    expected_tokens = h_patch * w_patch
    if feat.shape[:2] != (n_frames, expected_tokens):
        raise ValueError(
            f"Feature shape {tuple(feat.shape)} does not match preprocessed images "
            f"{tuple(images.shape)} -> expected {(n_frames, expected_tokens)} tokens"
        )
    return feat.reshape(n_frames, h_patch, w_patch, feat.shape[-1]).contiguous()


def _layer_to_block_idx(layer: int) -> int:
    if layer not in _SUPPORTED_LAYERS:
        raise ValueError(
            f"Unsupported VGGT-Omega layer {layer}. Supported layers: 1-24."
        )
    return layer - 1


def _forward_layers(encoder, images: torch.Tensor, layers: List[int]) -> dict[int, torch.Tensor]:
    block_indices = [_layer_to_block_idx(layer) for layer in layers]
    encoder.vggt_omega.eval()
    encoder.vggt_omega.aggregator.cached_layer_indices = set(block_indices)

    model_images = encoder._apply_reference_frame_transform(images)
    dtype, autocast_context = encoder._autocast_context(model_images)

    with torch.no_grad():
        with autocast_context:
            aggregated_tokens_list, patch_token_start = encoder.vggt_omega.aggregator(
                model_images[None]
            )

    features = {}
    for layer, block_idx in zip(layers, block_indices):
        tokens = aggregated_tokens_list[block_idx]
        if tokens is None:
            raise RuntimeError(
                f"VGGT-Omega layer {layer} (block index {block_idx}) was not cached."
            )
        tokens = encoder._apply_inverse_reference_frame_transform(tokens[0])
        features[layer] = tokens[:, patch_token_start:].to(dtype).contiguous()
    return features


def _missing_layers(out_dir: str, layers: List[int], force: bool) -> List[int]:
    if force:
        return layers
    return [
        layer
        for layer in layers
        if not os.path.exists(os.path.join(out_dir, f"feature_layer{layer}.sft"))
    ]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VGGT-Omega feature extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--data-sft", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-ext", default="png")
    parser.add_argument("--frame-ext", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-frame", choices=["first", "last"], default="first")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--preprocess-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--output-layers", nargs="+", type=int, default=[24])
    parser.add_argument("--use-query-frame-indices", action="store_true")
    parser.add_argument("--context-len", type=int, default=76)
    parser.add_argument("--query-idx-divisor", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime}: [{levelname}] {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    args = parse_args(argv)

    unsupported = sorted(set(args.output_layers) - set(_SUPPORTED_LAYERS))
    if unsupported:
        raise ValueError(
            f"Unsupported VGGT-Omega layer(s): {unsupported}. "
            "Supported layers: 1-24."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    layers = _missing_layers(args.out_dir, args.output_layers, args.force)
    if not layers:
        logger.info("All requested layer files already exist - nothing to do.")
        return

    frame_ext = args.frame_ext or args.image_ext
    start_idx, gt_num_frames = _metadata(args.data_sft)
    paths = _select_paths(
        args.scene_dir,
        frame_ext,
        args.num_frames,
        start_idx,
        gt_num_frames,
        args.use_query_frame_indices,
        args.context_len,
        args.query_idx_divisor,
    )
    logger.info("Selected %d frame(s) from %s", len(paths), args.scene_dir)

    preprocess = _load_preprocess_fn()
    images = preprocess(
        paths,
        mode=args.preprocess_mode,
        image_resolution=args.image_resolution,
        patch_size=16,
    ).to(args.device)
    logger.info("Preprocessed images shape: %s", tuple(images.shape))

    encoder = _load_encoder(args.model_path, args.device, args.reference_frame)
    feats = _forward_layers(encoder, images, layers)

    for layer, feat in feats.items():
        reshaped = _reshape_patch_tokens(feat, images, patch_size=16)
        out_path = os.path.join(args.out_dir, f"feature_layer{layer}.sft")
        save_file({"feat": reshaped.half().cpu()}, out_path)
        logger.info("Saved layer %s: shape %s -> %s", layer, tuple(reshaped.shape), out_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
