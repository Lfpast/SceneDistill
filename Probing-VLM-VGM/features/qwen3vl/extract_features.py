#!/usr/bin/env python3
"""
Extract features from Qwen3-VL models for probe training.

This script extracts hidden states from specified layers of Qwen3-VL models
and saves them in safetensor format for downstream probe training.

Usage:
    python -m features.qwen3vl.extract_features \
        --scene-dir data/DL3DV/DL3DV-ALL-960P/1K/<hash>/images_4 \
        --data-sft data/DL3DV/DL3DV-processed/1K/<hash>.sft \
        --out-dir data/DL3DV/FEAT/qwen3-vl-8b/1K/<hash> \
        --model-path ckpt/Qwen3-VL-8B-Instruct \
        --model-type qwen3vl \
        --use-query-frame-indices \
        --context-len 76 \
        --query-idx-divisor 4 \
        --output-layers 22
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

import builtins

from pprint import pprint
builtins.pp = pprint


import torch
from safetensors.torch import load_file, save_file

from .base_extractor import get_qwen3vl_extractor

logging.basicConfig(
    level=logging.INFO,
    format="{asctime}: [{levelname}] {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def parse_args(
    argv: List[str] | None = None,
    *,
    model_types=("qwen3vl", "sensenova"),
    default_model_type="qwen3vl",
    model_family="Qwen3-VL",
    default_layers=(9, 18, 27, 36),
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{model_family} feature extractor for probe training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Directory containing frame_*.png files",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for feature files",
    )
    parser.add_argument(
        "--data-sft",
        default=None,
        help="Path to processed .sft file (contains start_idx for frame alignment)",
    )
    parser.add_argument(
        "--image-ext",
        default="png",
        help="Frame file extension (alias for --frame-ext)",
    )
    parser.add_argument(
        "--frame-ext",
        default=None,
        help="Frame file extension",
    )
    
    # Model configuration
    parser.add_argument(
        "--model-path",
        required=True,
        help=f"Path to {model_family} model (local or HuggingFace)",
    )
    parser.add_argument(
        "--model-type",
        choices=model_types,
        default=default_model_type,
        help=f"Model type ({', '.join(model_types)})",
    )
    
    # Extraction configuration
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to sample (ignored when --use-query-frame-indices is set)",
    )
    parser.add_argument(
        "--output-layers",
        nargs="+",
        type=int,
        default=list(default_layers),
        help=f"Layer indices to extract from {model_family}",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Question/prompt for feature extraction",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to load model on (e.g., cuda:0, cuda:1)",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "eager", "flash_attention_2"],
        default="sdpa",
        help="Attention backend for Transformers model loading. "
             "Use sdpa for compatibility; use flash_attention_2 only when "
             "flash-attn is compatible with the system glibc/CUDA/PyTorch.",
    )
    parser.add_argument(
        "--geometry-encoder-path",
        default=None,
        help="Optional SpatialStack geometry encoder checkpoint path.",
    )
    parser.add_argument(
        "--geometry-encoder-type",
        default=None,
        help="Optional SpatialStack geometry encoder type, e.g. vggt_omega.",
    )
    parser.add_argument(
        "--target-size",
        nargs=2,
        type=int,
        default=[960, 540],
        metavar=("W", "H"),
        help="Resize all frames to (W, H) before processing. "
             "Set to 0 0 to keep original resolution.",
    )
    
    # Query frame indices mode (for video_probe_dataset_ compatibility)
    parser.add_argument(
        "--use-query-frame-indices",
        action="store_true",
        help="Use query frame indices matching video_probe_dataset_'s query mechanism "
             "(e.g., [0,1,5,9,...,73]) instead of uniform sampling. "
             "This ensures features align with the dataset's query sampling.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=76,
        help="Context length for query frame indices (default: 76)",
    )
    parser.add_argument(
        "--query-idx-divisor",
        type=int,
        default=4,
        help="Divisor for query frame alignment (default: 4)",
    )
    
    # Other
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing feature files",
    )
    
    return parser.parse_args(argv)


def check_existing_files(out_dir: str, layers: List[int]) -> List[int]:
    """Check which layer files already exist and return missing layers."""
    missing = []
    for layer in layers:
        path = os.path.join(out_dir, f"feature_layer{layer}.sft")
        if not os.path.exists(path):
            missing.append(layer)
    return missing


def main(
    argv: List[str] | None = None,
    *,
    extractor_factory=get_qwen3vl_extractor,
    model_types=("qwen3vl", "sensenova"),
    default_model_type="qwen3vl",
    model_family="Qwen3-VL",
    default_layers=(9, 18, 27, 36),
) -> None:
    args = parse_args(
        argv,
        model_types=model_types,
        default_model_type=default_model_type,
        model_family=model_family,
        default_layers=default_layers,
    )
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Check for existing files
    if not args.force:
        missing_layers = check_existing_files(args.out_dir, args.output_layers)
        if not missing_layers:
            logger.info("All layer files exist. Use --force to overwrite.")
            return
        if len(missing_layers) < len(args.output_layers):
            logger.info(f"Extracting only missing layers: {missing_layers}")
            args.output_layers = missing_layers
    
    # Validate scene directory
    if not os.path.isdir(args.scene_dir):
        logger.error(f"Scene directory not found: {args.scene_dir}")
        sys.exit(1)
    
    # Handle frame_ext/image_ext aliases
    frame_ext = args.frame_ext if args.frame_ext else args.image_ext
    
    # Read start_idx and gt_num_frames from data_sft for frame alignment
    start_idx = 0
    gt_num_frames = None
    if args.data_sft and os.path.exists(args.data_sft):
        meta = load_file(args.data_sft)
        start_idx = int(meta["start_idx"].item())
        # Prefer scalar 'gt_num_frames' key (lightweight metadata convention, e.g. ScanNet);
        # fall back to 'images' leading dim (DL3DV/CO3D convention where all frames are baked
        # into the .sft).
        if "gt_num_frames" in meta:
            gt_num_frames = int(meta["gt_num_frames"].item())
        elif "images" in meta:
            gt_num_frames = meta["images"].shape[0]
        else:
            raise KeyError(
                f"{args.data_sft} has neither 'gt_num_frames' nor 'images'; "
                f"cannot infer clip length."
            )
        logger.info(f"Loaded metadata: start_idx={start_idx}, gt_num_frames={gt_num_frames}")
    else:
        logger.warning("No data_sft provided, using default frame range (may cause misalignment)")
    
    # Parse target_size
    target_size = tuple(args.target_size) if args.target_size[0] > 0 else None
    
    # Get extractor
    logger.info(f"Loading {args.model_type} model from {args.model_path}")
    logger.info(
        f"Device: {args.device}, Layers: {args.output_layers}, "
        f"target_size: {target_size}, attn: {args.attn_implementation}"
    )
    extractor = extractor_factory(
        model_path=args.model_path,
        model_type=args.model_type,
        select_layers=tuple(args.output_layers),
        question=args.prompt,
        device=args.device,
        target_size=target_size,
        attn_implementation=args.attn_implementation,
        geometry_encoder_path=args.geometry_encoder_path,
        geometry_encoder_type=args.geometry_encoder_type,
    )
    
    # Extract features
    logger.info(f"Extracting features from {args.scene_dir}")
    if args.use_query_frame_indices:
        logger.info(f"Using query frame indices mode (context_len={args.context_len}, "
                   f"divisor={args.query_idx_divisor})")
    else:
        logger.info(f"Sampling {args.num_frames} frames from GT range "
                   f"[{start_idx}, {start_idx + (gt_num_frames or 'all')})")
    
    try:
        features = extractor.extract(
            frame_dir=args.scene_dir,
            num_frames=args.num_frames,
            frame_ext=frame_ext,
            start_idx=start_idx,
            gt_num_frames=gt_num_frames,
            use_query_frame_indices=args.use_query_frame_indices,
            context_len=args.context_len,
            query_idx_divisor=args.query_idx_divisor,
        )
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise
    
    # Save features
    saved = 0
    for layer, feat in features.items():
        out_path = os.path.join(args.out_dir, f"feature_layer{layer}.sft")

        if not torch.isfinite(feat).all().item():
            finite = torch.isfinite(feat)
            finite_ratio = finite.float().mean().item()
            nan_count = torch.isnan(feat).sum().item()
            posinf_count = torch.isposinf(feat).sum().item()
            neginf_count = torch.isneginf(feat).sum().item()
            raise FloatingPointError(
                f"Non-finite features for layer {layer} before saving {out_path}: "
                f"finite_ratio={finite_ratio:.6e}, nan={nan_count}, "
                f"posinf={posinf_count}, neginf={neginf_count}"
            )
        
        # Convert to half precision for storage
        feat_half = feat.half().cpu()
        if not torch.isfinite(feat_half).all().item():
            finite = torch.isfinite(feat_half)
            finite_ratio = finite.float().mean().item()
            nan_count = torch.isnan(feat_half).sum().item()
            posinf_count = torch.isposinf(feat_half).sum().item()
            neginf_count = torch.isneginf(feat_half).sum().item()
            raise FloatingPointError(
                f"Non-finite features for layer {layer} after fp16 conversion "
                f"before saving {out_path}: finite_ratio={finite_ratio:.6e}, "
                f"nan={nan_count}, posinf={posinf_count}, neginf={neginf_count}"
            )
        
        logger.info(
            f"Saving layer {layer}: shape {tuple(feat_half.shape)} -> {out_path}"
        )
        save_file({"feat": feat_half}, out_path)
        saved += 1
    
    logger.info(f"Done. Saved {saved} layer files to {args.out_dir}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
