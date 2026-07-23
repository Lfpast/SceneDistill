#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-ddz16/qwen35-4B-vgllm-add}"
export MODEL_IMPL="${MODEL_IMPL:-qwen3_5_vgllm_add}"
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-1B}"
export GEOMETRY_ENCODER_LAYERS="${GEOMETRY_ENCODER_LAYERS:-23}"
export BENCHMARKS="${BENCHMARKS:-vsibench, cvbench, blink_spatial, sparbench, videomme, mmsibench}"

export MODEL_ARGS_BASE="${MODEL_ARGS_BASE:-pretrained=${MODEL_PATH},use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=${GEOMETRY_ENCODER_PATH},geometry_encoder_layers=${GEOMETRY_ENCODER_LAYERS},disable_thinking=true}"

bash "${SCRIPT_DIR}/eval.sh"
