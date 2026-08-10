#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/project/peilab/jys/qwen3_5_output/SceneDistill-stage1}"
export MODEL_IMPL="${MODEL_IMPL:-qwen3_5_scene_distill}"
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-Omega}"
export BENCHMARKS="${BENCHMARKS:-vsibench,cvbench,blink_spatial,sparbench,videomme,mmsibench}"
export SCENE_DISTILL_STAGE1_COMPATIBILITY="${SCENE_DISTILL_STAGE1_COMPATIBILITY:-true}"

export MODEL_ARGS_BASE="${MODEL_ARGS_BASE:-pretrained=${MODEL_PATH},use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=${GEOMETRY_ENCODER_PATH},scene_distill_stage1_compatibility=${SCENE_DISTILL_STAGE1_COMPATIBILITY},disable_thinking=true}"

bash "${SCRIPT_DIR}/eval.sh"
