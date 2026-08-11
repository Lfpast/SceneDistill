#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/project/peilab/jys/qwen35_output/SceneDistill-02}"
export MODEL_IMPL="${MODEL_IMPL:-qwen3_5_scene_distill}"
export BENCHMARKS="${BENCHMARKS:-vsibench,cvbench,blink_spatial,sparbench,videomme,mmsibench}"

export MODEL_ARGS_BASE="${MODEL_ARGS_BASE:-pretrained=${MODEL_PATH},use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true}"

bash "${SCRIPT_DIR}/eval.sh"
