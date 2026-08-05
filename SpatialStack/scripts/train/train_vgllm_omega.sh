#!/bin/bash
# ============================================================================
# VG-LLM 变体: 用 VGGT-Omega 替代 VGGT-1B 做 post-merger add 融合
# ----------------------------------------------------------------------------
# 与 train_vgllm.sh 完全一致, 只是几何编码器换成 VGGT-Omega:
#   - GEOMETRY_ENCODER_TYPE=vggt_omega
#   - fusion_method=add   (post-merger 一次性相加)
#   - encoder_layers=[23]  (VGGT-Omega 最深 cached layer)
#
# Patch size 差异由 data 层自动处理, 见 train_spatialstack_omega.sh 的注释.
#
# 用法:
#   bash scripts/train/train_vgllm_omega.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export USE_GEOMETRY_ENCODER=True
export GEOMETRY_ENCODER_TYPE=vggt_omega
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-Omega}"
export FEATURE_FUSION_METHOD=add
export GEOMETRY_ENCODER_LAYERS="23"
export GEOMETRY_FUSION_LAYERS=""      # post-merger add 不需要 fusion_layers
export DATA_FLATTEN=False
export OUTPUT_DIR="${OUTPUT_DIR:-/project/peilab/jys/qwen3_5_output/vgllm-omega}"
export CACHE_DIR="${CACHE_DIR:-${HUGGINGFACE_HUB_CACHE:-/project/peilab/jys/spatialstack_store/hf_cache/hub}}"

bash "${SCRIPT_DIR}/train.sh"
