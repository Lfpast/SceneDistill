#!/bin/bash
# ============================================================================
# SpatialStack 变体: 用 VGGT-Omega 替代 VGGT-1B 做几何层级融合
# ----------------------------------------------------------------------------
# 与 train_spatialstack.sh 完全一致, 只是几何编码器换成 VGGT-Omega:
#   - GEOMETRY_ENCODER_TYPE=vggt_omega  (patch_size=16, feature_dim=2048)
#   - fusion_method=deepstack_language_add
#   - encoder_layers=[11,17,23]  (与 VGGT-Omega cached layers {4,11,17,23} 兼容)
#   - fusion_layers=[0,1,2]
#
# 与 VGGT-1B 版的差异 (代码自动处理, 无需手动干预):
#   - VGGT-Omega patch_size=16, VGGT-1B=14
#   - build_qwen3_5_geometry_inputs 会按 encoder patch_size 缩放输入图像,
#     使 merged token 数与 Qwen 视觉 token 对齐
#   - Omega 的 feature_dim 也是 2048, fusion MLP 结构不变
#
# 用法:
#   bash scripts/train/train_spatialstack_omega.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export USE_GEOMETRY_ENCODER=True
export GEOMETRY_ENCODER_TYPE=vggt_omega
# HF repo id: 编码器会 snapshot_download 到 HF_HOME/hub;
# 也可指本地 .pt 或含 vggt_omega_1b_512.pt 的目录.
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-Omega}"
export FEATURE_FUSION_METHOD=deepstack_language_add
export GEOMETRY_ENCODER_LAYERS="11 17 23"
export GEOMETRY_FUSION_LAYERS="0 1 2"
export DATA_FLATTEN=False
export OUTPUT_DIR="${OUTPUT_DIR:-/project/peilab/jys/qwen3_5_output/spatialstack-omega}"
export CACHE_DIR="${CACHE_DIR:-${HUGGINGFACE_HUB_CACHE:-/project/peilab/jys/spatialstack_store/hf_cache/hub}}"

bash "${SCRIPT_DIR}/train.sh"
