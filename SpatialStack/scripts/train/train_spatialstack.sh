#!/bin/bash
# ============================================================================
# SpatialStack 默认配方: Qwen3.5-4B + VGGT-1B 几何层级融合
# ----------------------------------------------------------------------------
# 走 train_qwen.py:240-260 的 Qwen3_5ForConditionalGenerationWithGeometry,
# fusion_method = deepstack_language_add:
#   - VGGT 取第 11/17/23 层 (24 层中末段三层) 几何特征, 全程冻结
#   - 在 Qwen 解码器第 0/1/2 层后, 把几何特征通过
#       RMSNorm -> MLP(把 4 个 patch 拼一起 -> lang_hidden)
#     加 (add) 到对应位置的视觉 token 上. 末层 Linear zero-init,
#     保证训练初期对底座 VLM 是 NoOp.
#   - loss = 标准 SFT Cross-Entropy.
# 可训练参数 = LLM 主干 + language_feature_fusion (zero-init MLP).
#
# 用法:
#   bash scripts/train/train_spatialstack.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export USE_GEOMETRY_ENCODER=True
export GEOMETRY_ENCODER_TYPE="${GEOMETRY_ENCODER_TYPE:-vggt}"
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-1B}"
export FEATURE_FUSION_METHOD=deepstack_language_add
export GEOMETRY_ENCODER_LAYERS="11 17 23"
export GEOMETRY_FUSION_LAYERS="0 1 2"
export DATA_FLATTEN=False
export OUTPUT_DIR="${OUTPUT_DIR:-./output/spatialstack_qwen35_train}"

bash "${SCRIPT_DIR}/train.sh"
