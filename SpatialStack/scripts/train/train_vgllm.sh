#!/bin/bash
# ============================================================================
# VG-LLM 风格: 在 LLM 输入端一次性把 VGGT 几何特征加进视觉 token
# ----------------------------------------------------------------------------
# 走 train_qwen.py:240-260 的 Qwen3_5ForConditionalGenerationWithGeometry,
# fusion_method = add (post-merger fusion):
#   - VGGT 取最后一层 (第 23 层) 几何特征, 全程冻结.
#   - GeometryFeatureMerger (mlp): 把几何特征下采样到与 Qwen merger 输出
#     同形状 ([N, H/2, W/2, lang_hidden]).
#   - 在 ViT 出 image_embeds 之后 / 进 LLM 之前, 直接相加一次:
#       image_embeds = image_embeds + geo_embeds
#     之后照常 masked_scatter 进 inputs_embeds, LLM 解码层不再注入.
#   - loss = 标准 SFT Cross-Entropy.
# 可训练参数 = LLM 主干 + geometry_merger (MLP) + feature_fusion (无参 add).
#
# 用法:
#   bash scripts/train/train_vgllm.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export USE_GEOMETRY_ENCODER=True
export GEOMETRY_ENCODER_TYPE="${GEOMETRY_ENCODER_TYPE:-vggt}"
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-1B}"
export FEATURE_FUSION_METHOD=add
export GEOMETRY_ENCODER_LAYERS="23"
# post-merger fusion 不需要 geometry_fusion_layers, 留空 (train.sh 已做空判断)
export GEOMETRY_FUSION_LAYERS=""
export DATA_FLATTEN=False
export OUTPUT_DIR="${OUTPUT_DIR:-./output/qwen35_vgllm_add}"

bash "${SCRIPT_DIR}/train.sh"
