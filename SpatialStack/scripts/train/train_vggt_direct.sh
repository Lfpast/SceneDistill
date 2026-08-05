#!/bin/bash
# ============================================================================
# VGGT-Direct: 把 VGGT-Omega 的 1 camera + 16 scene/register token 直接拼到
# 每帧 visual span 前, 无蒸馏, 只学 direct_projector + LLM
# ----------------------------------------------------------------------------
# 走 train_qwen.py 里 geometry_encoder_type=vggt_omega_direct 的分支:
#   Qwen3_5ForConditionalGenerationWithVGGTOmegaDirect
#     ├── geometry_encoder = VGGTOmegaDirectEncoder (冻结, 取第 23 层前 17 token)
#     ├── direct_projector  = LayerNorm+MLP+gate (可训, ~8M, gate init 1e-2)
#     └── expand_visual_placeholders: 每帧 visual span 前扩 17 个 placeholder,
#         MRoPE 坐标取该帧网格左上角 (frame_top_left)
#
# 可训练参数: LLM 主干 + direct_projector.  VGGT-Omega 全程冻结.
# Loss: 标准 SFT Cross-Entropy, 无蒸馏.
#
# 用法:
#   bash scripts/train/train_vggt_direct.sh
#   # 只注入 1 个 camera token (更轻):
#   GEOMETRY_DIRECT_TOKEN_MODE=camera bash scripts/train/train_vggt_direct.sh
#   # 只注入 16 个 register/scene token (无 camera):
#   GEOMETRY_DIRECT_TOKEN_MODE=scene16 bash scripts/train/train_vggt_direct.sh
#   # 后置注入:
#   GEOMETRY_TOKEN_INSERT_POSITION=back bash scripts/train/train_vggt_direct.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"

# HF repo id: encoder 会 snapshot_download 到 HF_HOME/hub (只下一次).
# 也可指本地 .pt 或含 vggt_omega_1b_*.pt 的目录 (调试用).
export USE_GEOMETRY_ENCODER=True
export GEOMETRY_ENCODER_TYPE=vggt_omega_direct
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-Omega}"

# Direct injection 特有参数
export GEOMETRY_DIRECT_TOKEN_MODE="${GEOMETRY_DIRECT_TOKEN_MODE:-special17}"   # camera(1) | scene16(16) | special17(17)
export GEOMETRY_TOKEN_INSERT_POSITION="${GEOMETRY_TOKEN_INSERT_POSITION:-front}"  # front | back

# Direct 分支不用 layered fusion, 显式清空这些
export FEATURE_FUSION_METHOD=""
export GEOMETRY_FUSION_LAYERS=""
export GEOMETRY_ENCODER_LAYERS=""

export DATA_FLATTEN=False
export OUTPUT_DIR="${OUTPUT_DIR:-./output/qwen35_vggt_direct}"

bash "${SCRIPT_DIR}/train.sh"
