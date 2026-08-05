#!/bin/bash
# ============================================================================
# 纯 Qwen3.5-4B 指令微调 (无 VGGT, 无几何特征注入)
# ----------------------------------------------------------------------------
# 走 train_qwen.py:262-266 的 Qwen3_5ForConditionalGeneration 原生分支:
#   - 不加载 VGGT 编码器
#   - 没有 fusion / merger / projector
#   - loss = 标准 SFT Cross-Entropy
# 可训练参数 = LLM 主干 (tune_mm_llm=True), ViT 冻结, merger 冻结.
#
# 用法:
#   bash scripts/train/train_pure.sh
#   # 或覆盖输出目录:
#   OUTPUT_DIR=./output/qwen35_pure_v2 bash scripts/train/train_pure.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export USE_GEOMETRY_ENCODER=False
export DATA_FLATTEN=False
export OUTPUT_DIR="${OUTPUT_DIR:-./output/qwen35_pure_sft}"

bash "${SCRIPT_DIR}/train.sh"
