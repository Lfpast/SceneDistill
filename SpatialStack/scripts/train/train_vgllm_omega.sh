#!/bin/bash
#SBATCH --job-name=vgllm-omega
#SBATCH --partition=normal            # Change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --output=vgllm-omega.out
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
export WANDB_PROJECT="spatialstack-omega"
export WANDB_MODE="online"
export WANDB_LOCAL_ROOT="${TMPDIR:-/tmp}/spatialstack-omega-wandb"
export WANDB_DIR="${WANDB_LOCAL_ROOT}/runs"
export WANDB_CACHE_DIR="${WANDB_LOCAL_ROOT}/cache"
export WANDB_DATA_DIR="${WANDB_LOCAL_ROOT}/data"
export WANDB_ARTIFACT_DIR="${WANDB_LOCAL_ROOT}/artifacts"
export WANDB_CONSOLE="off"
export WANDB_DISABLE_GIT="true"
export WANDB_DISABLE_CODE="true"

bash "${SCRIPT_DIR}/train.sh"
