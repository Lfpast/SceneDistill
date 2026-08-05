#!/bin/bash
#SBATCH --job-name=vggt-direct-scene
#SBATCH --partition=normal            # Change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --output=vggt-direct-scene.out
# ============================================================================
# VGGT-Direct (scene16 变体): 只把 VGGT-Omega 的 16 个 register/scene token
# 拼到每帧 visual span 前, 不含 camera token, 无蒸馏.
# ----------------------------------------------------------------------------
# 跟 train_vggt_direct.sh 完全一样, 只是 GEOMETRY_DIRECT_TOKEN_MODE=scene16.
# 每帧注入 K=16 个 register token (跳过 VGGT-Omega 输出的第一个 camera token).
#
# 用法:
#   bash scripts/train/train_vggt_direct_scene.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GEOMETRY_DIRECT_TOKEN_MODE=scene16
export OUTPUT_DIR="${OUTPUT_DIR:-/project/peilab/jys/qwen3_5_output/vggt-direct-scene}"
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

bash "${SCRIPT_DIR}/train_vggt_direct.sh"
