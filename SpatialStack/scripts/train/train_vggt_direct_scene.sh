#!/bin/bash
#SBATCH --job-name=vggt-direct-scene
#SBATCH --partition=normal            # Change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --account=peilab
#SBATCH --output=slurm_logs/vggt-direct-scene.out
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

module load slurm
module load cuda12.2/toolkit/12.2.2
source activate spatialstack
cd /home/dduab/jiayusheng/SpatialStack-omega/SpatialStack
export LD_LIBRARY_PATH=$(python -c "import os, glob; paths=[os.path.abspath(x) for x in glob.glob('/home/dduab/.conda/envs/spatialstack/lib/python3.12/site-packages/nvidia/*/lib')]; print(':'.join(paths))"):$LD_LIBRARY_PATH
export REPO_ROOT=/home/dduab/jiayusheng/SpatialStack-omega/SpatialStack
export SS_ROOT=/project/peilab/jys/spatialstack_store
export HF_HOME=$SS_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_XET_HIGH_PERFORMANCE=1
export LD_PRELOAD=/home/dduab/.conda/envs/spatialstack/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

export GEOMETRY_DIRECT_TOKEN_MODE=scene16
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"

# HF repo id: encoder 会 snapshot_download 到 HF_HOME/hub (只下一次).
# 也可指本地 .pt 或含 vggt_omega_1b_*.pt 的目录 (调试用).
export USE_GEOMETRY_ENCODER=True
export GEOMETRY_ENCODER_TYPE=vggt_omega_direct
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-Omega}"

# Direct injection 特有参数
export GEOMETRY_TOKEN_INSERT_POSITION="${GEOMETRY_TOKEN_INSERT_POSITION:-front}"  # front | back

# Direct 分支不用 layered fusion, 显式清空这些
export FEATURE_FUSION_METHOD=""
export GEOMETRY_FUSION_LAYERS=""
export GEOMETRY_ENCODER_LAYERS=""

export DATA_FLATTEN=False
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

bash "/home/dduab/jiayusheng/SpatialStack-omega/SpatialStack/scripts/train/train.sh"
