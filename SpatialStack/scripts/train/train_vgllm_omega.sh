#!/bin/bash
#SBATCH --job-name=vgllm-omega
#SBATCH --partition=normal            # Change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --time=22:00:00
#SBATCH --account=peilab
#SBATCH --output=slurm_logs/vgllm-omega.out
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
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

module load slurm
module load cuda12.2/toolkit/12.2.2
source activate spatialstack
cd "${PROJECT_ROOT}"
export LD_LIBRARY_PATH=$(python -c "import os, glob; paths=[os.path.abspath(x) for x in glob.glob('/home/yjiaag/.conda/envs/spatialstack/lib/python3.12/site-packages/nvidia/*/lib')]; print(':'.join(paths))"):$LD_LIBRARY_PATH
export REPO_ROOT=/home/yjiaag/SceneDistill/SpatialStack
export SS_ROOT=/project/peilab/jys/spatialstack_store
export HF_HOME=$SS_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_XET_HIGH_PERFORMANCE=1
export LD_PRELOAD=/home/yjiaag/.conda/envs/spatialstack/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export USE_GEOMETRY_ENCODER=True
export GEOMETRY_ENCODER_TYPE=vggt_omega
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-Omega}"
export FEATURE_FUSION_METHOD=add
export GEOMETRY_ENCODER_LAYERS="23"
export GEOMETRY_FUSION_LAYERS=""      # post-merger add 不需要 fusion_layers
export DATA_FLATTEN=False
export OUTPUT_DIR="${OUTPUT_DIR:-/project/peilab/jys/qwen35_output/vgllm-omega}"
export CACHE_DIR="${CACHE_DIR:-${HUGGINGFACE_HUB_CACHE:-/project/peilab/jys/spatialstack_store/hf_cache/hub}}"
export WANDB_PROJECT="SceneDistill"
export WANDB_MODE="online"
export WANDB_LOCAL_ROOT="${TMPDIR:-/tmp}/SceneDistill-wandb"
export WANDB_DIR="${WANDB_LOCAL_ROOT}/runs"
export WANDB_CACHE_DIR="${WANDB_LOCAL_ROOT}/cache"
export WANDB_DATA_DIR="${WANDB_LOCAL_ROOT}/data"
export WANDB_ARTIFACT_DIR="${WANDB_LOCAL_ROOT}/artifacts"
export WANDB_CONSOLE="off"
export WANDB_DISABLE_GIT="true"
export WANDB_DISABLE_CODE="true"

bash "/home/yjiaag/SceneDistill/SpatialStack/scripts/train/train.sh"
