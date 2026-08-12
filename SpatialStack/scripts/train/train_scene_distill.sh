#!/bin/bash
#SBATCH --job-name=SceneDistill-005
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=14:00:00
#SBATCH --account=peilab
#SBATCH --output=slurm_logs/SceneDistill-005.out
# ============================================================================
# SceneDistill (spetial17 变体): 把 VGGT-Omega 的 17 个 camera + scene token
# 拼到每帧 visual span 前并蒸馏.
# 蒸馏机制是 Frame-wise Cross-attention + Global Camera/Scene Self-attention
# ----------------------------------------------------------------------------
# 用法:
#   bash scripts/train/train_vggt_direct_scene.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# module load slurm
# module load cuda12.2/toolkit/12.2.2
# source activate spatialstack
# cd "${PROJECT_ROOT}"
# export LD_LIBRARY_PATH=$(python -c "import os, glob; paths=[os.path.abspath(x) for x in glob.glob('/home/yjiaag/.conda/envs/spatialstack/lib/python3.12/site-packages/nvidia/*/lib')]; print(':'.join(paths))"):$LD_LIBRARY_PATH
# export REPO_ROOT=/home/yjiaag/SceneDistill/SpatialStack
# export SS_ROOT=/project/peilab/jys/spatialstack_store
# export HF_HOME=$SS_ROOT/hf_cache
# export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
# export HF_XET_HIGH_PERFORMANCE=1
# export LD_PRELOAD=/home/yjiaag/.conda/envs/spatialstack/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12
# export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
export USE_GEOMETRY_ENCODER=True
export GEOMETRY_ENCODER_TYPE=scene_distill
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-Omega}"
export GEOMETRY_ENCODER_FREEZE=True
export REFERENCE_FRAME=first
export GEOMETRY_DIRECT_TOKEN_MODE=special17
export GEOMETRY_TOKEN_INSERT_POSITION=front
export PRE_DISTILL_WEIGHT="${PRE_DISTILL_WEIGHT:-0.05}"
export POST_DISTILL_WEIGHT="${POST_DISTILL_WEIGHT:-0.05}"
export FEATURE_FUSION_METHOD=none
export GEOMETRY_ENCODER_LAYERS=""
export GEOMETRY_FUSION_LAYERS=""
export VISION_LANGUAGE_FUSION_LAYERS=""
export TUNE_MM_LLM=True
export TUNE_MM_MLP=False
export TUNE_MM_VISION=False
export OUTPUT_DIR="${OUTPUT_DIR:-/project/peilab/jys/qwen35_output/temp}"
export CACHE_DIR="${CACHE_DIR:-${HUGGINGFACE_HUB_CACHE}}"

export WANDB_PROJECT="SceneDistill"
export WANDB_MODE="offline"
export WANDB_LOCAL_ROOT="${TMPDIR:-/tmp}/SceneDistill-wandb"
export WANDB_DIR="${WANDB_LOCAL_ROOT}/runs"
export WANDB_CACHE_DIR="${WANDB_LOCAL_ROOT}/cache"
export WANDB_DATA_DIR="${WANDB_LOCAL_ROOT}/data"
export WANDB_ARTIFACT_DIR="${WANDB_LOCAL_ROOT}/artifacts"
export WANDB_CONSOLE="off"
export WANDB_DISABLE_GIT="true"
export WANDB_DISABLE_CODE="true"

bash "/home/yjiaag/SceneDistill/SpatialStack/scripts/train/train.sh"
