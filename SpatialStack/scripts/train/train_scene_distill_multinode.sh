#!/bin/bash
#SBATCH --job-name=SceneDistill-multinode
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=3
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/SceneDistill-multinode-%j.out
#SBATCH --error=slurm_logs/SceneDistill-multinode-%j.err
# ============================================================================
# SceneDistill multi-node training (default: 3 nodes x 2 GPUs).
# Submit from SpatialStack with:
#   sbatch scripts/train/train_scene_distill_multinode.sh
# Override requested resources at submission time, for example:
#   sbatch --nodes=4 --gpus-per-node=2 scripts/train/train_scene_distill_multinode.sh
# NPROC_PER_NODE is derived from the GPUs allocated by Slurm.
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
export GEOMETRY_ENCODER_TYPE=scene_distill
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-Omega}"
export GEOMETRY_ENCODER_FREEZE=True
export REFERENCE_FRAME=first
export GEOMETRY_DIRECT_TOKEN_MODE=special17
export GEOMETRY_TOKEN_INSERT_POSITION=front
export DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.05}"
export FEATURE_FUSION_METHOD=none
export GEOMETRY_ENCODER_LAYERS=""
export GEOMETRY_FUSION_LAYERS=""
export VISION_LANGUAGE_FUSION_LAYERS=""
export DATA_FLATTEN=False
export TUNE_MM_LLM=True
export TUNE_MM_MLP=False
export TUNE_MM_VISION=False
export OUTPUT_DIR="${OUTPUT_DIR:-/project/peilab/jys/qwen35_output/SceneDistill-stage1}"
export CACHE_DIR="${CACHE_DIR:-${HUGGINGFACE_HUB_CACHE}}"

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

export MASTER_PORT="${MASTER_PORT:-29500}"

bash "${SCRIPT_DIR}/train_multinode.sh"
