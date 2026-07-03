#!/bin/bash
#SBATCH --job-name=qwen35-sbatch
#SBATCH --partition=normal            # Change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --time=10:00:00
#SBATCH --account=peilab
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

# Set your cluster account with `#SBATCH --account=...` if your scheduler requires it.

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
fi

cd "$PROJECT_ROOT"
mkdir -p slurm_logs
mkdir -p .cache/triton

source activate probing
module load slurm
module load cuda12.2/toolkit/12.2.2

export SS_ROOT=/project/peilab/jys/probing_data
export HF_HOME=$SS_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_XET_HIGH_PERFORMANCE=1
export PROBING_DATA=/project/peilab/jys/probing_data
export QWEN35_PATH=/project/peilab/jys/spatialstack_store/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name Layer1 \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT/qwen35-4b" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 1

python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name Layer5 \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT/qwen35-4b" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 5

python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name Layer9 \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT/qwen35-4b" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 9

python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name Layer13 \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT/qwen35-4b" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 13

python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name Layer17 \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT/qwen35-4b" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 17

python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name Layer21 \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT/qwen35-4b" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 21

python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name Layer25 \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT/qwen35-4b" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 25

python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name Layer29 \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT/qwen35-4b" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 29

python -m features.run_dl3dv \
  --vfm qwen35 \
  --vfm-name Layer32 \
  --subset all \
  --dl3dv-root "$PROBING_DATA/DL3DV/DL3DV-ALL-960P" \
  --processed-root "$PROBING_DATA/DL3DV/DL3DV-processed" \
  --out-root "$PROBING_DATA/DL3DV/FEAT/qwen35-4b" \
  --model-path "$QWEN35_PATH" \
  --model-type qwen35 \
  --use-query-frame-indices \
  --context-len 76 \
  --query-idx-divisor 4 \
  --output-layers 32