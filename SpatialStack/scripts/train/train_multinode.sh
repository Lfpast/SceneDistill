#!/bin/bash
# Launch one train.sh process per Slurm node. Each train.sh process then starts
# NPROC_PER_NODE local workers with torch.distributed.run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" || -z "${SLURM_NNODES:-}" ]]; then
    echo "[ERROR] train_multinode.sh must run inside a Slurm allocation (use sbatch or salloc)." >&2
    exit 1
fi

if ! command -v scontrol >/dev/null 2>&1 || ! command -v srun >/dev/null 2>&1; then
    echo "[ERROR] Slurm commands scontrol and srun must be available." >&2
    exit 1
fi

export NNODES="${SLURM_NNODES}"
allocated_gpus_per_node="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ ! "${allocated_gpus_per_node}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Cannot determine GPUs per node from the Slurm allocation." >&2
    echo "[ERROR] SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset} SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-unset}" >&2
    exit 1
fi

if [[ -n "${NPROC_PER_NODE:-}" && "${NPROC_PER_NODE}" != "${allocated_gpus_per_node}" ]]; then
    echo "[WARN] Ignoring NPROC_PER_NODE=${NPROC_PER_NODE}; Slurm allocated ${allocated_gpus_per_node} GPUs per node." >&2
fi
export NPROC_PER_NODE="${allocated_gpus_per_node}"
export SLURM_EXPORT_ENV=ALL

if [[ ! "${NNODES}" =~ ^[1-9][0-9]*$ || ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NNODES and NPROC_PER_NODE must be positive integers." >&2
    exit 1
fi

mapfile -t slurm_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#slurm_hosts[@]}" -ne "${NNODES}" ]]; then
    echo "[ERROR] Expected ${NNODES} Slurm hosts, found ${#slurm_hosts[@]}." >&2
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-${slurm_hosts[0]}}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "[INFO] slurm_job_id=${SLURM_JOB_ID}"
echo "[INFO] node_list=${SLURM_JOB_NODELIST}"
echo "[INFO] master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"
echo "[INFO] nnodes=${NNODES} allocated_gpus_per_node=${allocated_gpus_per_node} nproc_per_node=${NPROC_PER_NODE} world_size=$((NNODES * NPROC_PER_NODE))"

srun \
    --nodes "${NNODES}" \
    --ntasks "${NNODES}" \
    --ntasks-per-node 1 \
    --kill-on-bad-exit=1 \
    bash -c '
        export NODE_RANK="${SLURM_PROCID}"
        echo "[INFO] host=$(hostname) node_rank=${NODE_RANK}/${NNODES} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
        exec bash "$1"
    ' _ "${SCRIPT_DIR}/train.sh"
