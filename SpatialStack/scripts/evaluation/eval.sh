#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FORCE_OUTPUT_PATH="${FORCE_OUTPUT_PATH:-/project/peilab/jys/qwen35_output/eval}"
cd "${PROJECT_ROOT}"

set_default_env() {
    local name="$1"
    local value="$2"
    if [[ -z "${!name+x}" ]]; then
        export "${name}=${value}"
    fi
}

detect_visible_gpu_count() {
    if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
        local visible_devices="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
        if [[ -z "${visible_devices}" || "${visible_devices}" == "-1" || "${visible_devices}" == "NoDevFiles" ]]; then
            echo 0
            return
        fi

        local -a device_ids
        IFS=',' read -r -a device_ids <<< "${visible_devices}"
        local device_id
        for device_id in "${device_ids[@]}"; do
            if [[ -z "${device_id}" ]]; then
                echo "[ERROR] Invalid CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
                return 1
            fi
        done
        echo "${#device_ids[@]}"
        return
    fi

    local slurm_gpu_count="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
    if [[ "${slurm_gpu_count}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${slurm_gpu_count}"
        return
    fi

    if command -v python >/dev/null 2>&1; then
        python -c 'import torch; print(torch.cuda.device_count())'
        return
    fi

    echo "[ERROR] Cannot determine the number of visible GPUs." >&2
    return 1
}

if [[ -z "${PYTHONPATH+x}" ]]; then
    export PYTHONPATH="${PROJECT_ROOT}/src"
elif [[ ":${PYTHONPATH}:" != *":${PROJECT_ROOT}/src:"* ]]; then
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
fi

set_default_env MODEL_IMPL "qwen3_5"
set_default_env MODEL_PATH "Qwen/Qwen3.5-4B"
set_default_env MODEL_ARGS_BASE "pretrained=${MODEL_PATH},use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true"
set_default_env BENCHMARKS "vsibench,cvbench"
set_default_env OUTPUT_ROOT "logs/eval/$(TZ="${TIMEZONE:-Asia/Shanghai}" date "+%Y%m%d")"
set_default_env OUTPUT_PATH "${OUTPUT_ROOT}"
if [[ -n "${FORCE_OUTPUT_PATH:-}" ]]; then
    export OUTPUT_PATH="${FORCE_OUTPUT_PATH}"
elif [[ -n "${FORCE_OUTPUT_ROOT:-}" ]]; then
    export OUTPUT_PATH="${FORCE_OUTPUT_ROOT}"
fi
set_default_env BATCH_SIZE "1"
VISIBLE_GPU_COUNT="$(detect_visible_gpu_count)"
if [[ ! "${VISIBLE_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Evaluation requires at least one visible GPU, found ${VISIBLE_GPU_COUNT}." >&2
    exit 1
fi
set_default_env PROCESSES_PER_MACHINE "${NUM_PROCESSES:-${VISIBLE_GPU_COUNT}}"
if [[ ! "${PROCESSES_PER_MACHINE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] PROCESSES_PER_MACHINE must be a positive integer, got ${PROCESSES_PER_MACHINE}." >&2
    exit 1
fi
if (( PROCESSES_PER_MACHINE > VISIBLE_GPU_COUNT )); then
    echo "[ERROR] Requested ${PROCESSES_PER_MACHINE} evaluation processes, but only ${VISIBLE_GPU_COUNT} GPUs are visible." >&2
    echo "[ERROR] Check PROCESSES_PER_MACHINE, NUM_PROCESSES, and CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi
export PROCESSES_PER_MACHINE
set_default_env NUM_MACHINES "1"
set_default_env MACHINE_RANK "0"
set_default_env MASTER_ADDR "127.0.0.1"
set_default_env MASTER_PORT "29501"
set_default_env VERBOSITY "INFO"
set_default_env NCCL_NVLS_ENABLE "0"
set_default_env LMMS_EVAL_LAUNCHER "accelerate"

MODEL_ARGS="${MODEL_ARGS_BASE}"
if [[ -n "${MODEL_ARGS_EXTRA:-}" ]]; then
    MODEL_ARGS="${MODEL_ARGS},${MODEL_ARGS_EXTRA}"
fi

IFS=',' read -r -a TASK_LIST <<< "${BENCHMARKS//[[:space:]]/}"
mkdir -p "${OUTPUT_PATH}"
LOG_PATH="${OUTPUT_PATH}/eval.log"

run_benchmark() {
    local task="$1"
    local task_output_path="${OUTPUT_PATH}/${task}"
    mkdir -p "${task_output_path}"

    local cmd=(
        accelerate launch
        --num_processes "${PROCESSES_PER_MACHINE}"
        --num_machines "${NUM_MACHINES}"
        --machine_rank "${MACHINE_RANK}"
        --main_process_ip "${MASTER_ADDR}"
        --main_process_port "${MASTER_PORT}"
        -m lmms_eval
        --model "${MODEL_IMPL}"
        --model_args "${MODEL_ARGS}"
        --tasks "${task}"
        --batch_size "${BATCH_SIZE}"
        --output_path "${task_output_path}"
        --verbosity "${VERBOSITY}"
    )

    if [[ -n "${LIMIT:-}" ]]; then
        cmd+=(--limit "${LIMIT}")
    fi
    if [[ -n "${GEN_KWARGS:-}" ]]; then
        cmd+=(--gen_kwargs "${GEN_KWARGS}")
    fi
    if [[ -n "${INCLUDE_PATH:-}" ]]; then
        cmd+=(--include_path "${INCLUDE_PATH}")
    fi
    if [[ -n "${WANDB_ARGS:-}" ]]; then
        cmd+=(--wandb_args "${WANDB_ARGS}")
    fi
    if [[ -n "${FORCE_DATETIME_STR:-}" ]]; then
        cmd+=(--datetime_str "${FORCE_DATETIME_STR}")
    elif [[ -n "${DATETIME_STR:-}" ]]; then
        cmd+=(--datetime_str "${DATETIME_STR}")
    fi
    if [[ "${LOG_SAMPLES:-0}" == "1" || "${LOG_SAMPLES:-}" == "true" || "${LOG_SAMPLES:-}" == "True" ]]; then
        cmd+=(--log_samples)
        if [[ -n "${FORCE_LOG_SAMPLES_SUFFIX:-}" ]]; then
            cmd+=(--log_samples_suffix "${FORCE_LOG_SAMPLES_SUFFIX}")
        elif [[ -n "${LOG_SAMPLES_SUFFIX:-}" ]]; then
            cmd+=(--log_samples_suffix "${LOG_SAMPLES_SUFFIX}")
        fi
    fi

    echo "[INFO] benchmark=${task}"
    echo "[INFO] benchmark_output_path=${task_output_path}"
    "${cmd[@]}"
}

{
    echo "========== eval $(TZ="${TIMEZONE:-Asia/Shanghai}" date "+%Y-%m-%dT%H:%M:%S%z") =========="
    echo "[INFO] project_root=${PROJECT_ROOT}"
    echo "[INFO] model_impl=${MODEL_IMPL}"
    echo "[INFO] model_args=${MODEL_ARGS}"
    echo "[INFO] benchmarks=${BENCHMARKS//[[:space:]]/}"
    echo "[INFO] visible_gpu_count=${VISIBLE_GPU_COUNT} processes_per_machine=${PROCESSES_PER_MACHINE}"
    echo "[INFO] output_path=${OUTPUT_PATH}"
    echo "[INFO] eval_log=${LOG_PATH}"
    if [[ -n "${FORCE_OUTPUT_PATH:-}" || -n "${FORCE_OUTPUT_ROOT:-}" ]]; then
        echo "[INFO] force_output_path=${OUTPUT_PATH}"
    fi

    for task in "${TASK_LIST[@]}"; do
        if [[ -z "${task}" ]]; then
            continue
        fi
        run_benchmark "${task}"
    done
    echo "[INFO] eval_finished_at=$(TZ="${TIMEZONE:-Asia/Shanghai}" date "+%Y-%m-%dT%H:%M:%S%z")"
} 2>&1 | tee -a "${LOG_PATH}"
