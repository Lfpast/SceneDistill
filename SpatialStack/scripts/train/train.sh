#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

set_default_env() {
    local name="$1"
    local value="$2"
    if [[ -z "${!name+x}" ]]; then
        export "${name}=${value}"
    fi
}

if [[ -z "${PYTHONPATH+x}" ]]; then
    export PYTHONPATH="${PROJECT_ROOT}/src"
elif [[ ":${PYTHONPATH}:" != *":${PROJECT_ROOT}/src:"* ]]; then
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
fi

set_default_env MODEL_PATH "Qwen/Qwen3.5-4B"
set_default_env OUTPUT_DIR "./output/spatialstack_train"
set_default_env CACHE_DIR "./cache"
set_default_env DATASETS "spar_234k%60,llava_hound_64k%60,vlm3r_scannet%60,vsi_appr_order%50"

set_default_env USE_GEOMETRY_ENCODER "True"
set_default_env GEOMETRY_ENCODER_TYPE "vggt"
set_default_env GEOMETRY_ENCODER_PATH "facebook/VGGT-1B"
set_default_env GEOMETRY_ENCODER_FREEZE "True"
set_default_env REFERENCE_FRAME "first"
set_default_env FEATURE_FUSION_METHOD "deepstack_language_add"
set_default_env GEOMETRY_MERGER_TYPE "mlp"
set_default_env GEOMETRY_ENCODER_LAYERS "11 17 23"
set_default_env GEOMETRY_FUSION_LAYERS "0 1 2"
set_default_env VISION_LANGUAGE_FUSION_LAYERS ""
set_default_env GEOMETRY_DIRECT_TOKEN_MODE "special17"
set_default_env GEOMETRY_TOKEN_INSERT_POSITION "front"
set_default_env PRE_DISTILL_WEIGHT "0.05"
set_default_env POST_DISTILL_WEIGHT "0.05"
set_default_env POS_ENCODING_TYPE "none"

set_default_env DATA_FLATTEN "False"
set_default_env VIDEO_MAX_FRAMES "8"
set_default_env VIDEO_MIN_FRAMES "4"
set_default_env BASE_INTERVAL "2"
set_default_env MAX_PIXELS "451584"
set_default_env MIN_PIXELS "12544"
set_default_env VIDEO_MAX_FRAME_PIXELS "25088"
set_default_env VIDEO_MIN_FRAME_PIXELS "3136"
set_default_env MAX_SAMPLES "-1"
set_default_env SHUFFLE "True"

set_default_env LR "1e-5"
set_default_env TOTAL_BATCH_SIZE "64"
set_default_env PER_DEVICE_TRAIN_BATCH_SIZE "1"
set_default_env NUM_TRAIN_EPOCHS "1"
set_default_env MODEL_MAX_LENGTH "8192"
set_default_env WARMUP_RATIO "0.03"
set_default_env LR_SCHEDULER_TYPE "cosine"
export SAVE_STRATEGY="no"
set_default_env LOGGING_STEPS "1"
set_default_env BF16 "True"
set_default_env TF32 "True"
set_default_env GRADIENT_CHECKPOINTING "True"
set_default_env REPORT_TO "wandb"
set_default_env WANDB_PROJECT "SceneDistill"
set_default_env WANDB_MODE "online"
set_default_env WANDB_LOCAL_ROOT "${TMPDIR:-/tmp}/SceneDistill-wandb"
set_default_env WANDB_DIR "${WANDB_LOCAL_ROOT}/runs"
set_default_env WANDB_CACHE_DIR "${WANDB_LOCAL_ROOT}/cache"
set_default_env WANDB_DATA_DIR "${WANDB_LOCAL_ROOT}/data"
set_default_env WANDB_ARTIFACT_DIR "${WANDB_LOCAL_ROOT}/artifacts"
set_default_env WANDB_CONSOLE "off"
set_default_env WANDB_DISABLE_GIT "true"
set_default_env WANDB_DISABLE_CODE "true"
set_default_env RUN_NAME "$(basename "${OUTPUT_DIR}")"
set_default_env DATALOADER_NUM_WORKERS "4"
set_default_env DEEPSPEED_CONFIG "scripts/zero2_opt.json"

set_default_env TUNE_MM_LLM "True"
set_default_env TUNE_MM_MLP "False"
set_default_env TUNE_MM_VISION "False"

if [[ -z "${NPROC_PER_NODE+x}" ]]; then
    NPROC_PER_NODE="$(python - <<'PY'
import torch

count = torch.cuda.device_count()
print(count if count > 0 else 1)
PY
)"
fi
export NPROC_PER_NODE

set_default_env NNODES "1"
set_default_env NODE_RANK "0"
set_default_env MASTER_ADDR "127.0.0.1"
set_default_env MASTER_PORT "29500"

world_size=$((NNODES * NPROC_PER_NODE))
if [[ -z "${GRADIENT_ACCUMULATION_STEPS+x}" ]]; then
    denom=$((world_size * PER_DEVICE_TRAIN_BATCH_SIZE))
    GRADIENT_ACCUMULATION_STEPS=$(((TOTAL_BATCH_SIZE + denom - 1) / denom))
    if [[ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]]; then
        GRADIENT_ACCUMULATION_STEPS=1
    fi
fi
export GRADIENT_ACCUMULATION_STEPS

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_ARTIFACT_DIR}"

train_args=(
    --model_name_or_path "${MODEL_PATH}"
    --cache_dir "${CACHE_DIR}"
    --dataset_use "${DATASETS}"
    --output_dir "${OUTPUT_DIR}"
    --run_name "${RUN_NAME}"
    --report_to "${REPORT_TO}"
    --learning_rate "${LR}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --warmup_ratio "${WARMUP_RATIO}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --model_max_length "${MODEL_MAX_LENGTH}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --save_strategy "${SAVE_STRATEGY}"
    --logging_steps "${LOGGING_STEPS}"
    --bf16 "${BF16}"
    --tf32 "${TF32}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --remove_unused_columns False
    --tune_mm_llm "${TUNE_MM_LLM}"
    --tune_mm_mlp "${TUNE_MM_MLP}"
    --tune_mm_vision "${TUNE_MM_VISION}"
    --use_geometry_encoder "${USE_GEOMETRY_ENCODER}"
    --geometry_encoder_type "${GEOMETRY_ENCODER_TYPE}"
    --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}"
    --geometry_encoder_freeze "${GEOMETRY_ENCODER_FREEZE}"
    --reference_frame "${REFERENCE_FRAME}"
    --feature_fusion_method "${FEATURE_FUSION_METHOD}"
    --geometry_merger_type "${GEOMETRY_MERGER_TYPE}"
    --geometry_direct_token_mode "${GEOMETRY_DIRECT_TOKEN_MODE}"
    --geometry_token_insert_position "${GEOMETRY_TOKEN_INSERT_POSITION}"
    --pre_distill_weight "${PRE_DISTILL_WEIGHT}"
    --post_distill_weight "${POST_DISTILL_WEIGHT}"
    --pos_encoding_type "${POS_ENCODING_TYPE}"
    --data_flatten "${DATA_FLATTEN}"
    --video_max_frames "${VIDEO_MAX_FRAMES}"
    --video_min_frames "${VIDEO_MIN_FRAMES}"
    --base_interval "${BASE_INTERVAL}"
    --max_pixels "${MAX_PIXELS}"
    --min_pixels "${MIN_PIXELS}"
    --video_max_frame_pixels "${VIDEO_MAX_FRAME_PIXELS}"
    --video_min_frame_pixels "${VIDEO_MIN_FRAME_PIXELS}"
    --max_samples "${MAX_SAMPLES}"
    --shuffle "${SHUFFLE}"
)

if [[ -n "${GEOMETRY_ENCODER_LAYERS}" ]]; then
    train_args+=(--geometry_encoder_layers ${GEOMETRY_ENCODER_LAYERS})
fi

if [[ -n "${GEOMETRY_FUSION_LAYERS}" ]]; then
    train_args+=(--geometry_fusion_layers ${GEOMETRY_FUSION_LAYERS})
fi

if [[ -n "${VISION_LANGUAGE_FUSION_LAYERS}" ]]; then
    train_args+=(--vision_language_fusion_layers ${VISION_LANGUAGE_FUSION_LAYERS})
fi

if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
    train_args+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] model_path=${MODEL_PATH}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] datasets=${DATASETS}"
echo "[INFO] report_to=${REPORT_TO} wandb_project=${WANDB_PROJECT:-}"
echo "[INFO] wandb_mode=${WANDB_MODE:-} wandb_dir=${WANDB_DIR:-}"
echo "[INFO] nproc_per_node=${NPROC_PER_NODE} nnodes=${NNODES} node_rank=${NODE_RANK}"
echo "[INFO] total_batch_size=${TOTAL_BATCH_SIZE} per_device=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "[INFO] save_strategy=${SAVE_STRATEGY} final_model_save=true"
echo "[INFO] geometry=${USE_GEOMETRY_ENCODER} type=${GEOMETRY_ENCODER_TYPE} fusion=${FEATURE_FUSION_METHOD} pre_distill_weight=${PRE_DISTILL_WEIGHT} post_distill_weight=${POST_DISTILL_WEIGHT}"

python -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    src/qwen_vl/train/train_qwen.py \
    "${train_args[@]}"
