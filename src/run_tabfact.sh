#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/data/tyj/2D-TPE-main}
cd "${ROOT_DIR}/src"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"


# ===== Table block and read-order settings =====
ENABLE_TABLE_BLOCKS=${ENABLE_TABLE_BLOCKS:-False}
TABLE_BLOCK_ROWS=${TABLE_BLOCK_ROWS:-6}
TABLE_BLOCK_COLS=${TABLE_BLOCK_COLS:-999}
TABLE_HEADER_ROWS=${TABLE_HEADER_ROWS:-0}
TABLE_READ_MODE=${TABLE_READ_MODE:-adaptive}

# For TabFact, empty means use the full 92,283-example training set.
TRAIN_SAMPLE_SIZE=${TRAIN_SAMPLE_SIZE:-7000}

# ===== Adaptive MoE routing settings, mainly used when TABLE_READ_MODE=adaptive =====
ADAPTIVE_MOE_START_EPOCH=${ADAPTIVE_MOE_START_EPOCH:-1}
ADAPTIVE_ROUTER_ROW_BIAS=${ADAPTIVE_ROUTER_ROW_BIAS:-1.0}
ADAPTIVE_ROUTER_PRIOR=${ADAPTIVE_ROUTER_PRIOR:-row=0.40,column=0.22,snake=0.22,hilbert=0.08,spiral=0.08}
ADAPTIVE_ROUTER_INIT_STD=${ADAPTIVE_ROUTER_INIT_STD:-1e-4}
USE_ORDER_MOE=${USE_ORDER_MOE:-True}
ORDER_TOP_K=${ORDER_TOP_K:-2}
ORDER_ROUTER_ENTROPY_COEF=${ORDER_ROUTER_ENTROPY_COEF:-0.1}
ORDER_ROUTER_TEMPERATURE=${ORDER_ROUTER_TEMPERATURE:-0.5}
ORDER_ROUTER_INIT_STD=${ORDER_ROUTER_INIT_STD:-1e-3}
ORDER_ROUTER_BIAS_INIT_STD=${ORDER_ROUTER_BIAS_INIT_STD:-0.1}
ORDER_ROUTER_BIAS=${ORDER_ROUTER_BIAS:-}
ORDER_AUX_SCALE=${ORDER_AUX_SCALE:-0.2}
SHARED_ORDER=${SHARED_ORDER:-row}
ROUTED_ORDERS=${ROUTED_ORDERS:-spiral,hilbert,snake,column}

# ===== Paths =====
# Training loads the base model from MODEL_NAME_OR_PATH. MODEL_PATH is reserved
# for inference scripts, where it points at a trained checkpoint directory.
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"${ROOT_DIR}/model/MiniCPM-2B-sft-bf16-llama-format"}
DATA_PATH=${DATA_PATH:-"${ROOT_DIR}/data/tabfact_train_92283.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT_DIR}/output/tabfact_${TABLE_READ_MODE}"}
TABLE_BLOCKS_DUMP_PATH=${TABLE_BLOCKS_DUMP_PATH:-}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${ROOT_DIR}/ds_configs/stage2.json"}

# ===== Distributed settings =====
CUDA_DEVICES=${CUDA_DEVICES:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-20213}

# ===== Validation and early stopping settings =====
VALIDATION_SIZE=${VALIDATION_SIZE:-1}
EVAL_STEPS=${EVAL_STEPS:-9999}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
EARLY_STOPPING_START_STEP=${EARLY_STOPPING_START_STEP:-9999}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-5}

# ===== Training settings =====
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
LOGGING_STEPS=${LOGGING_STEPS:-10}

TRAIN_ARGS=(
        --model_name_or_path "${MODEL_NAME_OR_PATH}"
        --output_dir "${OUTPUT_DIR}"
        --data_path "${DATA_PATH}"

        --bf16 True
        --model_max_length "${MODEL_MAX_LENGTH}"
        --use_flash_attn True
        --tf32 True

        --enable_table_blocks "${ENABLE_TABLE_BLOCKS}"
        --table_block_rows "${TABLE_BLOCK_ROWS}"
        --table_block_cols "${TABLE_BLOCK_COLS}"
        --table_header_rows "${TABLE_HEADER_ROWS}"
        --table_read_mode "${TABLE_READ_MODE}"

        --adaptive_moe_start_epoch "${ADAPTIVE_MOE_START_EPOCH}"
        --adaptive_router_row_bias "${ADAPTIVE_ROUTER_ROW_BIAS}"
        --adaptive_router_prior "${ADAPTIVE_ROUTER_PRIOR}"
        --adaptive_router_init_std "${ADAPTIVE_ROUTER_INIT_STD}"
        --use_order_moe "${USE_ORDER_MOE}"
        --order_top_k "${ORDER_TOP_K}"
        --order_router_entropy_coef "${ORDER_ROUTER_ENTROPY_COEF}"
        --order_router_temperature "${ORDER_ROUTER_TEMPERATURE}"
        --order_router_init_std "${ORDER_ROUTER_INIT_STD}"
        --order_router_bias_init_std "${ORDER_ROUTER_BIAS_INIT_STD}"
        --order_router_bias "${ORDER_ROUTER_BIAS}"
        --order_aux_scale "${ORDER_AUX_SCALE}"
        --shared_order "${SHARED_ORDER}"
        --routed_orders "${ROUTED_ORDERS}"

        --validation_size "${VALIDATION_SIZE}"
        --evaluation_strategy steps
        --eval_steps "${EVAL_STEPS}"
        --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
        --save_strategy steps
        --save_steps "${EVAL_STEPS}"
        --load_best_model_at_end True
        --metric_for_best_model eval_loss
        --greater_is_better False
        --early_stopping_start_step "${EARLY_STOPPING_START_STEP}"
        --early_stopping_patience "${EARLY_STOPPING_PATIENCE}"
        --save_total_limit 2

        --low_rank_training False
        --num_train_epochs "${NUM_TRAIN_EPOCHS}"
        --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
        --learning_rate "${LEARNING_RATE}"
        --weight_decay 0.0
        --warmup_ratio 0.03
        --lr_scheduler_type cosine
        --logging_steps "${LOGGING_STEPS}"

        --deepspeed "${DEEPSPEED_CONFIG}"
)

if [[ -n "${TRAIN_SAMPLE_SIZE}" ]]; then
        TRAIN_ARGS+=(--train_sample_size "${TRAIN_SAMPLE_SIZE}")
fi

if [[ -n "${TABLE_BLOCKS_DUMP_PATH}" ]]; then
        TRAIN_ARGS+=(--table_blocks_dump_path "${TABLE_BLOCKS_DUMP_PATH}")
fi

echo "Running TabFact training"
echo "  model_name_or_path: ${MODEL_NAME_OR_PATH}"
echo "  table_read_mode: ${TABLE_READ_MODE}"
echo "  enable_table_blocks: ${ENABLE_TABLE_BLOCKS}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  train_sample_size: ${TRAIN_SAMPLE_SIZE:-full}"
echo "  table_blocks_dump_path: ${TABLE_BLOCKS_DUMP_PATH:-disabled}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        sft_minicpm_block_textmeta.py "${TRAIN_ARGS[@]}"
