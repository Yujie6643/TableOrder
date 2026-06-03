#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/data/tyj/2D-TPE-main}
cd "${ROOT_DIR}/src"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"


export DATASET_NAME=tabfact
export ENABLE_TABLE_BLOCKS=${ENABLE_TABLE_BLOCKS:-False}
export TABLE_BLOCK_ROWS=${TABLE_BLOCK_ROWS:-7}
export TABLE_BLOCK_COLS=${TABLE_BLOCK_COLS:-999}
export TABLE_HEADER_ROWS=${TABLE_HEADER_ROWS:-0}
export TABLE_READ_MODE=${TABLE_READ_MODE:-adaptive}
export MODEL_PATH=${MODEL_PATH:-"${ROOT_DIR}/output/${DATASET_NAME}_${TABLE_READ_MODE}"}
export OUTPUT_PATH=${OUTPUT_PATH:-"${ROOT_DIR}/res/${DATASET_NAME}_${TABLE_READ_MODE}_res.json"}

# Empty means evaluate the full tabfact_test.json.
export INFERENCE_SAMPLE_SIZE=${INFERENCE_SAMPLE_SIZE:-1000}
export INFERENCE_SAMPLE_SEED=${INFERENCE_SAMPLE_SEED:-42}
export CUDA_DEVICES=${CUDA_DEVICES:-0}
RUN_EVAL=${RUN_EVAL:-False}

echo "Running TabFact inference"
echo "  table_read_mode: ${TABLE_READ_MODE}"
echo "  enable_table_blocks: ${ENABLE_TABLE_BLOCKS}"
echo "  model_path: ${MODEL_PATH}"
echo "  output_path: ${OUTPUT_PATH}"
echo "  inference_sample_size: ${INFERENCE_SAMPLE_SIZE:-full}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python inference.py

if [[ "${RUN_EVAL,,}" == "true" || "${RUN_EVAL}" == "1" || "${RUN_EVAL,,}" == "yes" ]]; then
        python "${ROOT_DIR}/eval_scripts/eval_tabfact.py" --pred_file "${OUTPUT_PATH}"
fi
