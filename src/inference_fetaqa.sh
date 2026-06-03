#!/bin/bash

cd /data/tyj/2D-TPE-main/src
export PYTHONPATH=/data/tyj/2D-TPE-main:$PYTHONPATH



export DATASET_NAME=fetaqa
export ENABLE_TABLE_BLOCKS=${ENABLE_TABLE_BLOCKS:-True}
export TABLE_BLOCK_ROWS=${TABLE_BLOCK_ROWS:-7}
export TABLE_BLOCK_COLS=${TABLE_BLOCK_COLS:-999}
export TABLE_HEADER_ROWS=${TABLE_HEADER_ROWS:-0}
export TABLE_READ_MODE=${TABLE_READ_MODE:-adaptive}
export MODEL_PATH=${MODEL_PATH:-/data/tyj/2D-TPE-main/output/${DATASET_NAME}_${TABLE_READ_MODE}}
export OUTPUT_PATH=${OUTPUT_PATH:-/data/tyj/2D-TPE-main/res/${DATASET_NAME}_${TABLE_READ_MODE}_res.json}

CUDA_VISIBLE_DEVICES=0 python inference.py

