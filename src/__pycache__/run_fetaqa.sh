#!/bin/bash

cd /data/tyj/2D-TPE-main/src
export PYTHONPATH=/data/tyj/2D-TPE-main:$PYTHONPATH


ENABLE_TABLE_BLOCKS=${ENABLE_TABLE_BLOCKS:-True}
TABLE_BLOCK_ROWS=${TABLE_BLOCK_ROWS:-3}
TABLE_BLOCK_COLS=${TABLE_BLOCK_COLS:-999}
TABLE_HEADER_ROWS=${TABLE_HEADER_ROWS:-0}
TABLE_READ_MODE=${TABLE_READ_MODE:-adaptive}  # row / column / snake / hilbert / spiral / adaptive / 2d
ADAPTIVE_MOE_START_EPOCH=${ADAPTIVE_MOE_START_EPOCH:-0.2}
ADAPTIVE_ROUTER_ROW_BIAS=${ADAPTIVE_ROUTER_ROW_BIAS:-1.0}
ADAPTIVE_ROUTER_PRIOR=${ADAPTIVE_ROUTER_PRIOR:-row=0.40,snake=0.54,spiral=0.02,hilbert=0.02,column=0.02}
ADAPTIVE_ROUTER_INIT_STD=${ADAPTIVE_ROUTER_INIT_STD:-1e-4}
OUTPUT_DIR=${OUTPUT_DIR:-/data/tyj/2D-TPE-main/output/fetaqa_${TABLE_READ_MODE}}
TABLE_BLOCKS_DUMP_PATH=${TABLE_BLOCKS_DUMP_PATH:-/data/tyj/2D-TPE-main/output/fetaqa_table_blocks_3_999_${TABLE_READ_MODE}.json}

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=20213 sft_minicpm_block_textmeta.py  \
        --model_name_or_path /data/tyj/2D-TPE-main/model/MiniCPM-2B-sft-bf16-llama-format \
        --bf16 True \
        --output_dir ${OUTPUT_DIR} \
        --model_max_length 4096 \
        --use_flash_attn True \
        --data_path /data/tyj/2D-TPE-main/data/fetaqa_train_7325.json \
        --enable_table_blocks ${ENABLE_TABLE_BLOCKS} \
        --table_block_rows ${TABLE_BLOCK_ROWS} \
        --table_block_cols ${TABLE_BLOCK_COLS} \
        --table_header_rows ${TABLE_HEADER_ROWS} \
        --table_read_mode ${TABLE_READ_MODE} \
        --adaptive_moe_start_epoch ${ADAPTIVE_MOE_START_EPOCH} \
        --adaptive_router_row_bias ${ADAPTIVE_ROUTER_ROW_BIAS} \
        --adaptive_router_prior ${ADAPTIVE_ROUTER_PRIOR} \
        --adaptive_router_init_std ${ADAPTIVE_ROUTER_INIT_STD} \
        --table_blocks_dump_path ${TABLE_BLOCKS_DUMP_PATH} \
        --low_rank_training False \
        --num_train_epochs 2  \
        --per_device_train_batch_size 2     \
        --gradient_accumulation_steps 4     \
        --evaluation_strategy "no"     \
        --save_strategy "epoch"     \
        --save_total_limit 1     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_ratio 0.03     \
        --lr_scheduler_type "cosine"     \
        --logging_steps 10     \
        --deepspeed /data/tyj/2D-TPE-main/ds_configs/stage2.json \
        --tf32 True
