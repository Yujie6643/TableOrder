#!/bin/bash

cd /data/tyj/2D-TPE-main/src
export PYTHONPATH=/data/tyj/2D-TPE-main:$PYTHONPATH



# ===== 表格分块与读取顺序设置 =====
# True 表示把原始表格切成局部自解释表格块。
ENABLE_TABLE_BLOCKS=${ENABLE_TABLE_BLOCKS:-True}
# 每个局部表格块包含的表体行数。
TABLE_BLOCK_ROWS=${TABLE_BLOCK_ROWS:-3}
# 每个局部表格块包含的列数；999 基本等价于不按列切分。
TABLE_BLOCK_COLS=${TABLE_BLOCK_COLS:-999}
# 表头行数；col_type 使用 1。
TABLE_HEADER_ROWS=${TABLE_HEADER_ROWS:-0}
# 表格读取顺序：row / column / snake / hilbert / spiral / adaptive / 2d。
TABLE_READ_MODE=${TABLE_READ_MODE:-row}

# ===== col_type 专属采样设置 =====
# col_type 训练集较大，默认先随机采样 2000 条再预处理；其他数据集脚本不设置该默认采样。
TRAIN_SAMPLE_SIZE=${TRAIN_SAMPLE_SIZE:-20000}

# ===== Adaptive MoE 路由设置，主要在 TABLE_READ_MODE=adaptive 时使用 =====
# 在该 epoch 之前使用 row-major 预热，之后启用 adaptive 路由。
ADAPTIVE_MOE_START_EPOCH=${ADAPTIVE_MOE_START_EPOCH:-1}
# ADAPTIVE_ROUTER_PRIOR 为空时使用的 row 专家 bias。
ADAPTIVE_ROUTER_ROW_BIAS=${ADAPTIVE_ROUTER_ROW_BIAS:-1.0}
# adaptive 路由专家的初始先验。
ADAPTIVE_ROUTER_PRIOR=${ADAPTIVE_ROUTER_PRIOR:-row=0.40,column=0.22,snake=0.22,hilbert=0.08,spiral=0.08}
# 路由器输出层权重的小随机初始化，用于打破专家之间的完全对称。
ADAPTIVE_ROUTER_INIT_STD=${ADAPTIVE_ROUTER_INIT_STD:-1e-4}
# 是否在 adaptive attention 内启用 row 共享专家 + Top-K 路由专家融合。
USE_ORDER_MOE=${USE_ORDER_MOE:-True}
# 在 column/snake/hilbert/spiral 四个路由专家中保留的 Top-K 数量。
ORDER_TOP_K=${ORDER_TOP_K:-3}
# order MoE 路由熵损失系数；正值鼓励路由分布更尖锐。
ORDER_ROUTER_ENTROPY_COEF=${ORDER_ROUTER_ENTROPY_COEF:-0.1}
# order MoE routed logits softmax 温度；小于 1 会让 Top-K 前分布更尖锐。
ORDER_ROUTER_TEMPERATURE=${ORDER_ROUTER_TEMPERATURE:-0.5}
# order MoE 独立 routed gate 的随机初始化强度。
ORDER_ROUTER_INIT_STD=${ORDER_ROUTER_INIT_STD:-1e-3}
# order MoE 独立 routed gate 的随机 bias 初始化强度，用于打破四路完全对称。
ORDER_ROUTER_BIAS_INIT_STD=${ORDER_ROUTER_BIAS_INIT_STD:-0.1}
# 可选：固定初始 bias logits；默认留空，使用随机 bias。
ORDER_ROUTER_BIAS=${ORDER_ROUTER_BIAS:-}
# order MoE 辅助分支残差强度，row 主分支始终完整保留。
ORDER_AUX_SCALE=${ORDER_AUX_SCALE:-0.2}
# 共享 order 始终计算并参与融合，不参与 Top-K 路由选择。
SHARED_ORDER=${SHARED_ORDER:-snake}
# 参与 Top-K 路由选择的候选 order。
ROUTED_ORDERS=${ROUTED_ORDERS:-spiral,hilbert,row,column}

# ===== 路径设置 =====
# 最终模型目录。
OUTPUT_DIR=${OUTPUT_DIR:-/data/tyj/2D-TPE-main/output/col_type_${TABLE_READ_MODE}}
# 可选的表格块导出文件，用于检查生成的表格块。
TABLE_BLOCKS_DUMP_PATH=${TABLE_BLOCKS_DUMP_PATH:-/data/tyj/2D-TPE-main/output/col_type_table_blocks_3_999_${TABLE_READ_MODE}.json}

# ===== 验证集与早停设置 =====
# 从预处理后的训练样本中留出这么多条作为验证集。
VALIDATION_SIZE=${VALIDATION_SIZE:-1}
# 每 N 个优化器 step 执行一次验证并保存 checkpoint。
EVAL_STEPS=${EVAL_STEPS:-9999}
# 每张卡的验证 batch size；保持较小以避免验证阶段 OOM。
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
# 在该 step 之前忽略验证指标和 checkpoint 保存。
EARLY_STOPPING_START_STEP=${EARLY_STOPPING_START_STEP:-9999}
# eval_loss 连续这么多轮不提升时停止训练。
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-5}

TRAIN_ARGS=(
        # 模型、输出目录和训练数据。
        --model_name_or_path /data/tyj/2D-TPE-main/model/MiniCPM-2B-sft-bf16-llama-format
        --output_dir "${OUTPUT_DIR}"
        --data_path /data/tyj/2D-TPE-main/data/col_type_train_628254.json

        # 精度、上下文长度和加速配置。
        --bf16 True
        --model_max_length 4096
        --use_flash_attn True
        --tf32 True

        # col_type 专属训练采样。
        --train_sample_size "${TRAIN_SAMPLE_SIZE}"

        # 表格分块和读取顺序配置。
        --enable_table_blocks "${ENABLE_TABLE_BLOCKS}"
        --table_block_rows "${TABLE_BLOCK_ROWS}"
        --table_block_cols "${TABLE_BLOCK_COLS}"
        --table_header_rows "${TABLE_HEADER_ROWS}"
        --table_read_mode "${TABLE_READ_MODE}"
        --table_blocks_dump_path "${TABLE_BLOCKS_DUMP_PATH}"

        # adaptive 路由配置。
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

        # 验证集划分和早停配置。
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

        # 训练轮数、batch size、优化器和学习率调度配置。
        --low_rank_training False
        --num_train_epochs 2                  #0.24
        --per_device_train_batch_size 2
        --gradient_accumulation_steps 4
        --learning_rate 2e-5
        --weight_decay 0.0
        --warmup_ratio 0.03
        --lr_scheduler_type "cosine"
        --logging_steps 10

        # 分布式训练和 DeepSpeed 配置。
        --deepspeed /data/tyj/2D-TPE-main/ds_configs/stage2.json
)

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=20213 \
        sft_minicpm_block_textmeta.py "${TRAIN_ARGS[@]}"
