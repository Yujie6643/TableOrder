# TableOrder: Rethinking Cell Ordering for Table Understanding in Large Language Models

This repository contains the code for the paper **"TableOrder: Rethinking Cell Ordering for Table Understanding in Large Language Models"**.

<p align="center">
  <img src="2.png" alt="TableOrder Framework" width="90%">
</p>

---

## 1. 项目结构

下载项目后，主要目录如下：

```text
TableOrder/
├── TPE_Llama/                 # 修改后的 LLaMA / MiniCPM 模型实现
├── ds_configs/                # DeepSpeed 配置
│   └── stage2.json
├── eval_scripts/              # 评测脚本
│   ├── eval_hitab.py
│   ├── table_utils.py
│   └── metric.py
├── src/                       # 训练与推理入口
│   ├── sft_minicpm_block_textmeta.py
│   ├── run_hitab.sh
│   ├── inference_hitab.sh
│   └── inference.py
└── README.md
```

其中，HiTab 的训练入口是：

```bash
src/run_hitab.sh
```

HiTab 的推理入口是：

```bash
src/inference_hitab.sh
```

HiTab 的评测入口是：

```bash
eval_scripts/eval_hitab.py
```

---

## 2. 环境配置

建议使用 Conda 创建独立环境：

```bash
conda create -n tableorder python=3.10 -y
conda activate tableorder
```

安装核心依赖：

```bash
pip install torch transformers datasets accelerate peft deepspeed numpy tqdm psutil
```

如果需要启用 Flash Attention，并且 CUDA / PyTorch 版本支持，可以额外安装：

```bash
pip install flash-attn --no-build-isolation
```

如果安装 Flash Attention 失败，可以先在训练脚本中将：

```bash
--use_flash_attn True
```

改为：

```bash
--use_flash_attn False
```

---

## 3. 路径准备

原始脚本中包含作者本地路径，例如：

```bash
/data/tyj/2D-TPE-main
```

实际运行前，建议统一替换为你的项目路径。假设项目放在：

```bash
/home/your_name/TableOrder
```

可以在 `src/run_hitab.sh` 和 `src/inference_hitab.sh` 中将所有：

```bash
/data/tyj/2D-TPE-main
```

替换为：

```bash
/home/your_name/TableOrder
```

也可以直接在脚本开头设置：

```bash
PROJECT_ROOT=/home/your_name/TableOrder
cd ${PROJECT_ROOT}/src
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
```

然后把脚本中的数据、模型和输出路径都改成 `${PROJECT_ROOT}/...` 的形式。

---

## 4. 模型准备

训练脚本默认加载 MiniCPM 格式的模型：

```bash
--model_name_or_path /data/tyj/2D-TPE-main/model/MiniCPM-2B-sft-bf16-llama-format
```

请将基础模型放到项目的 `model/` 目录下，例如：

```text
TableOrder/
└── model/
    └── MiniCPM-2B-sft-bf16-llama-format/
```

并在 `src/run_hitab.sh` 中修改：

```bash
--model_name_or_path ${PROJECT_ROOT}/model/MiniCPM-2B-sft-bf16-llama-format
```

---

## 5. HiTab 数据准备

训练脚本默认读取：

```bash
data/hitab_train_7417.json
```

推理脚本默认读取：

```bash
eval_data/hitab_test.json
```

因此建议组织为：

```text
TableOrder/
├── data/
│   └── hitab_train_7417.json
└── eval_data/
    └── hitab_test.json
```

每条样本需要包含以下字段：

```json
{
  "instruction": "Answer the question based on the given table.",
  "input_seg": "[TAB] col: Year | Revenue | Profit row 1: 2020 | 10 | 2 [SEP] row 2: 2021 | 15 | 3",
  "question": "What is the profit in 2021?",
  "output": "3"
}
```

代码中会读取 `instruction`、`input_seg`、`question` 和 `output` 字段，并使用 `[TAB]` 标记定位表格内容。因此，HiTab 数据预处理后需要保证 `input_seg` 中包含 `[TAB]`。

---

## 6. 训练 HiTab

进入项目目录：

```bash
cd /home/your_name/TableOrder
```

运行默认 adaptive order 训练：

```bash
bash src/run_hitab.sh
```

默认设置中，HiTab 会启用表格分块：

```bash
ENABLE_TABLE_BLOCKS=True
TABLE_BLOCK_ROWS=6
TABLE_BLOCK_COLS=999
TABLE_HEADER_ROWS=0
TABLE_READ_MODE=adaptive
```

含义如下：

| 参数 | 含义 |
|---|---|
| `ENABLE_TABLE_BLOCKS` | 是否将原始表格切成局部自解释表格块 |
| `TABLE_BLOCK_ROWS` | 每个表格块包含的表体行数，HiTab 默认是 6 |
| `TABLE_BLOCK_COLS` | 每个表格块包含的列数，999 基本等价于不按列切分 |
| `TABLE_HEADER_ROWS` | 表头行数，0 表示自动检测 |
| `TABLE_READ_MODE` | 表格读取顺序，可选 `row`、`column`、`snake`、`hilbert`、`spiral`、`adaptive`、`2d` |

训练输出默认保存到：

```bash
output/hitab_adaptive
```

如果想训练固定 row-major baseline，可以运行：

```bash
TABLE_READ_MODE=row OUTPUT_DIR=output/hitab_row bash src/run_hitab.sh
```

训练 column order：

```bash
TABLE_READ_MODE=column OUTPUT_DIR=output/hitab_column bash src/run_hitab.sh
```

训练 Hilbert order：

```bash
TABLE_READ_MODE=hilbert OUTPUT_DIR=output/hitab_hilbert bash src/run_hitab.sh
```

训练 adaptive order：

```bash
TABLE_READ_MODE=adaptive OUTPUT_DIR=output/hitab_adaptive bash src/run_hitab.sh
```

---

## 7. 关键训练参数说明

`src/run_hitab.sh` 中比较重要的训练参数如下：

```bash
--num_train_epochs 2
--per_device_train_batch_size 2
--gradient_accumulation_steps 4
--learning_rate 2e-5
--model_max_length 4096
--bf16 True
--deepspeed ds_configs/stage2.json
```

如果显存不足，可以优先调整：

```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 8
--model_max_length 2048
--use_flash_attn False
```

如果希望快速检查代码是否能跑通，可以先设置很小的训练样本量，例如在训练参数中加入：

```bash
--train_sample_size 100
```

---

## 8. Adaptive Order MoE 参数说明

当 `TABLE_READ_MODE=adaptive` 时，代码会启用 order routing 相关参数：

```bash
USE_ORDER_MOE=True
SHARED_ORDER=row
ROUTED_ORDERS=spiral,hilbert,snake,column
ORDER_TOP_K=3
ORDER_ROUTER_ENTROPY_COEF=0.1
ORDER_ROUTER_TEMPERATURE=0.5
ORDER_AUX_SCALE=0.2
```

含义如下：

| 参数 | 含义 |
|---|---|
| `SHARED_ORDER=row` | row order 作为共享主分支 |
| `ROUTED_ORDERS` | 参与 Top-K 路由选择的候选 order |
| `ORDER_TOP_K` | 每次保留的 routed order 数量 |
| `ORDER_ROUTER_ENTROPY_COEF` | 路由熵正则系数，正值鼓励路由分布更尖锐 |
| `ORDER_ROUTER_TEMPERATURE` | softmax 温度，越小分布越尖锐 |
| `ORDER_AUX_SCALE` | routed 辅助分支残差强度 |

---

## 9. 推理 HiTab

训练完成后，运行：

```bash
bash src/inference_hitab.sh
```

默认推理配置为：

```bash
DATASET_NAME=hitab
ENABLE_TABLE_BLOCKS=True
TABLE_BLOCK_ROWS=6
TABLE_BLOCK_COLS=999
TABLE_HEADER_ROWS=0
TABLE_READ_MODE=adaptive
MODEL_PATH=output/hitab_adaptive
OUTPUT_PATH=res/hitab_adaptive_res.json
```

如果要推理 row-major 模型：

```bash
TABLE_READ_MODE=row \
MODEL_PATH=output/hitab_row \
OUTPUT_PATH=res/hitab_row_res.json \
bash src/inference_hitab.sh
```

如果要推理 adaptive 模型：

```bash
TABLE_READ_MODE=adaptive \
MODEL_PATH=output/hitab_adaptive \
OUTPUT_PATH=res/hitab_adaptive_res.json \
bash src/inference_hitab.sh
```

推理结果是 JSONL 格式，每一行是一条样本，包含：

```json
{
  "idx": 0,
  "instruction": "...",
  "input_seg": "...",
  "question": "...",
  "output": "gold answer",
  "table_block_mode": true,
  "table_read_mode": "adaptive",
  "table_block_count": 3,
  "predict": "model prediction"
}
```

---

## 10. 小样本推理检查

`src/inference.py` 支持通过环境变量 `INFERENCE_SAMPLE_SIZE` 随机抽取部分测试样本。为了快速检查推理流程，可以运行：

```bash
INFERENCE_SAMPLE_SIZE=100 \
INFERENCE_SAMPLE_SEED=42 \
TABLE_READ_MODE=adaptive \
MODEL_PATH=output/hitab_adaptive \
OUTPUT_PATH=res/hitab_adaptive_sample100_res.json \
bash src/inference_hitab.sh
```

---

## 11. 评测 HiTab

推理完成后，进入评测目录：

```bash
cd /home/your_name/TableOrder/eval_scripts
```

运行：

```bash
python eval_hitab.py --pred_file ../res/hitab_adaptive_res.json
```

评测脚本会读取每一行中的：

```text
predict
output
```

然后调用 `table_utils.evaluate()` 输出最终指标。

如果评测 row-major 结果：

```bash
python eval_hitab.py --pred_file ../res/hitab_row_res.json
```

如果评测 adaptive 结果：

```bash
python eval_hitab.py --pred_file ../res/hitab_adaptive_res.json
```

---

## 12. 推荐的完整运行流程

```bash
# 1. 进入项目
cd /home/your_name/TableOrder

# 2. 训练 adaptive TableOrder
TABLE_READ_MODE=adaptive \
OUTPUT_DIR=output/hitab_adaptive \
bash src/run_hitab.sh

# 3. 推理
TABLE_READ_MODE=adaptive \
MODEL_PATH=output/hitab_adaptive \
OUTPUT_PATH=res/hitab_adaptive_res.json \
bash src/inference_hitab.sh

# 4. 评测
cd eval_scripts
python eval_hitab.py --pred_file ../res/hitab_adaptive_res.json
```
