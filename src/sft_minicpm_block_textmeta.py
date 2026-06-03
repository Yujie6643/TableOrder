import traceback
import io
import os
import copy
import re
import json
import math
import logging
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from multiprocessing import cpu_count
from datasets import load_dataset
from tqdm import tqdm
import psutil

import torch
# 关闭 transformers 的建议类警告（例如输入过长提示），因为后面会显式处理超长样本。
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
import transformers
from torch.utils.data import Dataset, IterableDataset, random_split
from datasets.iterable_dataset import IterableDataset
from transformers import Trainer, DataCollatorForLanguageModeling, TrainerCallback, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
import sys
import os
import random
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

try:
    # 兼容不同 transformers 版本中 Flash Attention 可用性检测函数的命名差异。
    from transformers.utils import is_flash_attn_2_available as hf_is_flash_attn_2_available
except ImportError:
    from transformers.utils import is_flash_attn_available as hf_is_flash_attn_2_available

if project_root not in sys.path:
    # 将项目根目录加入导入路径，便于加载本地的 TPE_Llama 模型实现。
    sys.path.append(project_root)

from TPE_Llama.modeling_llama import LlamaForCausalLM


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
ADAPTIVE_TABLE_READ_MODES = ("row", "column", "snake", "hilbert", "spiral")
DEFAULT_ADAPTIVE_ROUTER_PRIOR = "row=0.40,column=0.22,snake=0.22,hilbert=0.08,spiral=0.08"


def _make_r_io_base(f, mode: str):
    """如果传入的是文件路径，则打开文件；如果已经是文件对象，则直接复用。"""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """读取 JSON 文件并返回 Python 对象。"""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def findAllFile(base):
    """遍历目录，依次产出后缀为 .json 的文件路径。"""
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.json'):
                fullname = os.path.join(root,f)
            yield fullname


# Alpaca 风格的提示词模板：有表格/输入时使用 prompt_input，否则使用 prompt_no_input。
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Question:\n{question}\n\n### Input:\n{input_seg}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    """模型路径相关参数。"""
    model_name_or_path: Optional[str] = field(default="/model/MiniCPM-2B-sft-bf16-llama-format")


@dataclass
class DataArguments:
    """数据路径和训练规模相关参数。"""
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_size: int = field(default=None, metadata={"help": "for calculate max steps."})
    gpu_size: int = field(default=None, metadata={"help": "for calculate max steps and for logging for calcuated intervel."})
    train_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Optional number of randomly sampled training examples to keep before preprocessing."},
    )
    validation_size: int = field(
        default=0,
        metadata={"help": "Number of preprocessed training examples to hold out as validation data."},
    )
    enable_table_blocks: bool = field(
        default=False,
        metadata={"help": "Whether to rewrite the table as self-contained local table blocks before 2D encoding."},
    )
    table_block_rows: int = field(
        default=3,
        metadata={"help": "Number of body rows in each local table block when enable_table_blocks is true."},
    )
    table_block_cols: int = field(
        default=4,
        metadata={"help": "Number of columns in each local table block when enable_table_blocks is true."},
    )
    table_header_rows: int = field(
        default=0,
        metadata={"help": "Number of table header rows to fuse for table blocks. Use 0 for automatic detection."},
    )
    table_blocks_dump_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to save the table block representation for all processed examples."},
    )
    table_read_mode: str = field(
        default="2d",
        metadata={"help": "Internal table read mode inside each block: 2d, row, column, snake, hilbert, spiral, or adaptive."},
    )
    adaptive_moe_start_epoch: float = field(
        default=1.0,
        metadata={"help": "For table_read_mode=adaptive, train with row-major order before this epoch, then enable MoE routing."},
    )
    adaptive_router_row_bias: float = field(
        default=2.0,
        metadata={"help": "Fallback row-expert bias when adaptive_router_prior is empty."},
    )
    adaptive_router_prior: Optional[str] = field(
        default=DEFAULT_ADAPTIVE_ROUTER_PRIOR,
        metadata={
            "help": (
                "Comma-separated initial adaptive router prior, e.g. "
                "'row=0.40,column=0.22,snake=0.22,hilbert=0.08,spiral=0.08'. "
                "Use an empty string to fall back to adaptive_router_row_bias."
            )
        },
    )
    adaptive_router_init_std: float = field(
        default=0.0,
        metadata={"help": "adaptive 路由器输出层权重初始化标准差；可用小随机值打破专家对称。"},
    )
    use_order_moe: bool = field(
        default=False,
        metadata={"help": "是否在 adaptive self-attention 内启用 row 共享专家 + Top-K 路由专家融合。"},
    )
    order_top_k: int = field(
        default=2,
        metadata={"help": "use_order_moe 为 true 时保留的 Top-K 路由 order 专家数量。"},
    )
    shared_order: str = field(
        default="row",
        metadata={"help": "共享 order 专家名称；当前实现默认使用 row。"},
    )
    routed_orders: str = field(
        default="column,snake,hilbert,spiral",
        metadata={"help": "参与 Top-K order MoE 路由的候选专家，使用逗号分隔。"},
    )
    order_router_entropy_coef: float = field(
        default=0.01,
        metadata={"help": "order MoE 路由熵损失系数；正值会鼓励 routed order 分布更尖锐。"},
    )
    order_router_temperature: float = field(
        default=0.5,
        metadata={"help": "order MoE routed logits softmax 温度；小于 1 会让 Top-K 前分布更尖锐。"},
    )
    order_router_init_std: float = field(
        default=1e-3,
        metadata={"help": "order MoE 独立 routed gate 输出层权重初始化标准差，用于打破独立 router 对称。"},
    )
    order_router_bias_init_std: float = field(
        default=0.1,
        metadata={"help": "order MoE 独立 routed gate bias 随机初始化标准差；order_router_bias 为空时生效。"},
    )
    order_router_bias: Optional[str] = field(
        default=None,
        metadata={"help": "order MoE routed gate 固定初始 bias logits；为空时使用随机 bias 初始化。"},
    )
    order_aux_scale: float = field(
        default=0.5,
        metadata={"help": "order MoE 辅助分支残差强度；row 主分支始终完整保留。"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """扩展 HuggingFace TrainingArguments，加入长上下文、Flash Attention 和 LoRA 开关。"""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    early_stopping_patience: int = field(
        default=0,
        metadata={"help": "Stop training after this many evals without metric improvement. 0 disables early stopping."},
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum metric improvement required by EarlyStoppingCallback."},
    )
    early_stopping_start_step: int = field(
        default=0,
        metadata={
            "help": (
                "Ignore validation metrics and checkpoint saves until global_step is greater than this value. "
                "Use 250 to start best-checkpoint/early-stop logic after the first 250 steps."
            )
        },
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """调整 tokenizer 和模型 embedding 的大小。

    注意：这是未优化版本，扩展后的 embedding 大小不一定能被 64 整除。
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        # 新增特殊 token 的 embedding 用原有 token embedding 的均值初始化，避免随机初始化过于突兀。
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def encode_and_insert_separators(table_array, tokenizer):
    """将二维表格编码为 token，并在行列之间插入特殊分隔 token。"""
    separator_col = [1425] # '▁|'
    separator_row = [48017] # '-'

    separator_row_end = [3] # '<SEP>'
    separator_col_end = [4] # '<CLS>'

    new_table = []
    
    for k, row in enumerate(table_array):
        new_row, new_separator = [], []
        for col in row:
            # 单元格内容先独立编码，再补列分隔符，保持表格结构可被位置编码识别。
            encoded_col = encode_no_warning(tokenizer, str(col))
            new_row.append(encoded_col)
            new_row.append(separator_col)  # 在每个编码后的列之间插入“|”语义的分隔符。

            # 最后一行使用列结束标记，其它行使用行分隔标记。
            new_separator.append(separator_col_end if k == len(table_array) - 1 else separator_row)
            new_separator.append(separator_col)
        new_row.append(separator_row_end)
        new_separator.append(separator_row_end)
        new_table.append(new_row)
        new_table.append(new_separator)
    return new_table


def encode_no_warning(tokenizer, text):
    """等价于 tokenizer.encode(...)，但关闭 max-length 警告日志。"""
    return tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]


def normalize_table_read_mode(table_read_mode):
    """Normalize aliases for the internal table read mode."""
    mode = str(table_read_mode or "2d").strip().lower()
    alias_map = {
        "2d": "2d",
        "2-d": "2d",
        "2_d": "2d",
        "two_d": "2d",
        "row": "row",
        "row-level": "row",
        "column": "column",
        "col": "column",
        "col-level": "column",
        "snake": "snake",
        "snake-level": "snake",
        "hilbert": "hilbert",
        "hilbert-order": "hilbert",
        "hilbert-level": "hilbert",
        "spiral": "spiral",
        "spiral-order": "spiral",
        "spiral-level": "spiral",
        "adaptive": "adaptive",
        "adaptive-order": "adaptive",
        "adaptive-moe": "adaptive",
    }
    normalized_mode = alias_map.get(mode)
    if normalized_mode is None:
        raise ValueError(
            f"Unsupported table_read_mode={table_read_mode}. "
            "Expected one of: 2d, row, column, snake, hilbert, spiral, adaptive."
        )
    return normalized_mode


def parse_adaptive_router_prior(prior_text, expert_names):
    """Parse a named expert prior and convert probabilities to bias logits."""
    if prior_text is None:
        return None
    prior_text = str(prior_text).strip()
    if not prior_text:
        return None

    values = {}
    for item in prior_text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid adaptive_router_prior item={item!r}. Expected format name=value."
            )
        name, value = item.split("=", 1)
        name = normalize_table_read_mode(name)
        if name not in expert_names:
            raise ValueError(
                f"Unknown adaptive router expert={name!r}. Expected one of: {', '.join(expert_names)}."
            )
        value = float(value)
        if value <= 0:
            raise ValueError("adaptive_router_prior values must be positive.")
        values[name] = value

    missing = [name for name in expert_names if name not in values]
    if missing:
        raise ValueError(
            f"adaptive_router_prior is missing experts: {', '.join(missing)}."
        )

    probs = np.array([values[name] for name in expert_names], dtype=np.float64)
    probs = probs / probs.sum()
    logits = np.log(probs)
    logits = logits - logits.min()
    return logits.astype(np.float32).tolist()


def parse_named_bias_values(bias_text, expert_names, field_name):
    """Parse direct named bias logits, preserving the order of expert_names."""
    if bias_text is None:
        return None
    bias_text = str(bias_text).strip()
    if not bias_text:
        return None

    values = {}
    for item in bias_text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid {field_name} item={item!r}. Expected format name=value.")
        name, value = item.split("=", 1)
        name = normalize_table_read_mode(name)
        if name not in expert_names:
            raise ValueError(f"Unknown {field_name} expert={name!r}. Expected one of: {', '.join(expert_names)}.")
        values[name] = float(value)

    missing = [name for name in expert_names if name not in values]
    if missing:
        raise ValueError(f"{field_name} is missing experts: {', '.join(missing)}.")

    return [float(values[name]) for name in expert_names]


def transpose_2d_rectangular(list_2d):
    """转置矩形二维列表，不依赖 numpy 的形状推断。"""
    if not list_2d:
        return []
    return [list(col) for col in zip(*list_2d)]


def next_power_of_two(value):
    """Return the smallest power of two that is greater than or equal to value."""
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def hilbert_d2xy(size, distance):
    """Convert a Hilbert curve distance to x/y coordinates for a power-of-two square."""
    x = 0
    y = 0
    step = 1
    t = int(distance)

    while step < size:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = step - 1 - x
                y = step - 1 - y
            x, y = y, x
        x += step * rx
        y += step * ry
        t //= 4
        step *= 2

    return x, y


def iter_hilbert_coordinates(height, width):
    """Yield rectangular coordinates by filtering a covering Hilbert square."""
    size = next_power_of_two(max(height, width))
    for distance in range(size * size):
        x, y = hilbert_d2xy(size, distance)
        if y < height and x < width:
            yield y, x


def iter_spiral_coordinates(height, width):
    """Yield rectangular coordinates in clockwise spiral order from the top-left corner."""
    top = 0
    bottom = height - 1
    left = 0
    right = width - 1

    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            yield top, j
        for i in range(top + 1, bottom + 1):
            yield i, right
        if top < bottom:
            for j in range(right - 1, left - 1, -1):
                yield bottom, j
        if left < right:
            for i in range(bottom - 1, top, -1):
                yield i, left
        top += 1
        bottom -= 1
        left += 1
        right -= 1


def parse_table_text(table_data):
    """将 [TAB] 后的文本表格解析为二维数组。"""
    if 'col:' in table_data and 'row 1:' in table_data:
        # 格式一：显式包含 col: 和 row 1:，先抽取表头，再逐行抽取单元格。
        headers_part, rows_part = table_data.split(' row 1:', 1)
        headers = headers_part.strip('col: ').split(' | ')
        headers = [header.strip(" |") if header.strip(" |") else 'None' for header in headers]
        rows_part = 'row 1:' + rows_part

        rows = rows_part.split(' [SEP]')
        data_rows = []
        for row in rows:
            if row:
                parts = row.strip().split(' | ')[1:]
                cleaned_parts = [part.strip(" |") if part.strip(" |") else 'None' for part in parts]
                data_rows.append(cleaned_parts)

        return [headers] + data_rows

    if 'col:' in table_data and 'row 1:' not in table_data:
        # 格式二：只有 col:，没有 row 1:，按 [SEP] 分割行。
        rows = table_data.split(" [SEP] ")
        headers = rows[0].split(" | ") if rows[0].endswith("|") else (rows[0] + " |").split(" | ")
        headers = [header.strip(" |") if header.strip(" |") else 'None' for header in headers][1:]
        data_rows = []
        for row in rows[1:]:
            if row:
                parts = row.strip("").split(' | ')
                cleaned_parts = [part.strip(" |") if part.strip(" |") else 'None' for part in parts]
                data_rows.append(cleaned_parts)

        return [headers] + data_rows

    # 格式三：普通 “|” 和 [SEP] 分隔的表格。
    rows = table_data.split(" [SEP] ")
    headers = rows[0].split(" | ") if rows[0].endswith("|") else (rows[0] + " |").split(" | ")
    headers = [header.strip(" |") if header.strip(" |") else 'None' for header in headers]
    data_rows = []
    for row in rows[1:]:
        if row:
            parts = row.strip("").split(' | ')
            cleaned_parts = [part.strip(" |") if part.strip(" |") else 'None' for part in parts]
            data_rows.append(cleaned_parts)

    return [headers] + data_rows


def is_rectangular_table(table_array):
    """检查二维表每行列数是否一致。"""
    if not table_array:
        return False
    expected_columns = len(table_array[0])
    return all(len(row) == expected_columns for row in table_array)


def normalize_table_cell(value):
    """将空单元统一成 None 字符串，便于后续构造自解释表头。"""
    text = str(value).strip()
    return text if text else 'None'


def is_number_like(value):
    """粗略判断一个单元是否更像数值型表体，而不是表头层。"""
    text = str(value).strip()
    if not text or text in {"-", "--", "None"}:
        return False
    return bool(re.fullmatch(r"[-+]?[$%]?\d[\d,]*(\.\d+)?%?", text))


def token_overlap(left, right):
    """计算两个表头文本的词集合重合度。"""
    left_tokens = set(re.findall(r"[a-z0-9]+", str(left).lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", str(right).lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def is_probable_extra_header_row(row, top_header):
    """判断首部连续行是否像额外表头层。"""
    if len(row) <= 1:
        return False

    first_cell_related = token_overlap(row[0], top_header[0]) >= 0.5
    non_key_cells = row[1:]
    numeric_count = sum(is_number_like(cell) for cell in non_key_cells)
    non_empty_count = sum(normalize_table_cell(cell) != 'None' for cell in non_key_cells)
    repeated_text_count = len(non_key_cells) - len({normalize_table_cell(cell).lower() for cell in non_key_cells})

    if numeric_count > 0:
        return False
    if first_cell_related and non_empty_count > 0:
        return True
    return repeated_text_count > 0 and non_empty_count >= max(1, len(non_key_cells) // 2)


def split_header_and_body_rows(table_array, header_rows=0):
    """拆分多层表头和表体；header_rows 为 0 时自动识别连续表头层。"""
    if not table_array:
        return [], []

    if header_rows and header_rows > 0:
        header_count = min(max(1, int(header_rows)), len(table_array))
        return table_array[:header_count], table_array[header_count:]

    header_count = 1
    max_auto_header_rows = min(len(table_array), 4)
    for row in table_array[1:max_auto_header_rows]:
        if is_probable_extra_header_row(row, table_array[0]):
            header_count += 1
        else:
            break

    return table_array[:header_count], table_array[header_count:]


def fuse_multilevel_headers(header_rows):
    """把多层列头融合成 H_i 的完整局部语义。"""
    if not header_rows:
        return []

    headers = []
    width = len(header_rows[0])
    for col_idx in range(width):
        parts = []
        seen = set()
        for row in header_rows:
            part = normalize_table_cell(row[col_idx] if col_idx < len(row) else 'None')
            key = part.lower()
            if part != 'None' and key not in seen:
                parts.append(part)
                seen.add(key)
        headers.append(" / ".join(parts) if parts else 'None')
    return headers


def build_self_explaining_table_blocks(table_array, block_rows, block_cols, header_rows=0, return_count=False):
    """将整表切成若干自解释局部子表；metadata 单独作为普通文本，不进入二维表。"""
    if not table_array:
        return ([], 0) if return_count else []

    block_rows = max(1, int(block_rows))
    block_cols = max(1, int(block_cols))
    header_row_list, body_rows = split_header_and_body_rows(table_array, header_rows)
    headers = fuse_multilevel_headers(header_row_list)
    blocks = []

    for row_start in range(0, len(body_rows), block_rows):
        row_end = min(row_start + block_rows, len(body_rows))
        block_row_id = row_start // block_rows + 1
        for col_start in range(0, len(headers), block_cols):
            col_end = min(col_start + block_cols, len(headers))
            block_col_id = col_start // block_cols + 1
            col_count = col_end - col_start

            metadata = (
                f"[BLOCK B_{block_row_id}_{block_col_id}]"
                f"[ROWSPAN R{row_start + 1}-R{row_end}]"
                f"[COLSPAN H{col_start + 1}-H{col_end}]"
            )
            block_table = []

            header_row = [
                f"[HEADERS] H{col_start + 1}={headers[col_start]}"
            ]
            for col_idx in range(col_start + 1, col_end):
                header_row.append(f"H{col_idx + 1}={headers[col_idx]}")
            block_table.append(header_row)

            for rid, row in enumerate(body_rows[row_start:row_end], start=row_start + 1):
                block_row = []
                for col_idx in range(col_start, col_end):
                    value = row[col_idx] if col_idx < len(row) and str(row[col_idx]).strip() else 'None'
                    prefix = f"[RID={rid}] " if col_idx == col_start else ""
                    block_row.append(f"{prefix}H{col_idx + 1}: {value}")
                block_table.append(block_row)

            blocks.append({
                "metadata": metadata,
                "block_table": block_table,
            })

    return (blocks, len(blocks)) if return_count else blocks


def iter_table_coordinates(height, width, table_read_mode):
    """Yield encoded-table coordinates in the requested traversal order."""
    if table_read_mode in {"2d", "row", "adaptive"}:
        for i in range(height):
            for j in range(width):
                yield i, j
        return

    if table_read_mode == "column":
        for j in range(width):
            for i in range(height):
                yield i, j
        return

    if table_read_mode == "snake":
        for i in range(height):
            # encode_and_insert_separators inserts one separator row after every
            # logical table row, so encoded rows 0/1 belong to logical row 0,
            # encoded rows 2/3 belong to logical row 1, and so on.
            logical_row = i // 2
            columns = range(width) if logical_row % 2 == 0 else range(width - 1, -1, -1)
            for j in columns:
                yield i, j
        return

    if table_read_mode == "hilbert":
        yield from iter_hilbert_coordinates(height, width)
        return

    if table_read_mode == "spiral":
        yield from iter_spiral_coordinates(height, width)
        return

    raise ValueError(f"Unsupported table_read_mode={table_read_mode}")


def build_coordinate_token_positions(new_table, start_position, table_read_mode):
    """Map each encoded-table cell coordinate to token positions in one order."""
    token_positions = {}
    current_position = start_position
    for i, j in iter_table_coordinates(len(new_table), len(new_table[0]), table_read_mode):
        item = new_table[i][j]
        item_start = current_position + 1
        item_end = item_start + len(item)
        token_positions[(i, j)] = list(range(item_start, item_end))
        current_position = item_end - 1
    return token_positions, current_position


def append_encoded_table_2d(new_table, input_ids, px, py, tx, ty, start_position):
    """Keep the original 2D position and dual-attention behavior unchanged."""
    current_position = start_position
    height = len(new_table)
    width = len(new_table[0])

    for i, row in enumerate(new_table):
        row_x = current_position + (width + 1) * (i + 1)
        for j, item in enumerate(row):
            row_y = current_position + (height + 1) * (j + 1)
            px.extend([row_x] * len(item))
            py.extend([row_y] * len(item))
            input_ids.extend(item)

    for row in new_table:
        for item in row:
            tx_count = len(tx)
            tx.extend(list(range(tx_count, tx_count + len(item))))

    transpose_new_table = transpose_2d_rectangular(new_table)
    ty_list, count = [], len(ty)
    for row in transpose_new_table:
        ty_list.append([])
        for item in row:
            ty_list[-1].append(list(range(count, count + len(item))))
            count += len(item)
    transpose_ty_list = transpose_2d_rectangular(ty_list)
    for row in transpose_ty_list:
        for item in row:
            ty.extend(item)

    return current_position + (width + 1) * (height + 1)


def append_encoded_table_linear(new_table, input_ids, px, py, tx, ty, start_position, table_read_mode):
    """Serialize the encoded table with a 1D traversal order."""
    current_position = start_position
    height = len(new_table)
    width = len(new_table[0])

    for i, j in iter_table_coordinates(height, width, table_read_mode):
        item = new_table[i][j]
        item_start = current_position + 1
        item_end = item_start + len(item)
        item_positions = list(range(item_start, item_end))

        input_ids.extend(item)
        px.extend(item_positions)
        py.extend(item_positions)

        # Non-2D modes use the same traversal for both attention branches.
        tx_count = len(tx)
        ty_count = len(ty)
        assert tx_count == ty_count
        tx.extend(list(range(tx_count, tx_count + len(item))))
        ty.extend(list(range(ty_count, ty_count + len(item))))
        current_position = item_end - 1

    return current_position


def append_encoded_table_adaptive(new_table, input_ids, px, py, adaptive_token_ids, start_position):
    """Serialize by row order while carrying all adaptive order-id channels."""
    row_positions, current_position = build_coordinate_token_positions(new_table, start_position, "row")
    order_positions = {
        mode: build_coordinate_token_positions(new_table, start_position, mode)[0]
        for mode in ADAPTIVE_TABLE_READ_MODES
    }

    for i, j in iter_table_coordinates(len(new_table), len(new_table[0]), "row"):
        item = new_table[i][j]
        input_ids.extend(item)
        px.extend(row_positions[(i, j)])
        py.extend(row_positions[(i, j)])
        for channel, mode in zip(adaptive_token_ids, ADAPTIVE_TABLE_READ_MODES):
            channel.extend(order_positions[mode][(i, j)])

    return current_position


def append_encoded_table(
    table_blocks,
    tokenizer,
    input_ids,
    px,
    py,
    tx,
    ty,
    start_position,
    table_read_mode="2d",
    adaptive_token_ids=None,
):
    """按块顺序追加 token；metadata 用一维位置，BLOCK_TABLE 用二维位置。"""
    current_position = start_position
    table_read_mode = normalize_table_read_mode(table_read_mode)

    for block_item in table_blocks:
        if isinstance(block_item, dict):
            metadata = block_item.get("metadata", "")
            table_block = block_item.get("block_table", block_item.get("table", []))
        else:
            metadata = ""
            table_block = block_item

        if metadata:
            metadata_en = encode_no_warning(tokenizer, metadata + "\n")
            meta_start = current_position + 1
            meta_end = meta_start + len(metadata_en)
            input_ids.extend(metadata_en)
            px.extend(list(range(meta_start, meta_end)))
            py.extend(list(range(meta_start, meta_end)))

            if table_read_mode == "adaptive":
                if adaptive_token_ids is None:
                    raise ValueError("adaptive_token_ids must be provided when table_read_mode='adaptive'.")
                for channel in adaptive_token_ids:
                    channel_count = len(channel)
                    channel.extend(list(range(channel_count, channel_count + len(metadata_en))))
            else:
                tx_count = len(tx)
                ty_count = len(ty)
                assert tx_count == ty_count
                tx.extend(list(range(tx_count, tx_count + len(metadata_en))))
                ty.extend(list(range(ty_count, ty_count + len(metadata_en))))
            current_position = meta_end - 1

        new_table = encode_and_insert_separators(table_block, tokenizer)

        if not is_rectangular_table(new_table):
            return None

        if table_read_mode == "adaptive":
            if adaptive_token_ids is None:
                raise ValueError("adaptive_token_ids must be provided when table_read_mode='adaptive'.")
            current_position = append_encoded_table_adaptive(
                new_table,
                input_ids,
                px,
                py,
                adaptive_token_ids,
                current_position,
            )
        elif table_read_mode == "2d":
            current_position = append_encoded_table_2d(
                new_table,
                input_ids,
                px,
                py,
                tx,
                ty,
                current_position,
            )
        else:
            current_position = append_encoded_table_linear(
                new_table,
                input_ids,
                px,
                py,
                tx,
                ty,
                current_position,
                table_read_mode,
            )

    return current_position


def dump_table_blocks(records, dump_path):
    """将切分后的 block 表示保存为 JSON，便于检查外层硬排序效果。"""
    if not dump_path:
        return
    dump_dir = os.path.dirname(os.path.abspath(dump_path))
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

tok_example_count = 0

class SupervisedDataset(Dataset):
    """监督微调数据集：读取原始样本，构造 input_ids、labels 和二维位置/顺序编码。"""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args=None,
        sample_seed: Optional[int] = None,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        train_sample_size = getattr(data_args, "train_sample_size", None)
        if train_sample_size is not None:
            train_sample_size = int(train_sample_size)
            if train_sample_size <= 0:
                raise ValueError("train_sample_size must be greater than 0 when provided.")
            original_size = len(list_data_dict)
            if train_sample_size < original_size:
                rng = random.Random(sample_seed)
                sampled_indices = rng.sample(range(original_size), train_sample_size)
                list_data_dict = [list_data_dict[idx] for idx in sampled_indices]
                logging.warning(
                    "Randomly sampled training data | original_size=%d sampled_size=%d seed=%s",
                    original_size,
                    train_sample_size,
                    sample_seed,
                )
            else:
                logging.warning(
                    "train_sample_size=%d >= dataset_size=%d, using the full dataset.",
                    train_sample_size,
                    original_size,
                )
        print(len(list_data_dict))
        enable_table_blocks = bool(getattr(data_args, "enable_table_blocks", False))
        table_block_rows = getattr(data_args, "table_block_rows", 3)
        table_block_cols = getattr(data_args, "table_block_cols", 4)
        table_header_rows = getattr(data_args, "table_header_rows", 0)
        table_blocks_dump_path = getattr(data_args, "table_blocks_dump_path", None)
        table_read_mode = normalize_table_read_mode(getattr(data_args, "table_read_mode", "2d"))
        logging.warning(
            "Table block preprocessing | enabled=%s block_rows=%s block_cols=%s header_rows=%s read_mode=%s dump_path=%s",
            enable_table_blocks,
            table_block_rows,
            table_block_cols,
            table_header_rows,
            table_read_mode,
            table_blocks_dump_path,
        )
        global tok_example_count

        input_ids_all = []
        labels_all = []
        token_ids_all = []
        position_ids_all = []
        problematic_indices = []
        skipped_non_rect_table = 0
        skipped_non_rect_new_table = 0
        skipped_too_long = 0
        skipped_exception = 0
        substart_all = []
        subend_all = []
        block_counts_all = []
        table_block_dump_records = [] if table_blocks_dump_path else None

        for idx, example in enumerate(list_data_dict):
            try:
                # 周期性输出 token 化进度，便于观察大数据集预处理是否仍在推进。
                tok_example_count += 1
                if tok_example_count % 2560 == 0:
                    logging.warning(f"tok_example_count: {tok_example_count}")
        
                # logging.warning("Formatting inputs...")
                prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
                # 根据样本是否包含 input_seg 选择不同提示词模板。
                source = prompt_input.format_map(example) if example.get("input_seg", "") != "" else prompt_no_input.format_map(example)
                target = f"{example['output']}"

                # 将 source 切成：表格前提示、表格正文、Response 前缀三部分。
                parts = re.split(r'(\[TAB\] )|(\n\n### Response)', source)
                parts = [part for part in parts if part is not None]

                part1 = parts[0] + parts[1]
                table_data = parts[2]
                part3 = parts[3] + parts[4]

                # Convert a table from text format to list format.
                table_array = parse_table_text(table_data)

                # Determine whether the table is a rectangle
                # 只有矩形表格才能稳定构造二维位置编码；非矩形样本直接跳过。
                if not is_rectangular_table(table_array):
                    skipped_non_rect_table += 1
                    problematic_indices.append(idx)
                    logging.warning(f"Skip non-rectangular table_array at index {idx}")
                    continue

                if enable_table_blocks:
                    table_blocks, block_count = build_self_explaining_table_blocks(
                        table_array,
                        table_block_rows,
                        table_block_cols,
                        table_header_rows,
                        return_count=True,
                    )
                else:
                    table_blocks = [table_array]
                    block_count = 1

                dump_record_idx = None
                if table_block_dump_records is not None:
                    header_row_list, body_rows_for_dump = split_header_and_body_rows(
                        table_array,
                        table_header_rows if enable_table_blocks else 1,
                    )
                    dump_record_idx = len(table_block_dump_records)
                    table_block_dump_records.append({
                        "idx": idx,
                        "status": "built",
                        "question": example.get("question", ""),
                        "output": example.get("output", ""),
                        "block_mode": enable_table_blocks,
                        "table_read_mode": table_read_mode,
                        "block_rows": table_block_rows if enable_table_blocks else None,
                        "block_cols": table_block_cols if enable_table_blocks else None,
                        "header_rows": len(header_row_list),
                        "body_rows": len(body_rows_for_dump),
                        "columns": len(table_array[0]) if table_array else 0,
                        "block_count": block_count,
                        "blocks": table_blocks,
                    })

                # Part I Encoded
                # 第一段：BOS + 表格前提示；普通文本部分使用一维递增位置。
                input_ids = [tokenizer.bos_token_id] + encode_no_warning(tokenizer, part1)
                l_part1 = len(input_ids)
                tx = list(range(l_part1))
                ty = list(range(l_part1))
                adaptive_token_ids = None
                if table_read_mode == "adaptive":
                    adaptive_token_ids = [list(range(l_part1)) for _ in ADAPTIVE_TABLE_READ_MODES]
                
                px = list(range(l_part1))
                py = list(range(l_part1))

                substart = input_ids[-4:]

                
                # Table Encoded
                # 第二段：表格内容。普通模式编码整表；block 模式逐块编码局部自解释子表。
                k_part3_start = append_encoded_table(
                    table_blocks,
                    tokenizer,
                    input_ids,
                    px,
                    py,
                    tx,
                    ty,
                    l_part1 - 1,
                    table_read_mode=table_read_mode,
                    adaptive_token_ids=adaptive_token_ids,
                )
                if k_part3_start is None:
                    skipped_non_rect_new_table += 1
                    problematic_indices.append(idx)
                    logging.warning(f"Skip non-rectangular encoded table at index {idx}")
                    continue

                # 第三段：Response 前缀 + 目标答案 + EOS。
                part3_target = part3 + target
                part3_target_en = encode_no_warning(tokenizer, part3_target) + [tokenizer.eos_token_id]
                input_ids.extend(part3_target_en)
                
                # 第三段恢复为普通一维递增 token 顺序。
                tx_count = len(tx)
                ty_count = len(ty)
                if table_read_mode == "adaptive":
                    for channel in adaptive_token_ids:
                        channel_count = len(channel)
                        channel.extend(list(range(channel_count, channel_count + len(part3_target_en))))
                else:
                    assert tx_count == ty_count
                    tx.extend(list(range(tx_count, tx_count + len(part3_target_en))))
                    ty.extend(list(range(ty_count, ty_count + len(part3_target_en))))


                # Part III Encoded

                # 第三段的二维位置也使用连续递增位置，与普通文本保持一致。
                k_part3_end = k_part3_start + len(part3_target_en)
                px.extend(list(range(k_part3_start, k_part3_end)))
                py.extend(list(range(k_part3_start, k_part3_end)))

                subend = part3_target_en[:4]

                # labels 只在 target 部分计算损失，prompt 和表格部分全部置为 IGNORE_INDEX。
                target_en = encode_no_warning(tokenizer, target) + [tokenizer.eos_token_id]
                target_len = len(target_en)
                labels = copy.deepcopy(input_ids)
                labels[:-target_len] = [IGNORE_INDEX] * (len(input_ids) - target_len)


                # 超过模型最大长度的样本跳过，避免位置编码和 batch padding 后出现异常。
                if len(input_ids) > tokenizer.model_max_length:
                    skipped_too_long += 1
                    problematic_indices.append(idx)
                    if dump_record_idx is not None:
                        table_block_dump_records[dump_record_idx]["status"] = "skipped_too_long"
                        table_block_dump_records[dump_record_idx]["input_token_length"] = len(input_ids)
                    continue
                    input_ids = input_ids[-tokenizer.model_max_length:]
                    labels = labels[-tokenizer.model_max_length:]
                    px = px[-tokenizer.model_max_length:]
                    py = py[-tokenizer.model_max_length:]
                    tx = tx[-tokenizer.model_max_length:]
                    ty = ty[-tokenizer.model_max_length:]

                # 将 x/y 两套位置或顺序编码拼接，collator 中再拆开并 padding。
                pi = np.concatenate([px, py])
                ti = np.concatenate(adaptive_token_ids) if table_read_mode == "adaptive" else np.concatenate([tx, ty])

                input_ids_all.append(torch.tensor(input_ids))
                labels_all.append(torch.tensor(labels))
                token_ids_all.append(torch.tensor(ti))
                position_ids_all.append(torch.tensor(pi))
                substart_all.append(torch.tensor(substart))
                subend_all.append(torch.tensor(subend))
                block_counts_all.append(block_count)
            
            except Exception as e:
                skipped_exception += 1
                problematic_indices.append(idx)
                logging.error(f"Error processing example at index {idx}: {str(e)}")
        
       

        print(len(input_ids_all))
        logging.warning(
            "Dataset build summary | total=%d kept=%d skipped=%d "
            "(too_long=%d, non_rect_table=%d, non_rect_new_table=%d, exception=%d)",
            len(list_data_dict),
            len(input_ids_all),
            skipped_too_long + skipped_non_rect_table + skipped_non_rect_new_table + skipped_exception,
            skipped_too_long,
            skipped_non_rect_table,
            skipped_non_rect_new_table,
            skipped_exception,
        )
        if block_counts_all:
            logging.warning(
                "Table block count summary | min=%d max=%d avg=%.2f",
                min(block_counts_all),
                max(block_counts_all),
                sum(block_counts_all) / len(block_counts_all),
            )
        if table_block_dump_records is not None:
            dump_table_blocks(table_block_dump_records, table_blocks_dump_path)
            logging.warning(
                "Saved table block dump | path=%s records=%d",
                table_blocks_dump_path,
                len(table_block_dump_records),
            )
        self.input_ids = input_ids_all
        self.labels = labels_all
        self.token_ids = token_ids_all
        self.position_ids = position_ids_all
        self.substart = substart_all
        self.subend = subend_all
        self.block_counts = block_counts_all

    def __len__(self):
        """返回可用样本数量。"""
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """返回单条样本的模型输入。"""
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], token_ids=self.token_ids[i], position_ids=self.position_ids[i], substart=self.substart[i], subend=self.subend[i])

    

@dataclass
class DataCollatorForSupervisedDataset(object):
    """监督微调的 batch 拼接器：对变长字段做 padding，并生成 attention_mask。"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # logging.warning(f"instances: {instances}")
        # 从样本列表中按字段拆出各类张量。
        input_ids, labels, token_ids, position_ids, substart, subend = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "token_ids", "position_ids", "substart", "subend"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        substart = torch.nn.utils.rnn.pad_sequence(substart, batch_first=True, padding_value=IGNORE_INDEX)
        subend = torch.nn.utils.rnn.pad_sequence(subend, batch_first=True, padding_value=IGNORE_INDEX)
        # token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=0)

        px_list = []
        py_list = []

        # position_ids 前半段是 px，后半段是 py；先拆开，分别 padding 后再拼回去。
        for pid in position_ids:
            s = len(pid) // 2
            px = pid[:s]
            py = pid[s:]
            px_list.append(px)
            py_list.append(py)

        px_padded = self.efficient_custom_pad_sequences(px_list)
        py_padded = self.efficient_custom_pad_sequences(py_list)
        position_ids = torch.cat((px_padded, py_padded), dim=-1)



        raw_token_ids = token_ids
        tx_list = []
        ty_list = []

        # token_ids 前半段是 tx，后半段是 ty；处理方式同 position_ids。
        for tid in token_ids:
            s = len(tid) // 2
            tx = tid[:s]
            ty = tid[s:]
            tx_list.append(tx)
            ty_list.append(ty)

        tx_padded = self.efficient_custom_pad_sequences(tx_list)
        ty_padded = self.efficient_custom_pad_sequences(ty_list)

        token_ids = torch.cat((tx_padded, ty_padded), dim=-1)

        token_id_channel_lists = []
        for tid, input_id in zip(raw_token_ids, input_ids):
            sample_len = int(input_id.ne(self.tokenizer.pad_token_id).sum().item())
            channel_count = len(tid) // sample_len
            if channel_count * sample_len != len(tid):
                raise ValueError(
                    f"token_ids length={len(tid)} is not divisible by sample_len={sample_len}."
                )
            while len(token_id_channel_lists) < channel_count:
                token_id_channel_lists.append([])
            for channel_idx in range(channel_count):
                start = channel_idx * sample_len
                end = start + sample_len
                token_id_channel_lists[channel_idx].append(tid[start:end])

        token_ids = torch.cat(
            [self.efficient_custom_pad_sequences(channel_list) for channel_list in token_id_channel_lists],
            dim=-1,
        )


        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            token_ids=token_ids,
            position_ids=position_ids,
            substart=substart, 
            subend=subend,
        )
    
    def efficient_custom_pad_sequences(self, sequence_list):
        """用连续递增值补齐位置序列，避免 padding 位置重复使用最后一个位置 id。"""
        tensors = [seq if torch.is_tensor(seq) else torch.tensor(seq) for seq in sequence_list]
        max_len = max(t.size(0) for t in tensors)

        pad_sizes = [max_len - t.size(0) for t in tensors]

        max_pad_size = max(pad_sizes)
        increment_ranges = torch.arange(1, max_pad_size + 1).unsqueeze(0)

        padded_tensors = []
        for tensor, pad_size in zip(tensors, pad_sizes):
            if pad_size > 0:
                # padding 的位置 id 从当前序列最后一个值继续递增。
                padded_tensor = torch.cat([tensor, tensor[-1] + increment_ranges[:, :pad_size].squeeze(0)])
            else:
                padded_tensor = tensor
            padded_tensors.append(padded_tensor)
        
        padded_tensor_batch = torch.stack(padded_tensors)

        return padded_tensor_batch
    def efficient_custom_pad_sequences(self, sequence_list):
        """Pad position/order id sequences with fresh increasing ids."""
        tensors = [seq if torch.is_tensor(seq) else torch.tensor(seq) for seq in sequence_list]
        max_len = max(t.size(0) for t in tensors)
        pad_sizes = [max_len - t.size(0) for t in tensors]
        max_pad_size = max(pad_sizes)
        increment_ranges = torch.arange(1, max_pad_size + 1).unsqueeze(0)

        padded_tensors = []
        for tensor, pad_size in zip(tensors, pad_sizes):
            if pad_size > 0:
                padded_tensor = torch.cat([tensor, tensor.max() + increment_ranges[:, :pad_size].squeeze(0)])
            else:
                padded_tensor = tensor
            padded_tensors.append(padded_tensor)

        return torch.stack(padded_tensors)


class ConsoleProgressCallback(TrainerCallback):
    """将紧凑的训练进度打印到 stdout，便于前台观察。"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 只在主进程打印，避免分布式训练时重复输出。
        if not state.is_local_process_zero or not logs:
            return
        loss = logs.get("loss")
        learning_rate = logs.get("learning_rate")
        epoch = logs.get("epoch")
        print(
            f"[train] step={state.global_step} "
            f"epoch={epoch} loss={loss} lr={learning_rate}",
            flush=True,
        )


class AdaptiveRouterStatsCallback(TrainerCallback):
    def __init__(self, expert_names):
        self.expert_names = list(expert_names)
        self._last_grad_stats = None
        self._grad_hook_handles = []
        self._hook_grad_sq = 0.0
        self._hook_grad_tensor_count = 0

    def _adaptive_logging_enabled(self, model):
        config = getattr(model, "config", None)
        if config is not None and getattr(config, "adaptive_router_logging", False):
            return True
        base_model = getattr(model, "base_model", None)
        base_config = getattr(base_model, "config", None)
        return bool(base_config is not None and getattr(base_config, "adaptive_router_logging", False))

    def _use_order_moe(self, model):
        config = getattr(model, "config", None)
        if config is not None and getattr(config, "use_order_moe", False):
            return True
        base_model = getattr(model, "base_model", None)
        base_config = getattr(base_model, "config", None)
        return bool(base_config is not None and getattr(base_config, "use_order_moe", False))

    def _get_order_moe_config(self, model):
        config = getattr(model, "config", None)
        if config is None or not getattr(config, "use_order_moe", False):
            base_model = getattr(model, "base_model", None)
            config = getattr(base_model, "config", None)

        shared_order = "row"
        routed_orders = list(self.expert_names[1:])
        if config is not None:
            shared_order = str(getattr(config, "shared_order", shared_order))
            routed_orders = getattr(config, "routed_orders", routed_orders)
            if isinstance(routed_orders, str):
                routed_orders = [name.strip() for name in routed_orders.split(",") if name.strip()]
            else:
                routed_orders = [str(name) for name in routed_orders]
        return shared_order, routed_orders

    def _flatten_gate_modules(self, gate_modules):
        gates = []
        for gate in gate_modules:
            if gate is None:
                continue
            if isinstance(gate, torch.nn.ModuleDict):
                gates.extend([subgate for subgate in gate.values() if subgate is not None])
            elif isinstance(gate, torch.nn.ModuleList):
                gates.extend([subgate for subgate in gate if subgate is not None])
            else:
                gates.append(gate)
        return gates

    def _collect_gate_stats(self, model, include_grad=False):
        gate_count = 0
        trainable_count = 0
        weight_sq = 0.0
        bias_sum = None
        bias_count = 0
        grad_sq = 0.0
        grad_tensor_count = 0
        missing_grad_count = 0

        for module in model.modules():
            if self._use_order_moe(model):
                gate_modules = [
                    getattr(module, "gate_1", None),
                    getattr(module, "gate_2", None),
                    getattr(module, "row_gate_proj", None),
                    getattr(module, "order_router_gate_3", None),
                ]
            else:
                gate_modules = [getattr(module, "adaptive_gate_3", None)]
            gate_modules = self._flatten_gate_modules(gate_modules)
            if not gate_modules:
                continue
            gate_count += 1
            params = []
            for gate in gate_modules:
                params.append(gate.weight)
                if gate.bias is not None:
                    params.append(gate.bias)
            if any(param.requires_grad for param in params):
                trainable_count += 1

            for gate in gate_modules:
                weight_sq += float(gate.weight.detach().float().pow(2).sum().cpu())
            if self._use_order_moe(model):
                row_gate = getattr(module, "row_gate_proj", None)
                routed_gates = self._flatten_gate_modules([getattr(module, "order_router_gate_3", None)])
                if (
                    row_gate is not None
                    and row_gate.bias is not None
                    and routed_gates
                    and all(gate.bias is not None for gate in routed_gates)
                ):
                    bias = torch.cat(
                        [row_gate.bias.detach().float().cpu()]
                        + [gate.bias.detach().float().cpu() for gate in routed_gates]
                    )
                    bias_sum = bias.clone() if bias_sum is None else bias_sum + bias
                    bias_count += 1
            else:
                gate = gate_modules[0]
                if gate.bias is not None:
                    bias = gate.bias.detach().float().cpu()
                    bias_sum = bias.clone() if bias_sum is None else bias_sum + bias
                    bias_count += 1

            if include_grad:
                for param in params:
                    if not param.requires_grad:
                        continue
                    if param.grad is None:
                        missing_grad_count += 1
                        continue
                    grad_sq += float(param.grad.detach().float().pow(2).sum().cpu())
                    grad_tensor_count += 1

        bias_mean = (bias_sum / bias_count).tolist() if bias_count else None
        return {
            "gate_count": gate_count,
            "trainable_count": trainable_count,
            "weight_norm": math.sqrt(weight_sq),
            "bias_mean": bias_mean,
            "grad_norm": math.sqrt(grad_sq) if include_grad else None,
            "grad_tensor_count": grad_tensor_count,
            "missing_grad_count": missing_grad_count,
        }

    def _reset_hook_grad_accumulator(self):
        self._hook_grad_sq = 0.0
        self._hook_grad_tensor_count = 0

    def _make_grad_hook(self):
        def hook(grad):
            if grad is None:
                return
            self._hook_grad_sq += float(grad.detach().float().pow(2).sum().cpu())
            self._hook_grad_tensor_count += 1

        return hook

    def _register_gate_grad_hooks(self, model):
        if self._grad_hook_handles:
            return
        for module in model.modules():
            if self._use_order_moe(model):
                gate_modules = [
                    getattr(module, "gate_1", None),
                    getattr(module, "gate_2", None),
                    getattr(module, "row_gate_proj", None),
                    getattr(module, "order_router_gate_3", None),
                ]
            else:
                gate_modules = [getattr(module, "adaptive_gate_3", None)]
            gate_modules = self._flatten_gate_modules(gate_modules)
            if not gate_modules:
                continue
            for gate in gate_modules:
                for param in (gate.weight, gate.bias):
                    if param is not None and param.requires_grad:
                        self._grad_hook_handles.append(param.register_hook(self._make_grad_hook()))

    def _consume_hook_grad_stats(self):
        if self._hook_grad_tensor_count == 0:
            return None
        stats = {
            "grad_norm": math.sqrt(self._hook_grad_sq),
            "grad_tensor_count": self._hook_grad_tensor_count,
            "missing_grad_count": 0,
        }
        self._reset_hook_grad_accumulator()
        return stats

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None or not self._adaptive_logging_enabled(model):
            return
        self._register_gate_grad_hooks(model)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None or not self._adaptive_logging_enabled(model):
            return
        self._last_grad_stats = self._collect_gate_stats(model, include_grad=True)

    def on_train_end(self, args, state, control, **kwargs):
        for handle in self._grad_hook_handles:
            handle.remove()
        self._grad_hook_handles = []

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if not state.is_local_process_zero or model is None:
            return

        if self._use_order_moe(model):
            total_count = 0
            row_gate_sum = None
            routed_score_sum = None
            topk_sum = None
            sparse_gate_sum = None
            entropy_sum = 0.0
            max_score_sum = 0.0
            table_total_count = 0
            table_row_gate_sum = None
            table_routed_score_sum = None
            table_topk_sum = None
            table_sparse_gate_sum = None
            table_entropy_sum = 0.0
            table_max_score_sum = 0.0

            for module in model.modules():
                pop_stats = getattr(module, "pop_order_moe_stats", None)
                if pop_stats is None:
                    continue
                stats = pop_stats()
                if stats is None:
                    continue
                count = stats["count"]
                total_count += count
                row_gate_sum = stats["row_gate_sum"] if row_gate_sum is None else row_gate_sum + stats["row_gate_sum"]
                routed_score_sum = (
                    stats["routed_score_sum"]
                    if routed_score_sum is None
                    else routed_score_sum + stats["routed_score_sum"]
                )
                topk_sum = stats["topk_sum"] if topk_sum is None else topk_sum + stats["topk_sum"]
                sparse_gate_sum = (
                    stats["sparse_gate_sum"] if sparse_gate_sum is None else sparse_gate_sum + stats["sparse_gate_sum"]
                )
                entropy_sum += float(stats["entropy_sum"].item())
                max_score_sum += float(stats["max_score_sum"].item())

                table_count = stats.get("table_count", 0)
                if table_count:
                    table_total_count += table_count
                    table_row_gate_sum = (
                        stats["table_row_gate_sum"]
                        if table_row_gate_sum is None
                        else table_row_gate_sum + stats["table_row_gate_sum"]
                    )
                    table_routed_score_sum = (
                        stats["table_routed_score_sum"]
                        if table_routed_score_sum is None
                        else table_routed_score_sum + stats["table_routed_score_sum"]
                    )
                    table_topk_sum = (
                        stats["table_topk_sum"] if table_topk_sum is None else table_topk_sum + stats["table_topk_sum"]
                    )
                    table_sparse_gate_sum = (
                        stats["table_sparse_gate_sum"]
                        if table_sparse_gate_sum is None
                        else table_sparse_gate_sum + stats["table_sparse_gate_sum"]
                    )
                    table_entropy_sum += float(stats["table_entropy_sum"].item())
                    table_max_score_sum += float(stats["table_max_score_sum"].item())

            if total_count > 0 and row_gate_sum is not None:
                shared_order, routed_names = self._get_order_moe_config(model)

                def format_routed(values):
                    return ", ".join(
                        f"{name}={value:.6f}"
                        for name, value in zip(routed_names, values)
                    )

                row_gate = float((row_gate_sum / total_count).item())
                routed_score = (routed_score_sum / total_count).tolist()
                topk = (topk_sum / total_count).tolist()
                sparse_gate = (sparse_gate_sum / total_count).tolist()
                entropy = entropy_sum / total_count
                max_score = max_score_sum / total_count
                print(
                    f"[order-moe] step={state.global_step} "
                    f"shared_order={shared_order} "
                    f"routed_orders={','.join(routed_names)} "
                    f"shared_probe({shared_order})={row_gate:.6f} "
                    f"routed_score({format_routed(routed_score)}) "
                    f"topk({format_routed(topk)}) "
                    f"sparse_gate({format_routed(sparse_gate)}) "
                    f"entropy={entropy:.6f} "
                    f"max_score={max_score:.6f}",
                    flush=True,
                )

                if table_total_count > 0 and table_row_gate_sum is not None:
                    table_row_gate = float((table_row_gate_sum / table_total_count).item())
                    table_routed_score = (table_routed_score_sum / table_total_count).tolist()
                    table_topk = (table_topk_sum / table_total_count).tolist()
                    table_sparse_gate = (table_sparse_gate_sum / table_total_count).tolist()
                    table_entropy = table_entropy_sum / table_total_count
                    table_max_score = table_max_score_sum / table_total_count
                    print(
                        f"[order-moe-table] step={state.global_step} "
                        f"shared_order={shared_order} "
                        f"routed_orders={','.join(routed_names)} "
                        f"shared_probe({shared_order})={table_row_gate:.6f} "
                        f"routed_score({format_routed(table_routed_score)}) "
                        f"topk({format_routed(table_topk)}) "
                        f"sparse_gate({format_routed(table_sparse_gate)}) "
                        f"entropy={table_entropy:.6f} "
                        f"max_score={table_max_score:.6f}",
                        flush=True,
                    )

            gate_stats = self._collect_gate_stats(model, include_grad=False)
            hook_grad_stats = self._consume_hook_grad_stats()
            grad_stats = hook_grad_stats or self._last_grad_stats
            self._last_grad_stats = None
            bias_text = "none"
            if gate_stats["bias_mean"] is not None:
                bias_values = gate_stats["bias_mean"]
                shared_order, routed_names = self._get_order_moe_config(model)
                bias_text = "shared_probe({})={:.6f}, routed({})".format(
                    shared_order,
                    bias_values[0],
                    ", ".join(
                        f"{name}={value:.6f}"
                        for name, value in zip(routed_names, bias_values[1:])
                    ),
                )
            grad_text = "unavailable"
            if grad_stats is not None:
                grad_text = (
                    f"{grad_stats['grad_norm']:.6e} "
                    f"tensors={grad_stats['grad_tensor_count']} "
                    f"missing={grad_stats['missing_grad_count']}"
                )
            print(
                f"[order-moe-grad] step={state.global_step} "
                f"gates={gate_stats['gate_count']} "
                f"trainable={gate_stats['trainable_count']} "
                f"weight_norm={gate_stats['weight_norm']:.6e} "
                f"bias({bias_text}) "
                f"grad_norm={grad_text}",
                flush=True,
            )
            return

        total_count = 0
        prob_sum = None
        top1_sum = None
        entropy_sum = 0.0
        table_total_count = 0
        table_prob_sum = None
        table_top1_sum = None
        table_entropy_sum = 0.0
        for module in model.modules():
            pop_stats = getattr(module, "pop_adaptive_router_stats", None)
            if pop_stats is None:
                continue
            stats = pop_stats()
            if stats is None:
                continue
            count = stats["count"]
            total_count += count
            prob_sum = stats["prob_sum"] if prob_sum is None else prob_sum + stats["prob_sum"]
            top1_sum = stats["top1_sum"] if top1_sum is None else top1_sum + stats["top1_sum"]
            entropy_sum += float(stats["entropy_sum"].item())
            table_count = stats.get("table_count", 0)
            if table_count:
                table_total_count += table_count
                table_prob_sum = (
                    stats["table_prob_sum"] if table_prob_sum is None else table_prob_sum + stats["table_prob_sum"]
                )
                table_top1_sum = (
                    stats["table_top1_sum"] if table_top1_sum is None else table_top1_sum + stats["table_top1_sum"]
                )
                table_entropy_sum += float(stats["table_entropy_sum"].item())

        if total_count == 0 or prob_sum is None:
            return

        probs = (prob_sum / total_count).tolist()
        top1 = (top1_sum / total_count).tolist()
        entropy = entropy_sum / total_count

        def format_values(values):
            return ", ".join(
                f"{name}={value:.6f}"
                for name, value in zip(self.expert_names, values)
            )

        print(
            f"[adaptive-router] step={state.global_step} "
            f"prob({format_values(probs)}) "
            f"top1({format_values(top1)}) "
            f"entropy={entropy:.3f}",
            flush=True,
        )

        if table_total_count > 0 and table_prob_sum is not None:
            table_probs = (table_prob_sum / table_total_count).tolist()
            table_top1 = (table_top1_sum / table_total_count).tolist()
            table_entropy = table_entropy_sum / table_total_count
            print(
                f"[adaptive-router-table] step={state.global_step} "
                f"prob({format_values(table_probs)}) "
                f"top1({format_values(table_top1)}) "
                f"entropy={table_entropy:.3f}",
                flush=True,
            )

        gate_stats = self._collect_gate_stats(model, include_grad=False)
        hook_grad_stats = self._consume_hook_grad_stats()
        grad_stats = hook_grad_stats or self._last_grad_stats
        self._last_grad_stats = None
        bias_text = "none"
        if gate_stats["bias_mean"] is not None:
            bias_text = format_values(gate_stats["bias_mean"])
        grad_text = "unavailable"
        if grad_stats is not None:
            grad_text = (
                f"{grad_stats['grad_norm']:.6e} "
                f"tensors={grad_stats['grad_tensor_count']} "
                f"missing={grad_stats['missing_grad_count']}"
            )
        print(
            f"[adaptive-router-grad] step={state.global_step} "
            f"gates={gate_stats['gate_count']} "
            f"trainable={gate_stats['trainable_count']} "
            f"weight_norm={gate_stats['weight_norm']:.6e} "
            f"bias({bias_text}) "
            f"grad_norm={grad_text}",
            flush=True,
        )


class HeartbeatCallback(TrainerCallback):
    """即使 step 日志延迟或被抑制，也周期性输出存活日志。"""

    def __init__(self, interval_sec=60):
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = None
        self._state = None
        self._start_time = None

    def _run(self):
        """后台线程：按固定间隔打印训练是否仍在运行。"""
        while not self._stop_event.wait(self.interval_sec):
            if self._state is None:
                continue
            elapsed = int(time.time() - self._start_time) if self._start_time else 0
            step = self._state.global_step
            max_steps = self._state.max_steps
            if max_steps and max_steps > 0:
                pct = 100.0 * step / max_steps
                print(
                    f"[heartbeat] alive elapsed={elapsed}s step={step}/{max_steps} ({pct:.2f}%)",
                    flush=True,
                )
            else:
                print(f"[heartbeat] alive elapsed={elapsed}s step={step}", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        # 训练开始时启动后台心跳线程。
        if not state.is_local_process_zero:
            return
        self._state = state
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束时停止心跳线程并输出总耗时。
        if not state.is_local_process_zero:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        elapsed = int(time.time() - self._start_time) if self._start_time else 0
        print(f"[heartbeat] training finished elapsed={elapsed}s", flush=True)


class SuppressSaveBeforeStepCallback(TrainerCallback):
    """Prevent checkpoint writes until global_step is greater than start_step."""

    def __init__(self, start_step=0):
        self.start_step = int(start_step or 0)
        self._printed = False

    def _suppress_if_needed(self, state, control):
        if self.start_step <= 0 or state.global_step > self.start_step:
            return control
        if control.should_save and state.is_local_process_zero and not self._printed:
            print(
                f"[early-stop-warmup] suppress checkpoint save at step={state.global_step}; "
                f"best-checkpoint logic starts after step>{self.start_step}",
                flush=True,
            )
            self._printed = True
        control.should_save = False
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return self._suppress_if_needed(state, control)

    def on_epoch_end(self, args, state, control, **kwargs):
        return self._suppress_if_needed(state, control)


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    """Run EarlyStoppingCallback only after global_step is greater than start_step."""

    def __init__(self, early_stopping_patience=1, early_stopping_threshold=0.0, start_step=0):
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
        self.start_step = int(start_step or 0)
        self._printed = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.start_step > 0 and state.global_step <= self.start_step:
            self.early_stopping_patience_counter = 0
            control.should_training_stop = False
            if state.is_local_process_zero and not self._printed:
                print(
                    f"[early-stop-warmup] ignore validation metrics at step={state.global_step}; "
                    f"early stopping starts after step>{self.start_step}",
                    flush=True,
                )
                self._printed = True
            return control
        return super().on_evaluate(args, state, control, metrics=metrics, **kwargs)


class AdaptiveMoeWarmupCallback(TrainerCallback):
    """Use row-major warmup before enabling adaptive MoE routing."""

    def __init__(self, requested_mode, start_epoch):
        self.requested_mode = requested_mode
        self.start_epoch = float(start_epoch or 0)
        self._last_mode = None

    def _set_table_read_mode(self, model, state, force_final=False):
        if self.requested_mode != "adaptive":
            return

        epoch = 0.0 if state.epoch is None else float(state.epoch)
        active_mode = "adaptive" if force_final or epoch >= self.start_epoch else "row"

        if getattr(model, "config", None) is not None:
            model.config.table_read_mode = active_mode
        base_model = getattr(model, "base_model", None)
        if base_model is not None and getattr(base_model, "config", None) is not None:
            base_model.config.table_read_mode = active_mode

        if active_mode != self._last_mode and state.is_local_process_zero:
            print(
                f"[adaptive] epoch={epoch:.4f} table_read_mode={active_mode} "
                f"(moe_start_epoch={self.start_epoch})",
                flush=True,
            )
            self._last_mode = active_mode

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._set_table_read_mode(model, state)

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        self._set_table_read_mode(model, state)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self._set_table_read_mode(model, state)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._set_table_read_mode(model, state, force_final=True)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args=None) -> Dict:
    """构建监督微调所需的数据集和 collator。"""
    sample_seed = getattr(training_args, "seed", None) if training_args is not None else None
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        sample_seed=sample_seed,
    )
    eval_dataset = None
    validation_size = int(getattr(data_args, "validation_size", 0) or 0)
    if validation_size > 0:
        dataset_size = len(train_dataset)
        if validation_size >= dataset_size:
            raise ValueError(
                f"validation_size={validation_size} must be smaller than the preprocessed dataset size={dataset_size}."
            )
        split_seed = int(sample_seed if sample_seed is not None else 0)
        split_generator = torch.Generator().manual_seed(split_seed)
        train_size = dataset_size - validation_size
        train_dataset, eval_dataset = random_split(
            train_dataset,
            [train_size, validation_size],
            generator=split_generator,
        )
        logging.warning(
            "Validation split enabled | train_size=%d validation_size=%d seed=%s",
            train_size,
            validation_size,
            split_seed,
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    """训练入口：解析参数、加载模型/tokenizer、构建数据并启动 Trainer。"""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    table_read_mode = normalize_table_read_mode(getattr(data_args, "table_read_mode", "2d"))
    adaptive_moe_start_epoch = float(getattr(data_args, "adaptive_moe_start_epoch", 1.0))

    # Set RoPE scaling factor
    # 根据目标最大长度调整 RoPE scaling，以支持超过原始上下文长度的训练。
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    # 加载 tokenizer，并设置右侧 padding，适配自回归语言模型训练。
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        # 如果原 tokenizer 没有 pad token，则补一个默认 pad token。
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)

    # 保留自定义字段（token_ids、position_ids、substart、subend），否则 Trainer 会自动丢弃。
    training_args.remove_unused_columns = False
    training_args.report_to = []
    training_args.disable_tqdm = False
    training_args.logging_first_step = True
    training_args.logging_steps = 1
    config.use_cache = False
   
    # 根据环境自动启用 Flash Attention 2 标记。
    config._flash_attn_2_enabled = bool(hf_is_flash_attn_2_available())
    logging.warning(f"flash_attn_2_enabled={config._flash_attn_2_enabled}")
    config.output_loss = True
    config.pad_token_id = 0

    config.lamda = 1
    config.adaptive_expert_nums = len(ADAPTIVE_TABLE_READ_MODES)
    config.adaptive_expert_names = list(ADAPTIVE_TABLE_READ_MODES)
    config.adaptive_residual_scale = 1.0
    config.adaptive_router_row_bias = float(getattr(data_args, "adaptive_router_row_bias", 2.0))
    config.adaptive_router_prior = parse_adaptive_router_prior(
        getattr(data_args, "adaptive_router_prior", DEFAULT_ADAPTIVE_ROUTER_PRIOR),
        ADAPTIVE_TABLE_READ_MODES,
    )
    config.adaptive_router_init_std = float(getattr(data_args, "adaptive_router_init_std", 0.0))
    config.use_order_moe = bool(getattr(data_args, "use_order_moe", False))
    config.order_top_k = int(getattr(data_args, "order_top_k", 2))
    config.order_router_entropy_coef = float(getattr(data_args, "order_router_entropy_coef", 0.01))
    config.order_router_temperature = float(getattr(data_args, "order_router_temperature", 0.5))
    config.order_router_init_std = float(getattr(data_args, "order_router_init_std", 1e-3))
    config.order_router_bias_init_std = float(getattr(data_args, "order_router_bias_init_std", 0.1))
    config.order_aux_scale = float(getattr(data_args, "order_aux_scale", 0.5))
    config.shared_order = normalize_table_read_mode(getattr(data_args, "shared_order", "row"))
    routed_orders_text = getattr(data_args, "routed_orders", "column,snake,hilbert,spiral")
    config.routed_orders = [
        normalize_table_read_mode(name)
        for name in str(routed_orders_text).split(",")
        if name.strip()
    ]
    config.order_router_bias = parse_named_bias_values(
        getattr(data_args, "order_router_bias", None),
        config.routed_orders,
        "order_router_bias",
    )
    logging.warning(
        "Order MoE config | enabled=%s shared_order=%s routed_orders=%s top_k=%s aux_scale=%s",
        config.use_order_moe,
        config.shared_order,
        ",".join(config.routed_orders),
        config.order_top_k,
        config.order_aux_scale,
    )
    config.adaptive_router_logging = table_read_mode == "adaptive"
    config.table_read_mode = (
        "row" if table_read_mode == "adaptive" and adaptive_moe_start_epoch > 0 else table_read_mode
    )
    config.adaptive_moe_target_mode = table_read_mode
    config.adaptive_moe_start_epoch = adaptive_moe_start_epoch

    # 加载本项目改造过的 LlamaForCausalLM，支持额外的二维位置/顺序输入。
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
    )
    if hasattr(model, "_init_adaptive_router_bias"):
        model._init_adaptive_router_bias()

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if training_args.low_rank_training:
        # 使用 LoRA 只训练注意力投影层的低秩增量，降低显存和训练成本。
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        # 除 LoRA 权重外，额外放开 trainable_params 指定的参数（默认 embed,norm）。
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]
        if bool(getattr(data_args, "use_order_moe", False)):
            order_moe_trainable = ("gate_1", "gate_2", "order_router_gate_3", "row_gate_proj")
            [p.requires_grad_() for n, p in model.named_parameters() if any(k in n for k in order_moe_trainable)]
        base_model = getattr(model, "base_model", None)
        if base_model is not None and hasattr(base_model, "_init_adaptive_router_bias"):
            base_model._init_adaptive_router_bias()

    model.enable_input_require_grads()     # 梯度检查点需要输入可求梯度。
    model.gradient_checkpointing_enable()  # 启用梯度检查点以节省显存。

    logging.warning(f"data_module: {data_module}")
    logging.warning(
        "Training monitor enabled | logging_steps=%s, gradient_accumulation_steps=%s",
        training_args.logging_steps,
        training_args.gradient_accumulation_steps,
    )
    
    # HuggingFace Trainer 负责训练循环、日志、保存状态和模型。
    callbacks = [
        ConsoleProgressCallback,
        AdaptiveRouterStatsCallback(ADAPTIVE_TABLE_READ_MODES),
        HeartbeatCallback(interval_sec=60),
        AdaptiveMoeWarmupCallback(table_read_mode, adaptive_moe_start_epoch),
    ]
    early_stopping_start_step = int(getattr(training_args, "early_stopping_start_step", 0) or 0)
    if early_stopping_start_step > 0:
        callbacks.append(SuppressSaveBeforeStepCallback(start_step=early_stopping_start_step))
    early_stopping_patience = int(getattr(training_args, "early_stopping_patience", 0) or 0)
    if early_stopping_patience > 0:
        if data_module.get("eval_dataset") is None:
            raise ValueError("early_stopping_patience > 0 requires validation_size > 0.")
        if not training_args.load_best_model_at_end:
            raise ValueError("early_stopping_patience > 0 requires load_best_model_at_end=True.")
        callbacks.append(
            DelayedEarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=float(getattr(training_args, "early_stopping_threshold", 0.0) or 0.0),
                start_step=early_stopping_start_step,
            )
        )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )
    trainer.train()
    if getattr(model, "config", None) is not None:
        model.config.table_read_mode = table_read_mode
    base_model = getattr(model, "base_model", None)
    if base_model is not None and getattr(base_model, "config", None) is not None:
        base_model.config.table_read_mode = table_read_mode
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    # 作为脚本直接运行时启动训练。
    train()
