# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model.

中文阅读导览：
1. 这份文件整体沿用 HuggingFace LLaMA 的模型结构：RMSNorm、RoPE、MLP、Attention、
   DecoderLayer、LlamaModel、LlamaForCausalLM。
2. 本项目主要改动集中在自定义注意力 `LlamaFlashAttention2Ours`：
   - `position_ids` 仍表示常规序列/二维位置；
   - `token_ids` 表示表格读取顺序，用于给 RoPE 注入 row/column/adaptive 等顺序；
   - `table_read_mode == "2d"` 时会同时跑 x/y 两个分支；
   - `table_read_mode == "adaptive"` 时会同时构造 5 个候选排序分支：
     row、column、hilbert、snake、spiral。
3. adaptive 分支中的 router 是逐层、逐 token、逐 head 计算的：
   `hidden_states -> gate_1/gate_2/adaptive_gate_3 -> 5 路 logits -> softmax`。
   当前实现已经改成 hard top-1 routing：测试时只使用权重最高的排序专家；训练时
   使用 straight-through 写法，让前向是硬选择，同时保留 router 的梯度。
4. `substart/subend` 用来定位表格片段，训练时可在表格区域上计算辅助熵正则。
"""
import math
import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_llama import LlamaConfig

try:
    from transformers.utils import is_flash_attn_2_available
except ImportError:
    from transformers.utils import is_flash_attn_available as is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
else:
    flash_attn_func = None
    flash_attn_varlen_func = None

    # Keep symbol defined to avoid NameError in environments where the old helper
    # incorrectly reports flash-attn availability.
    def index_first_axis(x, indices):
        return x[indices]

    def pad_input(*args, **kwargs):
        raise RuntimeError("flash-attn is unavailable but flash attention path was enabled.")

    def unpad_input(*args, **kwargs):
        raise RuntimeError("flash-attn is unavailable but flash attention path was enabled.")


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(padding_mask):
    # FlashAttention 的 varlen 接口需要把 padding 后的 batch 展平成非 padding token 序列。
    # 这里返回三类信息：
    # - indices：非 padding token 在展平序列中的索引；
    # - cu_seqlens：每个样本有效长度的前缀和；
    # - max_seqlen_in_batch：当前 batch 的最大有效长度。
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    # input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
    input_ids, position_ids, substart, subend, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    构造自回归 causal mask，并额外定位表格区域。

    普通 LLaMA 只需要 causal mask；本项目还传入 `substart/subend`，用于在训练时
    找到 prompt 中表格片段的起止位置，从而只在表格相关 token 上统计辅助损失。
    """
    bsz, tgt_len = input_ids.shape
    # 标准 causal mask：当前位置只能看见自己以及之前的位置。
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    pos_A = None
    pos_B = None

    if substart is not None:
        # `substart/subend` 本身是一段 token 模式。unfold 会在 input_ids 上滑动窗口，
        # 然后用整段匹配找到这段模式第一次出现的位置。
        sub_len = subend.size(1)

        windows = input_ids.unfold(dimension=1, size=sub_len, step=1)

        substart = substart[:, None, :]  # [bsz, 1, sub_len]
        subend = subend[:, None, :]  # [bsz, 1, sub_len]

        matches_start = (windows == substart).all(dim=2)  # [bsz, seq_len - sub_len + 1]
        matches_end = (windows == subend).all(dim=2)  # [bsz, seq_len - sub_len + 1]

        pos_A = matches_start.long().argmax(dim=1)  # [bsz]
        pos_B = matches_end.long().argmax(dim=1)  # [bsz]

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length), pos_A, pos_B


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

        # `expert_nums=2` 用于 2D 模式：x 顺序分支和 y 顺序分支。
        # `adaptive_expert_nums` 用于 adaptive 模式：默认多种表格读取顺序视角。
        self.expert_nums = 2
        self.adaptive_expert_nums = int(getattr(config, "adaptive_expert_nums", 3))
        # router 以单个 attention head 的 hidden slice 为输入：
        # [bsz, q_len, num_heads, head_dim] -> [bsz, q_len, num_heads, expert_nums]。
        # gate_1/gate_2 形成类似 GLU 的门控特征，gate_3/adaptive_gate_3 输出专家 logits。
        self.gate_1 = nn.Linear(self.head_dim, 4 * self.head_dim, bias=True)
        self.gate_2 = nn.Linear(self.head_dim, 4 * self.head_dim, bias=True)
        self.gate_3 = nn.Linear(4 * self.head_dim, self.expert_nums, bias=True)
        self.adaptive_gate_3 = nn.Linear(4 * self.head_dim, self.adaptive_expert_nums, bias=True)
        self.order_router_gate_3 = nn.ModuleList(
            [nn.Linear(4 * self.head_dim, 1, bias=True) for _ in range(4)]
        )
        self.row_gate_proj = nn.Linear(self.head_dim, 1, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]
        self.row_gate = None
        self.routed_order_scores = None
        self.topk_indices = None
        self.final_sparse_gates = None
        self._adaptive_router_stat_count = 0
        self._adaptive_router_prob_sum = None
        self._adaptive_router_top1_sum = None
        self._adaptive_router_entropy_sum = None
        self._adaptive_router_table_count = 0
        self._adaptive_router_table_prob_sum = None
        self._adaptive_router_table_top1_sum = None
        self._adaptive_router_table_entropy_sum = None
        self._order_moe_stat_count = 0
        self._order_moe_row_gate_sum = None
        self._order_moe_routed_score_sum = None
        self._order_moe_topk_sum = None
        self._order_moe_sparse_gate_sum = None
        self._order_moe_entropy_sum = None
        self._order_moe_max_score_sum = None
        self._order_moe_table_count = 0
        self._order_moe_table_row_gate_sum = None
        self._order_moe_table_routed_score_sum = None
        self._order_moe_table_topk_sum = None
        self._order_moe_table_sparse_gate_sum = None
        self._order_moe_table_entropy_sum = None
        self._order_moe_table_max_score_sum = None

    def _record_adaptive_router_stats(self, routing_probs, table_token_mask=None):
        if not (self.training and getattr(self.config, "adaptive_router_logging", False)):
            return
        with torch.no_grad():
            probs = routing_probs.detach().float()
            prob_mean = probs.mean(dim=(0, 1, 2))
            top1 = F.one_hot(
                torch.argmax(probs, dim=-1),
                num_classes=self.adaptive_expert_nums,
            ).float().mean(dim=(0, 1, 2))
            entropy = -(probs.clamp_min(1e-10) * probs.clamp_min(1e-10).log()).sum(dim=-1).mean()

            if self._adaptive_router_prob_sum is None:
                self._adaptive_router_prob_sum = torch.zeros_like(prob_mean)
                self._adaptive_router_top1_sum = torch.zeros_like(top1)
                self._adaptive_router_entropy_sum = torch.zeros_like(entropy)

            self._adaptive_router_prob_sum += prob_mean
            self._adaptive_router_top1_sum += top1
            self._adaptive_router_entropy_sum += entropy
            self._adaptive_router_stat_count += 1

            if table_token_mask is not None:
                table_token_mask = table_token_mask.detach().bool()
                if table_token_mask.any():
                    table_probs = probs[table_token_mask]
                    table_prob_mean = table_probs.mean(dim=(0, 1))
                    table_top1 = F.one_hot(
                        torch.argmax(table_probs, dim=-1),
                        num_classes=self.adaptive_expert_nums,
                    ).float().mean(dim=(0, 1))
                    table_entropy = -(
                        table_probs.clamp_min(1e-10) * table_probs.clamp_min(1e-10).log()
                    ).sum(dim=-1).mean()

                    if self._adaptive_router_table_prob_sum is None:
                        self._adaptive_router_table_prob_sum = torch.zeros_like(table_prob_mean)
                        self._adaptive_router_table_top1_sum = torch.zeros_like(table_top1)
                        self._adaptive_router_table_entropy_sum = torch.zeros_like(table_entropy)

                    self._adaptive_router_table_prob_sum += table_prob_mean
                    self._adaptive_router_table_top1_sum += table_top1
                    self._adaptive_router_table_entropy_sum += table_entropy
                    self._adaptive_router_table_count += 1

    def pop_adaptive_router_stats(self):
        count = self._adaptive_router_stat_count
        if count == 0 or self._adaptive_router_prob_sum is None:
            return None
        stats = {
            "count": count,
            "prob_sum": self._adaptive_router_prob_sum.detach().float().cpu(),
            "top1_sum": self._adaptive_router_top1_sum.detach().float().cpu(),
            "entropy_sum": self._adaptive_router_entropy_sum.detach().float().cpu(),
        }
        table_count = self._adaptive_router_table_count
        if table_count > 0 and self._adaptive_router_table_prob_sum is not None:
            stats.update(
                {
                    "table_count": table_count,
                    "table_prob_sum": self._adaptive_router_table_prob_sum.detach().float().cpu(),
                    "table_top1_sum": self._adaptive_router_table_top1_sum.detach().float().cpu(),
                    "table_entropy_sum": self._adaptive_router_table_entropy_sum.detach().float().cpu(),
                }
            )
        self._adaptive_router_stat_count = 0
        self._adaptive_router_prob_sum = None
        self._adaptive_router_top1_sum = None
        self._adaptive_router_entropy_sum = None
        self._adaptive_router_table_count = 0
        self._adaptive_router_table_prob_sum = None
        self._adaptive_router_table_top1_sum = None
        self._adaptive_router_table_entropy_sum = None
        return stats

    def _record_order_moe_stats(self, row_gate, routed_scores, topk_indices, sparse_gates, table_token_mask=None):
        if not (self.training and getattr(self.config, "adaptive_router_logging", False)):
            return
        with torch.no_grad():
            row_gate = row_gate.detach().float()
            routed_scores = routed_scores.detach().float()
            sparse_gates = sparse_gates.detach().float()
            topk_mask = torch.zeros_like(routed_scores)
            topk_mask.scatter_(-1, topk_indices.detach(), 1.0)

            row_gate_mean = row_gate.mean()
            routed_score_mean = routed_scores.mean(dim=(0, 1, 2))
            topk_mean = topk_mask.mean(dim=(0, 1, 2))
            sparse_gate_mean = sparse_gates.mean(dim=(0, 1, 2))
            entropy = -(routed_scores.clamp_min(1e-10) * routed_scores.clamp_min(1e-10).log()).sum(dim=-1).mean()
            max_score = routed_scores.max(dim=-1).values.mean()

            if self._order_moe_row_gate_sum is None:
                self._order_moe_row_gate_sum = torch.zeros_like(row_gate_mean)
                self._order_moe_routed_score_sum = torch.zeros_like(routed_score_mean)
                self._order_moe_topk_sum = torch.zeros_like(topk_mean)
                self._order_moe_sparse_gate_sum = torch.zeros_like(sparse_gate_mean)
                self._order_moe_entropy_sum = torch.zeros_like(entropy)
                self._order_moe_max_score_sum = torch.zeros_like(max_score)

            self._order_moe_row_gate_sum += row_gate_mean
            self._order_moe_routed_score_sum += routed_score_mean
            self._order_moe_topk_sum += topk_mean
            self._order_moe_sparse_gate_sum += sparse_gate_mean
            self._order_moe_entropy_sum += entropy
            self._order_moe_max_score_sum += max_score
            self._order_moe_stat_count += 1

            if table_token_mask is not None:
                table_token_mask = table_token_mask.detach().bool()
                if table_token_mask.any():
                    table_row_gate = row_gate[table_token_mask]
                    table_routed_scores = routed_scores[table_token_mask]
                    table_topk_mask = topk_mask[table_token_mask]
                    table_sparse_gates = sparse_gates[table_token_mask]

                    table_row_gate_mean = table_row_gate.mean()
                    table_routed_score_mean = table_routed_scores.mean(dim=(0, 1))
                    table_topk_mean = table_topk_mask.mean(dim=(0, 1))
                    table_sparse_gate_mean = table_sparse_gates.mean(dim=(0, 1))
                    table_entropy = -(
                        table_routed_scores.clamp_min(1e-10) * table_routed_scores.clamp_min(1e-10).log()
                    ).sum(dim=-1).mean()
                    table_max_score = table_routed_scores.max(dim=-1).values.mean()

                    if self._order_moe_table_row_gate_sum is None:
                        self._order_moe_table_row_gate_sum = torch.zeros_like(table_row_gate_mean)
                        self._order_moe_table_routed_score_sum = torch.zeros_like(table_routed_score_mean)
                        self._order_moe_table_topk_sum = torch.zeros_like(table_topk_mean)
                        self._order_moe_table_sparse_gate_sum = torch.zeros_like(table_sparse_gate_mean)
                        self._order_moe_table_entropy_sum = torch.zeros_like(table_entropy)
                        self._order_moe_table_max_score_sum = torch.zeros_like(table_max_score)

                    self._order_moe_table_row_gate_sum += table_row_gate_mean
                    self._order_moe_table_routed_score_sum += table_routed_score_mean
                    self._order_moe_table_topk_sum += table_topk_mean
                    self._order_moe_table_sparse_gate_sum += table_sparse_gate_mean
                    self._order_moe_table_entropy_sum += table_entropy
                    self._order_moe_table_max_score_sum += table_max_score
                    self._order_moe_table_count += 1

    def pop_order_moe_stats(self):
        count = self._order_moe_stat_count
        if count == 0 or self._order_moe_row_gate_sum is None:
            return None
        stats = {
            "count": count,
            "row_gate_sum": self._order_moe_row_gate_sum.detach().float().cpu(),
            "routed_score_sum": self._order_moe_routed_score_sum.detach().float().cpu(),
            "topk_sum": self._order_moe_topk_sum.detach().float().cpu(),
            "sparse_gate_sum": self._order_moe_sparse_gate_sum.detach().float().cpu(),
            "entropy_sum": self._order_moe_entropy_sum.detach().float().cpu(),
            "max_score_sum": self._order_moe_max_score_sum.detach().float().cpu(),
        }
        table_count = self._order_moe_table_count
        if table_count > 0 and self._order_moe_table_row_gate_sum is not None:
            stats.update(
                {
                    "table_count": table_count,
                    "table_row_gate_sum": self._order_moe_table_row_gate_sum.detach().float().cpu(),
                    "table_routed_score_sum": self._order_moe_table_routed_score_sum.detach().float().cpu(),
                    "table_topk_sum": self._order_moe_table_topk_sum.detach().float().cpu(),
                    "table_sparse_gate_sum": self._order_moe_table_sparse_gate_sum.detach().float().cpu(),
                    "table_entropy_sum": self._order_moe_table_entropy_sum.detach().float().cpu(),
                    "table_max_score_sum": self._order_moe_table_max_score_sum.detach().float().cpu(),
                }
            )

        self._order_moe_stat_count = 0
        self._order_moe_row_gate_sum = None
        self._order_moe_routed_score_sum = None
        self._order_moe_topk_sum = None
        self._order_moe_sparse_gate_sum = None
        self._order_moe_entropy_sum = None
        self._order_moe_max_score_sum = None
        self._order_moe_table_count = 0
        self._order_moe_table_row_gate_sum = None
        self._order_moe_table_routed_score_sum = None
        self._order_moe_table_topk_sum = None
        self._order_moe_table_sparse_gate_sum = None
        self._order_moe_table_entropy_sum = None
        self._order_moe_table_max_score_sum = None
        return stats

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2Ours(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.

    本项目的核心定制也在这个类里：
    - 普通 1D 模式：按单一路径跑 FlashAttention；
    - 2D 模式：分别按 x/y 两套 token order 做 RoPE 和 attention，再由 2 路 router 融合；
    - adaptive 模式：按多套候选 order 分别跑 attention，再由 router 选择/融合。
    """

    def _restore_original_order(self, sorted_tensor, sorted_indices):
        # `_run_order_branch` 为了让 FlashAttention 沿某种表格顺序计算，会先按 order_id 排序。
        # attention 输出后，需要用这里的 inverse gather 恢复到原始 input_ids 的 token 顺序。
        inverse_indices = torch.empty_like(sorted_indices)
        restore_indices = torch.arange(sorted_indices.size(1), device=sorted_indices.device)
        inverse_indices.scatter_(1, sorted_indices, restore_indices.unsqueeze(0).expand_as(sorted_indices))
        inverse_indices = inverse_indices.unsqueeze(2).unsqueeze(3).expand(
            -1, -1, self.num_heads, self.head_dim
        )
        return torch.gather(sorted_tensor, 1, inverse_indices)

    def _run_order_branch(
        self,
        query_states,
        key_states,
        value_states,
        cos,
        sin,
        order_ids,
        padding_mask,
        q_len,
        dropout_rate,
        past_key_state=None,
    ):
        # 对某一种读取顺序单独跑一个 attention 分支。
        # order_ids 是这一分支的 token 顺序编号，例如 row-major 或 hilbert 顺序。
        # RoPE 使用 order_ids 而不是普通 position_ids，因此同一批 input_ids 可以拥有不同的表格顺序视角。
        q_order, k_order = apply_rotary_pos_emb(query_states, key_states, cos, sin, order_ids)
        if past_key_state is not None:
            # 自回归生成时 q_len 通常为 1。past_key_state 缓存的是同一排序分支历史 token 的 key。
            k_order = torch.cat([past_key_state, k_order], dim=2)

        cached_k_order = k_order
        k_order = repeat_kv(k_order, self.num_key_value_groups)
        value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)

        q_order = q_order.transpose(1, 2)
        k_order = k_order.transpose(1, 2)
        v_order = value_states_repeated.transpose(1, 2)

        if q_len > 1:
            # prefilling 阶段一次输入完整 prompt，需要把 Q/K/V 都按 order_ids 升序重排，
            # 让 FlashAttention 看到“该排序下”的连续序列。生成阶段 q_len=1 时无需重排。
            _, sorted_indices = torch.sort(order_ids, descending=False, dim=-1)
            gather_indices = sorted_indices.unsqueeze(2).unsqueeze(3).expand(
                -1, -1, self.num_heads, self.head_dim
            )
            q_order = torch.gather(q_order, 1, gather_indices)
            k_order = torch.gather(k_order, 1, gather_indices)
            v_order = torch.gather(v_order, 1, gather_indices)

        attn_output = self._flash_attention_forward(
            q_order,
            k_order,
            v_order,
            padding_mask,
            q_len,
            dropout=dropout_rate,
        )

        if q_len > 1:
            # attention 已经在排序后的序列上完成，输出再还原回原 input_ids 顺序，
            # 这样后续 residual/MLP/lm_head 都仍然对齐原始 token 位置。
            attn_output = self._restore_original_order(attn_output, sorted_indices)

        return attn_output, cached_k_order

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_ids: Optional[torch.LongTensor] = None,
        pos_A: Optional[torch.Tensor] = None,
        pos_B: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dime x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=max(position_ids.max()+1,token_ids.max()+1))

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        table_read_mode = str(getattr(self.config, "table_read_mode", "2d")).lower()
        use_2d_attention = table_read_mode == "2d"
        use_adaptive_attention = table_read_mode == "adaptive"
        padding_mask = padding_mask.long()
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        if use_adaptive_attention:
            # adaptive 的 token_ids 不是 [x_ids, y_ids] 两路，而是多路顺序直接拼接。
            # 每一路长度都是 q_len，因此总宽度必须为 adaptive_expert_nums * q_len。
            assert token_ids.size()[1] == self.adaptive_expert_nums * q_len
            adaptive_order_ids = [
                token_ids[..., idx * q_len:(idx + 1) * q_len]
                for idx in range(self.adaptive_expert_nums)
            ]

            if past_key_value is not None:
                # adaptive cache 的最后一个元素保存 value_states；前 5 个元素分别保存 5 个排序分支的 key。
                value_states = torch.cat([past_key_value[-1], value_states], dim=2)

            attn_outputs = []
            cached_keys = []
            for idx, order_ids in enumerate(adaptive_order_ids):
                # 依次跑多个排序视角。idx 与数据预处理中的 ADAPTIVE_TABLE_READ_MODES 对齐。
                past_key_state = past_key_value[idx] if past_key_value is not None else None
                branch_output, cached_key = self._run_order_branch(
                    query_states,
                    key_states,
                    value_states,
                    cos,
                    sin,
                    order_ids,
                    padding_mask,
                    q_len,
                    dropout_rate,
                    past_key_state=past_key_state,
                )
                attn_outputs.append(branch_output)
                cached_keys.append(cached_key)

            # use_cache=True 时返回 adaptive_expert_nums 路排序 key + 共享的 value_states。
            past_key_value = tuple(cached_keys + [value_states]) if use_cache else None

            # router 是逐层、逐 token、逐 attention head 计算的。
            # hidden_states:            [bsz, q_len, hidden_size]
            # hidden_states_for_router: [bsz, q_len, num_heads, head_dim]
            # adaptive_gate_3 输出 5 路 logits；order MoE 下每个 routed order 独立输出 1 路 logit。
            hidden_states_for_router = hidden_states.view(bsz, q_len, self.num_heads, self.head_dim)
            table_token_mask = None
            if pos_A is not None and pos_B is not None and q_len > 1:
                token_positions = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
                table_start = (pos_A.to(hidden_states.device) + 4).unsqueeze(1)
                table_end = pos_B.to(hidden_states.device).unsqueeze(1)
                table_token_mask = (token_positions >= table_start) & (token_positions < table_end)
                if padding_mask is not None and padding_mask.size(-1) >= q_len:
                    table_token_mask = table_token_mask & padding_mask[:, -q_len:].bool()
            if bool(getattr(self.config, "use_order_moe", False)):
                expert_names = list(
                    getattr(
                        self.config,
                        "adaptive_expert_names",
                        ["row", "column", "snake", "hilbert", "spiral"],
                    )
                )
                shared_order = str(getattr(self.config, "shared_order", "row")).lower()
                routed_orders = getattr(
                    self.config,
                    "routed_orders",
                    ["column", "snake", "hilbert", "spiral"],
                )
                if isinstance(routed_orders, str):
                    routed_orders = [name.strip().lower() for name in routed_orders.split(",") if name.strip()]
                else:
                    routed_orders = [str(name).lower() for name in routed_orders]

                shared_idx = expert_names.index(shared_order)
                routed_indices = [expert_names.index(name) for name in routed_orders]
                routed_count = len(routed_indices)
                if routed_count > len(self.order_router_gate_3):
                    raise ValueError(
                        f"routed_orders has {routed_count} entries, but order_router_gate_3 only has "
                        f"{len(self.order_router_gate_3)} independent router heads."
                    )
                top_k = min(max(int(getattr(self.config, "order_top_k", 2)), 1), routed_count)

                router_features = self.act_fn(self.gate_1(hidden_states_for_router)) * self.gate_2(
                    hidden_states_for_router
                )
                routed_logits = torch.cat(
                    [self.order_router_gate_3[idx](router_features) for idx in range(routed_count)],
                    dim=-1,
                )
                router_temperature = float(getattr(self.config, "order_router_temperature", 0.5))
                if router_temperature <= 0:
                    raise ValueError("order_router_temperature must be greater than 0.")
                routed_order_scores = nn.functional.softmax(
                    routed_logits / router_temperature,
                    dim=-1,
                    dtype=torch.float,
                )
                topk_values, topk_indices = torch.topk(routed_order_scores, k=top_k, dim=-1)
                topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                final_sparse_gates = torch.zeros_like(routed_order_scores)
                final_sparse_gates.scatter_(-1, topk_indices, topk_values)

                row_gate = torch.sigmoid(self.row_gate_proj(hidden_states_for_router)).to(hidden_states.dtype)
                sparse_gates = final_sparse_gates.to(hidden_states.dtype)

                self.row_gate = row_gate.detach()
                self.routed_order_scores = routed_order_scores.detach()
                self.topk_indices = topk_indices.detach()
                self.final_sparse_gates = final_sparse_gates.detach()

                self._record_order_moe_stats(
                    row_gate,
                    routed_order_scores,
                    topk_indices,
                    final_sparse_gates,
                    table_token_mask=table_token_mask,
                )

                attn_output = attn_outputs[shared_idx]
                aux_scale = float(getattr(self.config, "order_aux_scale", 0.5))
                routed_attn_outputs = torch.stack([attn_outputs[idx] for idx in routed_indices], dim=0)
                routed_deltas = routed_attn_outputs - attn_outputs[shared_idx].unsqueeze(0)
                routed_attn_delta = sparse_gates.permute(3, 0, 1, 2).unsqueeze(-1) * routed_deltas
                attn_output = attn_output + aux_scale * torch.sum(routed_attn_delta, dim=0)
            else:
                router_logits = self.adaptive_gate_3(
                    self.act_fn(self.gate_1(hidden_states_for_router)) * self.gate_2(hidden_states_for_router)
                )

                routing_probs = nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
                self._record_adaptive_router_stats(routing_probs, table_token_mask=table_token_mask)

                routing_weights = routing_probs.to(hidden_states.dtype)
                attn_output = attn_outputs[0]
                if self.adaptive_expert_nums > 1:
                    supplement_scale = float(getattr(self.config, "adaptive_residual_scale", 1.0))
                    supplement_weights = routing_weights[..., 1:].permute(3, 0, 1, 2).contiguous()
                    supplement_outputs = torch.stack(
                        [branch - attn_outputs[0] for branch in attn_outputs[1:]],
                        dim=0,
                    )
                    adaptive_delta = supplement_weights.unsqueeze(-1) * supplement_outputs.view(
                        self.adaptive_expert_nums - 1,
                        bsz,
                        q_len,
                        self.num_heads,
                        self.head_dim,
                    )
                    attn_output = attn_output + supplement_scale * torch.sum(adaptive_delta, dim=0)

        else:
            # 非 adaptive 模式有两种输入宽度：
            # - 2 * q_len：普通 2D/row/column/snake/hilbert/spiral 模式；
            # - adaptive_expert_nums * q_len：adaptive warmup 时模型 config 可能临时设成 row，但数据仍保留多路 token_ids。
            token_id_width = token_ids.size()[1]
            if token_id_width == 2 * q_len:
                token_id_channels = 2
            elif token_id_width == self.adaptive_expert_nums * q_len:
                token_id_channels = self.adaptive_expert_nums
            else:
                raise ValueError(
                    f"Unexpected token_ids width={token_id_width} for q_len={q_len}. "
                    f"Expected {2 * q_len} for normal modes or "
                    f"{self.adaptive_expert_nums * q_len} for adaptive warmup."
                )
            q_x, k_x = apply_rotary_pos_emb(query_states, key_states, cos, sin, token_ids[..., :q_len])

            if use_2d_attention or use_adaptive_attention:
                # 2D 模式中 token_ids 前半段视作 x/row 顺序，后半段视作 y/column 顺序。
                q_y, k_y = apply_rotary_pos_emb(query_states, key_states, cos, sin, token_ids[..., q_len:])
            else:
                # 1D 表格读取模式只需要一个顺序；这里优先使用 position_ids，保持线性序列行为。
                rotary_ids_1d = position_ids[..., :q_len] if position_ids is not None else token_ids[..., :q_len]
                q_1d, k_1d = apply_rotary_pos_emb(query_states, key_states, cos, sin, rotary_ids_1d)

            if past_key_value is not None:
                # cache 中保存的是已经做过 RoPE 的 key，以及共享 value_states。
                if use_2d_attention:
                    k_x = torch.cat([past_key_value[0], k_x], dim=2)
                    k_y = torch.cat([past_key_value[1], k_y], dim=2)
                else:
                    k_1d = torch.cat([past_key_value[0], k_1d], dim=2)
                value_states = torch.cat([past_key_value[-1], value_states], dim=2)

            if use_2d_attention:
                # 2D 模式缓存三项：x 分支 key、y 分支 key、共享 value。
                past_key_value = (k_x, k_y, value_states) if use_cache else None
                k_x = repeat_kv(k_x, self.num_key_value_groups)
                k_y = repeat_kv(k_y, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                q_x, q_y = q_x.transpose(1, 2), q_y.transpose(1, 2)
                k_x, k_y = k_x.transpose(1, 2), k_y.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                v_x, v_y = value_states.clone(), value_states.clone()

                if q_len > 1:
                    # y 分支需要按 column order 排序后再跑 FlashAttention。
                    # 由于 FlashAttention 本身假设序列顺序已经排好，这里用 gather 显式重排。
                    _, token_col = torch.sort(token_ids[..., q_len:], descending=False, dim=-1)
                    token_col = token_col.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.num_heads, self.head_dim)
                    q_y = torch.gather(q_y, 1, token_col)
                    k_y = torch.gather(k_y, 1, token_col)
                    v_y = torch.gather(value_states, 1, token_col)
                    inverse_indices = token_ids[..., q_len:].unsqueeze(2).unsqueeze(3).expand(
                        -1, -1, self.num_heads, self.head_dim
                    )

                attn_output_x = self._flash_attention_forward(q_x, k_x, v_x, padding_mask, q_len, dropout=dropout_rate)
                attn_output_y = self._flash_attention_forward(q_y, k_y, v_y, padding_mask, q_len, dropout=dropout_rate)

                if q_len > 1:
                    # y 分支输出恢复到原始 token 顺序，以便和 x 分支逐位置融合。
                    attn_output_y = torch.gather(attn_output_y, 1, inverse_indices)

                # 2D 模式的 router 输出 2 路权重：x 分支和 y 分支。
                # 这里仍是 soft mixture，未改成 hard top-1。
                hidden_states_for_router = hidden_states.view(bsz, q_len, self.num_heads, self.head_dim)
                router_logits = self.gate_3(
                    self.act_fn(self.gate_1(hidden_states_for_router)) * self.gate_2(hidden_states_for_router)
                )
                routing_weights = nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
                routing_weights = routing_weights.to(hidden_states.dtype)
                routing_weights = routing_weights.permute(3, 0, 1, 2).contiguous()

                attn_output = torch.stack([attn_output_x, attn_output_y], dim=0)
                attn_output = routing_weights.unsqueeze(-1) * attn_output.view(
                    self.expert_nums, bsz, q_len, self.num_heads, self.head_dim
                )
                attn_output = torch.sum(attn_output, dim=0)
            else:
                # 1D 模式只有一个顺序分支，例如 row/column/hilbert/snake/spiral 的硬编码读取顺序。
                past_key_value = (k_1d, k_1d, value_states) if use_cache else None
                k_1d = repeat_kv(k_1d, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                q_1d = q_1d.transpose(1, 2)
                k_1d = k_1d.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                attn_output = self._flash_attention_forward(
                    q_1d, k_1d, value_states, padding_mask, q_len, dropout=dropout_rate
                )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        #loss
        epsilon = 1e-10
        entropy_avg = 0.0
        entropy_list = []
        order_moe_entropy = None
        if self.config.output_loss:
            if use_2d_attention:
                # 这里的辅助项只在 2D soft router 上生效：
                # 取表格区域内的 x/y 路由权重，计算熵并加入最终 loss。
                # adaptive hard routing 分支当前不计算该熵项，返回 0。
                for i in range(bsz):
                    routing_back = routing_weights.clone()[:, i, pos_A[i]+4:pos_B[i], ...]
                    routing_back = torch.clamp(routing_back, min=epsilon, max=1.0)
                    entropy = -torch.sum(routing_back * torch.log(routing_back), dim=0)
                    entropy = entropy.mean(dim=(0,1))
                    entropy_list.append(entropy)

                entropy_avg = torch.mean(torch.stack(entropy_list))
            elif use_adaptive_attention and bool(getattr(self.config, "use_order_moe", False)):
                coef = float(getattr(self.config, "order_router_entropy_coef", 0.01))
                if coef > 0 and "routed_order_scores" in locals():
                    routed_entropy = -(
                        routed_order_scores.clamp_min(epsilon) * routed_order_scores.clamp_min(epsilon).log()
                    ).sum(dim=-1)
                    if table_token_mask is not None and table_token_mask.any():
                        order_moe_entropy = routed_entropy[table_token_mask].mean()
                    else:
                        order_moe_entropy = routed_entropy.mean()
                    entropy_avg = coef * order_moe_entropy
                else:
                    entropy_avg = hidden_states.new_zeros(())
            else:
                entropy_avg = hidden_states.new_zeros(())

        return attn_output, attn_weights, past_key_value, entropy_avg

    def _flash_attention_forward(
        self, query_states, key_states, value_states, padding_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            padding_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Fallback when flash-attn kernels are unavailable in the runtime env.
        if flash_attn_func is None or flash_attn_varlen_func is None:
            logger.warning_once("flash-attn kernels are unavailable; falling back to PyTorch attention.")
            bsz, q_len, _, head_dim = query_states.size()
            k_len = key_states.size(1)
            scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))

            q = query_states.transpose(1, 2)  # [bsz, heads, q_len, head_dim]
            k = key_states.transpose(1, 2)    # [bsz, heads, k_len, head_dim]
            v = value_states.transpose(1, 2)  # [bsz, heads, k_len, head_dim]

            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            causal_mask = torch.triu(
                torch.ones(q_len, k_len, dtype=torch.bool, device=scores.device), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)

            if padding_mask is not None:
                key_padding = (padding_mask[:, -k_len:] == 0).unsqueeze(1).unsqueeze(2)  # [bsz,1,1,k_len]
                scores = scores.masked_fill(key_padding, torch.finfo(scores.dtype).min)

            attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            if dropout and self.training:
                attn_weights = torch.dropout(attn_weights, p=dropout, train=True)
            attn_output = torch.matmul(attn_weights, v)  # [bsz, heads, q_len, head_dim]
            return attn_output.transpose(1, 2).contiguous()  # [bsz, q_len, heads, head_dim]

        # Contains at least one padding token in the sequence
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, padding_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=True
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        num_heads = query_layer.size()[-2]
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            padding_mask = padding_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, padding_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dime x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, padding_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, padding_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            padding_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, padding_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=True
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            padding_mask = padding_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, padding_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Keep using the custom attention module for interface compatibility
        # (it consumes input_ids/token_ids/pos_A/pos_B/output_loss).
        # The module internally falls back when flash-attn kernels are unavailable.
        # 这里固定使用项目自定义 attention，以便所有 decoder layer 都能接收 token_ids、
        # substart/subend 等额外字段。
        self.self_attn = LlamaFlashAttention2Ours(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor = None,
        token_ids: Optional[torch.LongTensor] = None,
        pos_A: Optional[torch.Tensor] = None,
        pos_B: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_loss: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # Pre-norm 结构：先归一化再进入 attention，输出与 residual 相加。
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, entropy = self.self_attn(
            hidden_states=hidden_states,
            input_ids=input_ids,
            token_ids=token_ids,
            pos_A=pos_A,
            pos_B=pos_B,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        # 第二个 residual block：RMSNorm -> MLP -> residual add。
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_loss:
            outputs += (entropy,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, input_ids, position_ids, substart, subend, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # 除了标准 causal mask，本函数还返回 pos_A/pos_B：
        # 它们是 substart/subend 在 input_ids 中匹配到的位置，用于辅助 loss 定位表格区域。
        combined_attention_mask = None
        pos_A = None
        pos_B = None
        if input_shape[-1] > 1:
            # combined_attention_mask = _make_causal_mask(
            #     input_shape,
            #     inputs_embeds.dtype,
            #     device=inputs_embeds.device,
            #     past_key_values_length=past_key_values_length,
            # )
            combined_attention_mask, pos_A, pos_B = _make_causal_mask(
                input_ids,
                position_ids,
                substart, 
                subend,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask, pos_A, pos_B

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_ids: Optional[torch.LongTensor] = None,
        substart: Optional[torch.LongTensor] = None,
        subend: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_loss: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_loss = output_loss if output_loss is not None else self.config.output_loss
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            # 本项目的 position_ids 在数据侧通常按 [px, py] 拼接，因此宽度是 2 * seq_length。
            # attention 内部会按需要读取前后两段。
            position_ids = position_ids.view(-1, seq_length*2).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None
        # 当前实现强制把 padding_mask 设为 attention_mask，保证 FlashAttention 分支始终拿到有效 mask。
        padding_mask = attention_mask

        attention_mask, pos_A, pos_B = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), input_ids, position_ids, substart, subend, inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        entropy_list = []

        for idx, decoder_layer in enumerate(self.layers):
            # 逐层传递 token_ids/position_ids/cache。注意 token_ids 在整层中保持同一套顺序编码，
            # 每层的 router 会基于该层 hidden_states 重新计算专家权重。
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, output_loss=output_loss, padding_mask=padding_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    input_ids,
                    token_ids,
                    pos_A,
                    pos_B,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    input_ids=input_ids,
                    token_ids=token_ids,
                    pos_A=pos_A,
                    pos_B=pos_B,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    output_loss=output_loss,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                )

            hidden_states = layer_outputs[0]

            if output_loss:
                entropy_list.append(layer_outputs[1])
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if output_loss:       
            entropy_all = torch.mean(torch.stack(entropy_list))

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, entropy_all, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self._init_adaptive_router_bias()

    def _init_adaptive_router_bias(self):
        row_bias = float(getattr(self.config, "adaptive_router_row_bias", 2.0))
        prior_bias = getattr(self.config, "adaptive_router_prior", None)
        init_std = float(getattr(self.config, "adaptive_router_init_std", 0.0))
        order_init_std = float(getattr(self.config, "order_router_init_std", 0.0))
        order_bias_init_std = float(getattr(self.config, "order_router_bias_init_std", 0.0))
        order_bias = getattr(self.config, "order_router_bias", None)
        for layer in getattr(self.model, "layers", []):
            attn = getattr(layer, "self_attn", None)
            gate = getattr(attn, "adaptive_gate_3", None)
            order_gates = getattr(attn, "order_router_gate_3", None)
            if gate is None and order_gates is None:
                continue
            with torch.no_grad():
                if gate is not None:
                    if init_std > 0:
                        gate.weight.normal_(mean=0.0, std=init_std)
                    else:
                        gate.weight.zero_()
                    if gate.bias is not None:
                        gate.bias.zero_()
                        if prior_bias is not None:
                            prior_bias_tensor = torch.as_tensor(
                                prior_bias,
                                dtype=gate.bias.dtype,
                                device=gate.bias.device,
                            )
                            if prior_bias_tensor.numel() != gate.bias.numel():
                                raise ValueError(
                                    "adaptive_router_prior size must match adaptive_gate_3 bias size."
                                )
                            gate.bias.copy_(prior_bias_tensor)
                        elif gate.bias.numel() > 0:
                            gate.bias[0] = row_bias

                if order_gates is not None:
                    if order_bias is not None:
                        order_bias_tensor = torch.as_tensor(order_bias, dtype=torch.float)
                        if order_bias_tensor.numel() != len(order_gates):
                            raise ValueError(
                                "order_router_bias size must match order_router_gate_3 size."
                            )
                    else:
                        order_bias_tensor = None
                    for idx, order_gate in enumerate(order_gates):
                        if order_init_std > 0:
                            order_gate.weight.normal_(mean=0.0, std=order_init_std)
                        if order_gate.bias is not None:
                            order_gate.bias.zero_()
                            if order_bias_tensor is not None:
                                order_gate.bias.fill_(float(order_bias_tensor[idx].item()))
                            elif order_bias_init_std > 0:
                                order_gate.bias.normal_(mean=0.0, std=order_bias_init_std)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_ids: Optional[torch.LongTensor] = None,
        substart: Optional[torch.LongTensor] = None,
        subend: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_loss: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_loss = output_loss if output_loss is not None else self.config.output_loss
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_loss:  
            return_dict = False
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # LlamaModel 会把 token_ids/position_ids 继续传给每一层自定义 attention。
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_ids=token_ids,
            substart=substart,
            subend=subend,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if output_loss:  
            entropy = outputs[1]

        
        if self.config.lamda:
            lamda = self.config.lamda
        else:
            lamda = 1

        # lm_head 将最后一层 hidden state 投到词表维度，得到标准自回归 LM logits。
        # 注意它和 adaptive router_logits 不是同一个概念：
        # - router_logits 只在 5 个排序专家之间选择；
        # - lm_head logits 在整个 vocab 上预测下一个 token。
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # 标准 causal LM 训练：第 t 个位置预测第 t+1 个 token。
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # 训练脚本中 config.output_loss=True 时，entropy 是 attention router 的辅助项。
            loss = loss + lamda * entropy

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            # 生成阶段已经缓存了历史 K/V，因此每一步只喂最后一个新 token。
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        token_ids = kwargs.get("token_ids", None)
        substart = kwargs.get("substart", None)
        subend = kwargs.get("subend", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                # inference.py 会手动维护 position_ids/token_ids：
                # adaptive 模式下 token_ids 每一步仍是 5 路拼接，模型侧再做 hard top-1 routing。
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "token_ids": token_ids,
                "substart": substart,
                "subend": subend,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
