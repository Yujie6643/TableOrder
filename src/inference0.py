from transformers import AutoConfig, AutoTokenizer
import torch
import numpy as np
from typing import List
import multiprocessing
import datasets
import queue
import time
import os
import pickle
import json
import logging
import re
from tqdm import tqdm
from sft_minicpm_block_textmeta_demo import (
    ADAPTIVE_TABLE_READ_MODES,
    PROMPT_DICT,
    append_encoded_table,
    build_self_explaining_table_blocks,
    is_rectangular_table,
    normalize_table_read_mode,
    parse_table_text,
)
import sys
import os
import random
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if project_root not in sys.path:
    sys.path.append(project_root)

from TPE_Llama.modeling_llama import LlamaForCausalLM

torch.set_printoptions(profile="full")
torch.multiprocessing.set_start_method('spawn',force=True)

def infer_visible_gpu_num():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices and visible_devices.lower() not in {"none", "-1"}:
        return max(1, len([device for device in visible_devices.split(",") if device.strip()]))
    return max(1, torch.cuda.device_count())


gpu_num = int(os.environ.get("GPU_NUM", infer_visible_gpu_num()))
num_workers = int(os.environ.get("NUM_WORKERS", max(1, gpu_num * 3)))
max_length = 4096
max_new_tokens = 100


DATASET_NAME = os.environ.get("DATASET_NAME", "hitab")
TABLE_READ_MODE = os.environ.get("TABLE_READ_MODE", "").strip()
PATH_MODE_SUFFIX = normalize_table_read_mode(TABLE_READ_MODE) if TABLE_READ_MODE else "2d"
ROOT_DIR = os.environ.get("ROOT_DIR", "/data/tyj/2D-TPE-main")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT_DIR, "output", f"{DATASET_NAME}_{PATH_MODE_SUFFIX}"))
INPUT_PATH = os.environ.get("INPUT_PATH", os.path.join(ROOT_DIR, "eval_data", f"{DATASET_NAME}_test.json"))
DEFAULT_OUTPUT_PATH = (
    os.path.join(ROOT_DIR, "res", f"{DATASET_NAME}_{PATH_MODE_SUFFIX}_res.json")
    if TABLE_READ_MODE
    else os.path.join(ROOT_DIR, "res", f"{DATASET_NAME}_res.json")
)
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", DEFAULT_OUTPUT_PATH)

ENABLE_TABLE_BLOCKS = os.environ.get("ENABLE_TABLE_BLOCKS", "False").lower() in {"1", "true", "yes", "y"}
TABLE_BLOCK_ROWS = int(os.environ.get("TABLE_BLOCK_ROWS", "3"))
TABLE_BLOCK_COLS = int(os.environ.get("TABLE_BLOCK_COLS", "4"))
TABLE_HEADER_ROWS = int(os.environ.get("TABLE_HEADER_ROWS", "0"))
INFERENCE_SAMPLE_SIZE = os.environ.get("INFERENCE_SAMPLE_SIZE", "").strip()
INFERENCE_SAMPLE_SIZE = int(INFERENCE_SAMPLE_SIZE) if INFERENCE_SAMPLE_SIZE else None
INFERENCE_SAMPLE_SEED = int(os.environ.get("INFERENCE_SAMPLE_SEED", os.environ.get("SEED", "42")))



def generate_prompt(instruction, question, input_seg=None):
  if input_seg:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input_seg=input_seg, question=question)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def read_data(input_file, to_tokenize_queue):
    with open(input_file, "r") as f:
        ds = json.load(f)

    original_size = len(ds)
    indexed_ds = list(enumerate(ds))
    if INFERENCE_SAMPLE_SIZE is not None:
        if INFERENCE_SAMPLE_SIZE <= 0:
            raise ValueError("INFERENCE_SAMPLE_SIZE must be greater than 0 when provided.")
        if INFERENCE_SAMPLE_SIZE < original_size:
            rng = random.Random(INFERENCE_SAMPLE_SEED)
            sampled_indices = rng.sample(range(original_size), INFERENCE_SAMPLE_SIZE)
            indexed_ds = [(idx, ds[idx]) for idx in sampled_indices]
            print(
                f"Randomly sampled inference data | "
                f"original_size={original_size} sampled_size={INFERENCE_SAMPLE_SIZE} "
                f"seed={INFERENCE_SAMPLE_SEED}"
            )
        else:
            print(
                f"INFERENCE_SAMPLE_SIZE={INFERENCE_SAMPLE_SIZE} >= dataset_size={original_size}, "
                "using the full dataset."
            )

    print(len(indexed_ds))

    for i, data in tqdm(indexed_ds, total=len(indexed_ds)):
        data = dict(data)
        data['idx'] = i
        to_tokenize_queue.put(data)

    for i in range(num_workers):
        to_tokenize_queue.put(None)


def encode_and_insert_separators(table_array, tokenizer):
    separator_col = [1425] # '▁|'
    separator_row = [48017] # '-'

    separator_row_end = [3] # '<SEP>'
    separator_col_end = [4] # '<CLS>'

    new_table = []
    
    for k, row in enumerate(table_array):
        new_row, new_separator = [], []
        for col in row:
            encoded_col = tokenizer.encode(str(col), add_special_tokens=False)
            new_row.append(encoded_col)
            new_row.append(separator_col)  # Insert '|' between each coded column

            new_separator.append(separator_col_end if k == len(table_array) - 1 else separator_row)
            new_separator.append(separator_col)
        new_row.append(separator_row_end)
        new_separator.append(separator_row_end)
        new_table.append(new_row)
        new_table.append(new_separator)
    return new_table


def transpose_2d_rectangular(list_2d):
    """Transpose a rectangular 2D python list without numpy shape inference."""
    if not list_2d:
        return []
    return [list(col) for col in zip(*list_2d)]


def tokenize_data(to_tokenize_queue, to_output_queue, rank):
    model_name = MODEL_PATH
    config = AutoConfig.from_pretrained(model_name)
    config.remove_unused_columns = False
    config._flash_attn_2_enabled = True
    config.output_loss = False
    config.pad_token_id = 0
    table_read_mode = normalize_table_read_mode(TABLE_READ_MODE or getattr(config, "table_read_mode", "2d"))
    if not hasattr(config, "adaptive_expert_nums"):
        config.adaptive_expert_nums = len(ADAPTIVE_TABLE_READ_MODES)
    if not hasattr(config, "adaptive_expert_names"):
        config.adaptive_expert_names = list(ADAPTIVE_TABLE_READ_MODES)
    config.table_read_mode = table_read_mode
    model = LlamaForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16).to(f"cuda:{rank%gpu_num}")
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    
    while True:
        data = to_tokenize_queue.get()
        if data is None:
            break

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=False)
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        tokenizer.pad_token_id = 0  
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left" 

        source = generate_prompt(instruction = data["instruction"], input_seg = data["input_seg"], question = data["question"])

        parts = re.split(r'(\[TAB\] )|(\n\n### Response)', source)
        parts = [part for part in parts if part is not None]

        part1 = parts[0] + parts[1]
        table_data = parts[2]
        part3 = parts[3] + parts[4]

        table_array = parse_table_text(table_data)
        

        # Determine whether the table is a rectangle
        if not is_rectangular_table(table_array):
            logging.error(f"table_array at index {data['idx']}")
            continue

        if ENABLE_TABLE_BLOCKS:
            table_blocks, block_count = build_self_explaining_table_blocks(
                table_array,
                TABLE_BLOCK_ROWS,
                TABLE_BLOCK_COLS,
                TABLE_HEADER_ROWS,
                return_count=True,
            )
        else:
            table_blocks = [table_array]
            block_count = 1

        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(text=part1, add_special_tokens=False)
        l_part1 = len(input_ids)
        tx = list(range(l_part1))
        ty = list(range(l_part1))
        adaptive_token_ids = None
        if table_read_mode == "adaptive":
            adaptive_token_ids = [list(range(l_part1)) for _ in ADAPTIVE_TABLE_READ_MODES]

        px = list(range(l_part1))
        py = list(range(l_part1))

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
            logging.error(f"encoded table at index {data['idx']}")
            continue

        part3_en = tokenizer.encode(text=part3, add_special_tokens=False)
        input_ids.extend(part3_en)
        tx_count = len(tx)
        ty_count = len(ty)
        if table_read_mode == "adaptive":
            for channel in adaptive_token_ids:
                channel_count = len(channel)
                channel.extend(list(range(channel_count, channel_count + len(part3_en))))
        else:
            assert tx_count == ty_count
            tx.extend(list(range(tx_count, tx_count + len(part3_en))))
            ty.extend(list(range(ty_count, ty_count + len(part3_en))))

        k_part3_end = k_part3_start + len(part3_en)
        px.extend(list(range(k_part3_start, k_part3_end)))
        py.extend(list(range(k_part3_start, k_part3_end)))

        if len(input_ids) > tokenizer.model_max_length-1:
            continue
            input_ids = input_ids[-tokenizer.model_max_length+1:]
            px = px[-tokenizer.model_max_length+1:]
            py = py[-tokenizer.model_max_length+1:]
            tx = tx[-tokenizer.model_max_length+1:]
            ty = ty[-tokenizer.model_max_length+1:]

        pi = np.concatenate([px, py])
        ti = np.concatenate(adaptive_token_ids) if table_read_mode == "adaptive" else np.concatenate([tx, ty])

        input_ids = torch.tensor(input_ids).reshape(1, -1)
        token_ids = torch.tensor(ti).reshape(1, -1)
        position_ids = torch.tensor(pi).reshape(1, -1)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(f"cuda:{rank%gpu_num}"),
                token_ids=token_ids.to(f"cuda:{rank%gpu_num}"),
                position_ids=position_ids.to(f"cuda:{rank%gpu_num}"),
                use_cache=True,
            )
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            past_key_values = outputs.past_key_values
            npx = [px[-1] + 1]
            npy = [py[-1] + 1]

            if table_read_mode == "adaptive":
                next_token_channels = [[max(channel) + 1] for channel in adaptive_token_ids]
            else:
                ntx = [tx[-1] + 1]
                nty = [ty[-1] + 1]

            pi = np.concatenate([npx, npy])
            position_ids = torch.tensor(pi).reshape(1, -1)
            ti = np.concatenate(next_token_channels) if table_read_mode == "adaptive" else np.concatenate([ntx, nty])
            token_ids = torch.tensor(ti).reshape(1, -1)
            generated_ids = [pred_token_idx.item()]

            for _ in range(max_new_tokens - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    token_ids=token_ids.to(f"cuda:{rank%gpu_num}"),
                    position_ids=position_ids.to(f"cuda:{rank%gpu_num}"),
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                npx = [npx[-1] + 1]
                npy = [npy[-1] + 1]
                if table_read_mode == "adaptive":
                    next_token_channels = [[channel[-1] + 1] for channel in next_token_channels]
                else:
                    ntx = [ntx[-1] + 1]
                    nty = [nty[-1] + 1]

                pi = np.concatenate([npx, npy])
                position_ids = torch.tensor(pi).reshape(1, -1)
                ti = np.concatenate(next_token_channels) if table_read_mode == "adaptive" else np.concatenate([ntx, nty])
                token_ids = torch.tensor(ti).reshape(1, -1)
                generated_ids.append(pred_token_idx.item())

                if pred_token_idx == tokenizer.eos_token_id:
                    break
            
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            result = { 'idx': data['idx'],
                       'instruction': data['instruction'],
                       'input_seg': data['input_seg'],
                       'question': data['question'],
                       'output': data['output'],
                       'table_block_mode': ENABLE_TABLE_BLOCKS,
                       'table_read_mode': table_read_mode,
                       'table_block_count': block_count,
                       'predict': generated_text}
       
        to_output_queue.put(result)

    to_output_queue.put(None)

def output_data(to_output_queue):
    count = 0
    start_time = None
    finish_tag = 0
    
    while True:
        data = to_output_queue.get()
        if start_time is None:
            start_time = time.time()
        if data is None:
            finish_tag += 1
            if finish_tag == num_workers:
                print("End")
                break
            else:
                continue
        else:
            with open(OUTPUT_PATH, 'a') as f:
                try:
                    json.dump(data, f)
                    f.write('\n')
                except:
                    continue
            
            count += 1
            if count % 100 == 0:
                end_time = time.time()
                print(count)
                print(f"Spend:{(end_time-start_time)} s")
        

if __name__ == "__main__":
    import sys

    output_dir = os.path.dirname(os.path.abspath(OUTPUT_PATH))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    open(OUTPUT_PATH, "w").close()

    to_tokenize_queue = multiprocessing.Queue(maxsize=100000)
    to_output_queue = multiprocessing.Queue(maxsize=100000)
    
    # start
    reader_process = multiprocessing.Process(target=read_data, args=(INPUT_PATH, to_tokenize_queue))
    tokenizer_processes = [multiprocessing.Process(target=tokenize_data, args=(to_tokenize_queue, to_output_queue, rank)) for rank in range(num_workers)]
    output_process = multiprocessing.Process(target=output_data, args=(to_output_queue,))
    
    reader_process.start()
    for p in tokenizer_processes:
        p.start()
    output_process.start()

    start_time =  time.time()
    reader_process.join()
    for p in tokenizer_processes:
        p.join()
    output_process.join()
    end_time = time.time()
    print(end_time-start_time)
