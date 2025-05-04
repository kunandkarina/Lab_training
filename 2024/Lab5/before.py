import os
import sys
import argparse
import json
import warnings
import logging
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
import transformers, datasets
from peft import PeftModel
from colorama import *

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GenerationConfig
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 生成訓練資料
def generate_training_data(data_point):
    """
    (1) Goal:
        - This function is used to transform a data point (input and output texts) to tokens that our model can read

    (2) Arguments:
        - data_point: dict, with field "instruction", "input", and "output" which are all str

    (3) Returns:
        - a dict with model's input tokens, attention mask that make our model causal, and corresponding output targets

    (3) Example:
        - If you construct a dict, data_point_1, with field "instruction", "input", and "output" which are all str, you can use the function like this:
            formulate_article(data_point_1)

    """
    # construct full input prompt
    prompt = f"""\
[INST] <<SYS>>
You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
<</SYS>>

{data_point["instruction"]}
{data_point["input"]}
[/INST]"""
    # count the number of input tokens
    len_user_prompt_tokens = (
        len(
            tokenizer(
                prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
                padding="max_length",
            )["input_ids"]
        ) - 1
    )
    # transform input prompt into tokens
    full_tokens = tokenizer(
        prompt + " " + data_point["output"] + "</s>",
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

# 進行生成回覆的評估
def evaluate(instruction, generation_config, max_len, input="", verbose=True):
    """
    (1) Goal:
        - This function is used to get the model's output given input strings

    (2) Arguments:
        - instruction: str, description of what you want model to do
        - generation_config: transformers.GenerationConfig object, to specify decoding parameters relating to model inference
        - max_len: int, max length of model's output
        - input: str, input string the model needs to solve the instruction, default is "" (no input)
        - verbose: bool, whether to print the mode's output, default is True

    (3) Returns:
        - output: str, the mode's response according to the instruction and the input

    (3) Example:
        - If you the instruction is "ABC" and the input is "DEF" and you want model to give an answer under 128 tokens, you can use the function like this:
            evaluate(instruction="ABC", generation_config=generation_config, max_len=128, input="DEF")

    """
    # construct full input prompt
    prompt = f"""\
[INST] <<SYS>>
You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
<</SYS>>

{instruction}
{input}
[/INST]"""
    # 將提示文本轉換為模型所需的數字表示形式
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    # 使用模型進行生成回覆
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
    )
    # 將生成的回覆解碼並印出
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        output = output.split("[/INST]")[1].replace("</s>", "").replace("<s>", "").replace("Assistant:", "").replace("Assistant", "").strip()
        if (verbose):
            print(output)

    return output

model_name = "MediaTek-Research/Breeze-7B-Instruct-v0_1"

cache_dir = "./cache"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# 從指定的模型名稱或路徑載入預訓練的語言模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config,
    low_cpu_mem_usage = True
)

# 創建 tokenizer 並設定結束符號 (eos_token)
logging.getLogger('transformers').setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)
tokenizer.pad_token = tokenizer.eos_token

# 設定模型推理時需要用到的decoding parameters
max_len = 128
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    num_beams=1,
    top_p=0.3,
    no_repeat_ngram_size=3,
    pad_token_id=2,
)

# demo examples
test_tang_list = ['相見時難別亦難，東風無力百花殘。', '重帷深下莫愁堂，臥後清宵細細長。', '芳辰追逸趣，禁苑信多奇。']

# get the model output for each examples
demo_before_finetune = []
for tang in test_tang_list:
  demo_before_finetune.append(f'模型輸入:\n以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。{tang}\n\n模型輸出:\n'+evaluate('以下是一首唐詩的第一句話，請用你的知識判斷並完成整首詩。', generation_config, max_len, tang, verbose = False))

# print and store the output to text file
for idx in range(len(demo_before_finetune)):
  print(f"Example {idx + 1}:")
  print(demo_before_finetune[idx])
  print("-" * 80)