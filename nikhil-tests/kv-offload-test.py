import gc
import time
from dataclasses import dataclass, field
import csv
import torch
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from gpu_gpu_kv_offload_test import *
from memory_calculator import *
import os
huggingface_api_key = os.getenv("hf_token")

STAT_KEYS = [
    "blocks_offloaded",
    "blocks_reloaded",
    "blocks_allocated",
    "blocks_evicted",
    "blocks_freed",
]

with open("prompts.csv") as f:
    reader = csv.DictReader(f)
    prompts = [row["prompt"] for row in reader]

configs = {
    "1": {
        "huggingface_model_name": "facebook/opt-125m",
        "model_parameters_in_billions": 0.125,
        "avg_input_length": 1500,
        "avg_output_length": 200,
        "max_num_sequences": 500,
        "memory_usage": None,
    } 
}

for config_name, config in configs.items():
    print(f"Running KV Offload Test for Config: {config_name}")
    mem_usage = get_model_size(
        model_name=config["huggingface_model_name"],
        model_params_billion=config["model_parameters_in_billions"],
        avg_input_length=config["avg_input_length"],
        avg_output_length=config["avg_output_length"],
        max_num_sequences=config["max_num_sequences"],
    )
    config["memory_usage"] = mem_usage


