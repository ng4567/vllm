import gc
import time
from dataclasses import dataclass, field
import csv
import torch
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from transformers import AutoTokenizer
from gpu_gpu_kv_offload_test import *
from memory_calculator import *
import os
huggingface_api_key = os.getenv("hf_token")

BLOCK_SIZE = 16
EVICTION_POLICY = "lru"
DEST_GPU_ID = 1
NUM_BLOCKS = 10000
NUM_TRIALS = 1

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

with open("prompts_shared_prefix.csv") as f:
    reader = csv.DictReader(f)
    prompts_shared_prefix = [row["prompt"] for row in reader]

def calculate_gpu_mem_util(gpu_id: int, model_size_gb: float, buffer_gb: float = 0.01) -> float:
    """
    Calculate the GPU memory utilization for vLLM.
    
    Args:
        gpu_id: GPU device ID
        model_size_gb: Model size in GB
        buffer_gb: Buffer to leave free (default 2 GB for overhead)
    
    Returns:
        Memory utilization fraction (0.0 to 0.95)
    """
    torch.cuda.synchronize(gpu_id)
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    total_gb = total_bytes / (1024**3)
    
    # Calculate utilization: (model_size + buffer) / total_memory
    # This tells vLLM how much of the GPU to use for the model + KV cache
    required_gb = model_size_gb + buffer_gb
    utilization = required_gb / total_gb
    
    # Clamp between 0.1 and 0.95 (vLLM needs some headroom)
    utilization = max(0.1, min(0.95, utilization))
    
    print(f"GPU {gpu_id}: {total_gb:.1f} GB total, model needs {model_size_gb:.2f} GB, setting utilization to {utilization:.2%}")
    return utilization

configs = {
    # Baseline - small scale
    "small_fb": {
        "huggingface_model_name": "facebook/opt-125m",
        "model_parameters_in_billions": 0.125,
        "avg_input_length": None,
        "avg_output_length": 100,
        "max_num_sequences": 100,
        "num_gpu_blocks": NUM_BLOCKS,
        "max_tokens": 100,
        "dest_gpu_id": DEST_GPU_ID,
        "gpu_mem_util": None,
        "memory_usage": None,
        "gpu_offloading_stats": [],
        "cpu_offloading_stats": [],
        "runtime_gpu": [],
        "runtime_cpu": [],
    },
    # Medium - more sequences
    "medium_fb": {
        "huggingface_model_name": "facebook/opt-125m",
        "model_parameters_in_billions": 0.125,
        "avg_input_length": None,
        "avg_output_length": 200,
        "max_num_sequences": 500,
        "num_gpu_blocks": NUM_BLOCKS,
        "max_tokens": 200,
        "dest_gpu_id": DEST_GPU_ID,
        "gpu_mem_util": None,
        "memory_usage": None,
        "gpu_offloading_stats": [],
        "cpu_offloading_stats": [],
        "runtime_gpu": [],
        "runtime_cpu": [],
    },
    # Large - stress test with many sequences and longer output
    "large_fb": {
        "huggingface_model_name": "facebook/opt-125m",
        "model_parameters_in_billions": 0.125,
        "avg_input_length": None,
        "avg_output_length": 500,
        "max_num_sequences": 1500,
        "num_gpu_blocks": NUM_BLOCKS,
        "max_tokens": 500,
        "dest_gpu_id": DEST_GPU_ID,
        "gpu_mem_util": None,
        "memory_usage": None,
        "gpu_offloading_stats": [],
        "cpu_offloading_stats": [],
        "runtime_gpu": [],
        "runtime_cpu": [],
    },
    # XLarge - maximum stress
    "xlarge_fb": {
        "huggingface_model_name": "facebook/opt-125m",
        "model_parameters_in_billions": 0.125,
        "avg_input_length": None,
        "avg_output_length": 1000,
        "max_num_sequences": 3000,
        "num_gpu_blocks": NUM_BLOCKS,
        "max_tokens": 1000,
        "dest_gpu_id": DEST_GPU_ID,
        "gpu_mem_util": None,
        "memory_usage": None,
        "gpu_offloading_stats": [],
        "cpu_offloading_stats": [],
        "runtime_gpu": [],
        "runtime_cpu": [],
    },
}

def init_configs() -> None:
    for config_name, config in configs.items():
        print(f"Running KV Offload Test for Config: {config_name}")
        tokenizer = AutoTokenizer.from_pretrained(config["huggingface_model_name"])
        # Tokenize each prompt and count tokens
        total_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)
        config["avg_input_length"] = total_tokens // len(prompts)
        print(f"Average input length: {config['avg_input_length']} tokens (from {len(prompts)} prompts)")
        mem_usage = get_model_size(
            model_name=config["huggingface_model_name"],
            model_params_billion=config["model_parameters_in_billions"],
            avg_input_length=config["avg_input_length"],
            avg_output_length=config["avg_output_length"],
            max_num_sequences=config["max_num_sequences"],
            verbose=False
        )
        config["memory_usage"] = mem_usage
        config["gpu_mem_util"] = calculate_gpu_mem_util(config["dest_gpu_id"], mem_usage.model_weight_memory)
        print(f"Memory Usage for Config {config_name}: {config['memory_usage'].total_memory_usage:.2f} GB\n")

def run_tests(prompts: list) -> None:
    for config_name, config in configs.items():
        llm_gpu = None
        llm_cpu = None
        try:
            llm_gpu = LLM(
                model=config["huggingface_model_name"],
                kv_transfer_config=KVTransferConfig(
                    kv_connector="OffloadingConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={
                        "spec_name": "GPUOffloadingSpec",
                        "num_gpu_blocks": NUM_BLOCKS,
                        "dest_gpu_id": DEST_GPU_ID,
                        "block_size": BLOCK_SIZE,
                        "eviction_policy": EVICTION_POLICY,
                    },
                ),
                gpu_memory_utilization=config["gpu_mem_util"],
            )
            
            for i in range(NUM_TRIALS):
                start = time.perf_counter()
                llm_gpu.generate(prompts, sampling_params=SamplingParams(max_tokens=config["max_tokens"]))
                config["runtime_gpu"].append(time.perf_counter() - start)
            config["gpu_offloading_stats"].append(get_offloading_stats(llm_gpu))                
        except Exception as e:
            print(f"Error running test for config {config_name}: {e}")
            if llm_gpu is not None:
                shutdown_llm(llm_gpu)
            config["gpu_offloading_stats"].append({})
            config["runtime_gpu"].append(e)
            
        try:
            llm_cpu = LLM(
                model=config["huggingface_model_name"],
                kv_transfer_config=KVTransferConfig(
                    kv_connector="OffloadingConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={
                        "spec_name": "GPUOffloadingSpec",
                        "num_gpu_blocks": NUM_BLOCKS,
                        "dest_gpu_id": DEST_GPU_ID,
                        "block_size": BLOCK_SIZE,
                        "eviction_policy": EVICTION_POLICY,
                    },
                ),
                gpu_memory_utilization=config["gpu_mem_util"],
            )

            for i in range(NUM_TRIALS):
                start = time.perf_counter()
                llm_cpu.generate(prompts, sampling_params=SamplingParams(max_tokens=config["max_tokens"]))
                config["runtime_cpu"].append(time.perf_counter() - start)
            config["cpu_offloading_stats"].append(get_offloading_stats(llm_cpu))
        except Exception as e:
                print(f"Error running test for config {config_name}: {e}")
                if llm_cpu is not None:
                    shutdown_llm(llm_cpu)
                config["cpu_offloading_stats"].append({})
                config["runtime_cpu"].append(e)

        
def print_results() -> None:
    for config_name, config_data in configs.items():
        print("\n" + "=" * 60)
        print(f"CONFIG: {config_name}")
        print("=" * 60)
        
        # Model info
        print(f"Model:              {config_data['huggingface_model_name']}")
        print(f"Model Size:         {config_data['model_parameters_in_billions']}B params")
        
        # Sequence parameters
        print(f"\n--- Sequence Parameters ---")
        print(f"Avg Input Length:   {config_data['avg_input_length']} tokens")
        print(f"Avg Output Length:  {config_data['avg_output_length']} tokens")
        print(f"Max Tokens:         {config_data['max_tokens']} tokens")
        print(f"Max Num Sequences:  {config_data['max_num_sequences']}")
        
        # Memory info
        print(f"\n--- Memory ---")
        print(f"GPU Memory Util:    {config_data['gpu_mem_util']:.2%}")
        if config_data['memory_usage']:
            print(f"Est. Total Memory:  {config_data['memory_usage'].total_memory_usage:.2f} GB")
        
        # Runtime results
        print(f"\n--- Runtime Results ({NUM_TRIALS} trials) ---")
        if config_data["runtime_gpu"] and not any(isinstance(r, Exception) for r in config_data["runtime_gpu"]):
            avg_gpu_runtime = sum(config_data["runtime_gpu"]) / len(config_data["runtime_gpu"])
            print(f"Avg GPU Runtime:    {avg_gpu_runtime:.2f} s")
        else:
            print(f"GPU Runtime:        Error or no data")
            avg_gpu_runtime = None
            
        if config_data["runtime_cpu"] and not any(isinstance(r, Exception) for r in config_data["runtime_cpu"]):
            avg_cpu_runtime = sum(config_data["runtime_cpu"]) / len(config_data["runtime_cpu"])
            print(f"Avg CPU Runtime:    {avg_cpu_runtime:.2f} s")
        else:
            print(f"CPU Runtime:        Error or no data")
            avg_cpu_runtime = None
        
        if avg_gpu_runtime and avg_cpu_runtime:
            speedup = avg_cpu_runtime - avg_gpu_runtime
            speedup_pct = (speedup / avg_cpu_runtime) * 100 if avg_cpu_runtime > 0 else 0
            print(f"\nSpeedup (GPU vs CPU): {speedup:.2f} s ({speedup_pct:+.1f}%)")
        
        # Offloading stats (last trial)
        print(f"\n--- Offloading Stats (last trial) ---")
        if config_data["gpu_offloading_stats"]:
            last_gpu_stats = config_data["gpu_offloading_stats"][-1]
            print(f"GPU Offload: offloaded={last_gpu_stats.get('blocks_offloaded', 0)}, "
                  f"reloaded={last_gpu_stats.get('blocks_reloaded', 0)}, "
                  f"evicted={last_gpu_stats.get('blocks_evicted', 0)}")
        if config_data["cpu_offloading_stats"]:
            last_cpu_stats = config_data["cpu_offloading_stats"][-1]
            print(f"CPU Offload: offloaded={last_cpu_stats.get('blocks_offloaded', 0)}, "
                  f"reloaded={last_cpu_stats.get('blocks_reloaded', 0)}, "
                  f"evicted={last_cpu_stats.get('blocks_evicted', 0)}")
        
        print("=" * 60)

if __name__ == "__main__":
    init_configs()
    run_tests(prompts)
    print_results()