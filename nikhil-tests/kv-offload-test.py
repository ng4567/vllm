import gc
import time
from dataclasses import dataclass, field
import csv
import torch
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from transformers import AutoTokenizer, AutoConfig
from gpu_gpu_kv_offload_test import *
from memory_calculator import *
from collections import defaultdict
import os
huggingface_api_key = os.getenv("hf_token")

BLOCK_SIZE = 16
EVICTION_POLICY = "arc"
DEST_GPU_ID = 1
NUM_TRIALS = 1

STAT_KEYS = [
    "blocks_offloaded",
    "blocks_reloaded",
    "blocks_allocated",
    "blocks_evicted",
    "blocks_freed",
]


def calculate_max_offload_blocks(
    dest_gpu_id: int,
    model_name: str,
    block_size: int = 16,
    headroom_fraction: float = 0.1,
    dtype_bytes: int = 2,  # bfloat16/float16
) -> int:
    """
    Calculate maximum number of KV cache blocks that can fit on the destination GPU.
    
    Args:
        dest_gpu_id: GPU ID to offload to
        model_name: HuggingFace model name to get config from
        block_size: Number of tokens per block
        headroom_fraction: Fraction of memory to leave free (default 10%)
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32)
    
    Returns:
        Maximum number of blocks that can fit on the destination GPU
    """
    # Get model config to determine KV cache dimensions
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Calculate bytes per KV cache block
    # Each block stores: 2 (K+V) × num_kv_heads × head_dim × num_layers × dtype_bytes × block_size
    bytes_per_block = 2 * num_kv_heads * head_dim * num_layers * dtype_bytes * block_size
    
    # Get available memory on destination GPU
    torch.cuda.synchronize(dest_gpu_id)
    free_bytes, total_bytes = torch.cuda.mem_get_info(dest_gpu_id)
    
    # Leave headroom
    usable_bytes = free_bytes * (1 - headroom_fraction)
    
    max_blocks = int(usable_bytes / bytes_per_block)
    
    print(f"\n=== Max Offload Blocks Calculation ===")
    print(f"Model: {model_name}")
    print(f"  Layers: {num_layers}, KV Heads: {num_kv_heads}, Head Dim: {head_dim}")
    print(f"  Block size: {block_size} tokens")
    print(f"  Bytes per block: {bytes_per_block / 1024:.1f} KB")
    print(f"Destination GPU {dest_gpu_id}:")
    print(f"  Total: {total_bytes / 1e9:.1f} GB")
    print(f"  Free: {free_bytes / 1e9:.1f} GB")
    print(f"  Usable (with {headroom_fraction:.0%} headroom): {usable_bytes / 1e9:.1f} GB")
    print(f"  Max offload blocks: {max_blocks:,}")
    print(f"======================================\n")
    
    return max_blocks


# Will be computed per model
NUM_BLOCKS = 10000  # Default, will be updated per model

# ============================================================
# Prompt Sets
# ============================================================
PROMPT_SETS = {
    "unique": "prompts_unique.csv",
    "shared_prefix": "prompts_shared_prefix.csv",
    "default": "prompts.csv",
}

def load_prompts(prompt_set: str) -> list[str]:
    """Load prompts from a CSV file based on prompt set name."""
    if prompt_set in PROMPT_SETS:
        filepath = PROMPT_SETS[prompt_set]
    else:
        # Assume it's a direct file path
        filepath = prompt_set
    
    with open(filepath) as f:
        reader = csv.DictReader(f)
        res = [row["prompt"] for row in reader]
        return res

def calculate_gpu_mem_util(
    gpu_id: int, 
    model_weight_gb: float, 
    activation_gb: float = 0.0,
    min_kv_cache_gb: float = 0.1,
    cuda_overhead_gb: float = 0.25
) -> float:
    """
    Calculate GPU memory utilization to force KV offloading.
    
    We allocate just enough for:
    - Model weights (required)
    - Minimal KV cache working set (required for operation)
    - Activation memory for forward pass (required)
    - CUDA/PyTorch overhead (required)
    
    This forces most KV cache to be offloaded to secondary storage.
    
    Args:
        gpu_id: GPU device ID
        model_weight_gb: Model weights size in GB
        activation_gb: Activation memory needed during forward pass
        min_kv_cache_gb: Minimum KV cache on GPU (working set)
        cuda_overhead_gb: CUDA context and PyTorch allocator overhead
    
    Returns:
        Memory utilization fraction (0.0 to 0.95)
    """
    torch.cuda.synchronize(gpu_id)
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    total_gb = total_bytes / (1024**3)
    
    # Minimum memory needed to run at all
    required_gb = (
        model_weight_gb +      # Model weights (non-negotiable)
        activation_gb +        # Forward pass activations  
        min_kv_cache_gb +      # Minimal KV cache working set
        cuda_overhead_gb       # CUDA context, allocator fragmentation
    )
    
    utilization = required_gb / total_gb
    
    # Clamp between 0.1 and 0.95
    utilization = max(0.1, min(0.95, utilization))
    
    print(f"GPU {gpu_id}: {total_gb:.1f} GB total")
    print(f"  Model weights: {model_weight_gb:.2f} GB")
    print(f"  Activations:   {activation_gb:.2f} GB")
    print(f"  Min KV cache:  {min_kv_cache_gb:.2f} GB")
    print(f"  CUDA overhead: {cuda_overhead_gb:.2f} GB")
    print(f"  Total needed:  {required_gb:.2f} GB → utilization: {utilization:.2%}")
    
    return utilization

# ============================================================
# Config Generation
# ============================================================
MAX_TOKENS = 250  # Global max tokens for all configs

# Size tiers: name -> max_num_sequences
SIZE_TIERS = {
    #"test": 2,
    #"xsmall": 50,
#    "small": 100,
#    "medium": 200,
    "large": 300,
}

# Prompt sets to test
PROMPT_SET_NAMES = ["unique", "shared_prefix"]

def generate_configs(models: dict[str, float]) -> dict:
    """
    Generate test configurations for each model across all size tiers and prompt sets.
    
    Args:
        models: Dict mapping model_name -> param_size_billions
    
    Returns:
        Dict of config_name -> config dict
    """
    configs = {}
    
    for model_name, param_size in models.items():
        # Create a short name for the model (e.g., "opt-125m" from "facebook/opt-125m")
        short_name = model_name.split("/")[-1].lower()
        
        for prompt_set in PROMPT_SET_NAMES:
            prompt_abbrev = "uniq" if prompt_set == "unique" else "shared"
            
            for size_name, max_seqs in SIZE_TIERS.items():
                config_name = f"{short_name}_{size_name}_{prompt_abbrev}"
                configs[config_name] = {
                    "huggingface_model_name": model_name,
                    "model_parameters_in_billions": param_size,
                    "prompt_set": prompt_set,
                    "prompts": None,
                    "avg_input_length": None,
                    "avg_output_length": MAX_TOKENS,
                    "max_num_sequences": max_seqs,
                    "num_gpu_blocks": NUM_BLOCKS,
                    "max_tokens": MAX_TOKENS,
                    "dest_gpu_id": DEST_GPU_ID,
                    "gpu_mem_util": None,
                    "memory_usage": None,
                    "gpu_offloading_stats": [],
                    "cpu_offloading_stats": [],
                    "runtime_gpu": [],
                    "runtime_cpu": [],
                }
    
    return configs

# Will be populated by generate_configs() in main
configs = {}

# Cache for max blocks per model (calculated once per model)
_max_blocks_cache: dict[str, int] = {}

def init_configs() -> None:
    for config_name, config in configs.items():
        print(f"Initializing Config: {config_name}")
        
        model_name = config["huggingface_model_name"]
        
        # Calculate max offload blocks for this model (cached per model)
        if model_name not in _max_blocks_cache:
            _max_blocks_cache[model_name] = calculate_max_offload_blocks(
                dest_gpu_id=config["dest_gpu_id"],
                model_name=model_name,
                block_size=BLOCK_SIZE,
            )
        config["num_gpu_blocks"] = _max_blocks_cache[model_name]
        
        # Load prompts for this config
        config["prompts"] = load_prompts(config["prompt_set"])
        prompts = config["prompts"]
        print(f"Loaded {len(prompts)} prompts from '{config['prompt_set']}'")
        
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
            huggingface_api_key=huggingface_api_key,
            verbose=False
        )
        config["memory_usage"] = mem_usage
        config["gpu_mem_util"] = calculate_gpu_mem_util(
            gpu_id=config["dest_gpu_id"], 
            model_weight_gb=mem_usage.model_weight_memory,
            activation_gb=mem_usage.activation_memory,
            min_kv_cache_gb=2.0,      # Minimal working set for KV cache on GPU
            cuda_overhead_gb=2.0       # CUDA context + allocator overhead
        )
        print(f"Full KV cache would need: {mem_usage.kv_cache_memory:.2f} GB (will be offloaded)")
        print(f"Memory Usage for Config {config_name}: {config['memory_usage'].total_memory_usage:.2f} GB\n")

def run_tests() -> None:
    for config_name, config in configs.items():
        prompts = config["prompts"]
        num_gpu_blocks = config["num_gpu_blocks"]
        print(f"\n{'='*60}")
        print(f"Running tests for: {config_name} ({len(prompts)} prompts)")
        print(f"{'='*60}")
        
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
                        "num_gpu_blocks": num_gpu_blocks,
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
            # For CPU offloading, use same number of blocks as GPU
            # (CPU RAM is larger but allocation is slower)
            num_cpu_blocks = num_gpu_blocks
            print(f"CPU offloading with {num_cpu_blocks:,} blocks")
            llm_cpu = LLM(
                model=config["huggingface_model_name"],
                kv_transfer_config=KVTransferConfig(
                    kv_connector="OffloadingConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={
                        "spec_name": "CPUOffloadingSpec",
                        "num_cpu_blocks": num_cpu_blocks,
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
    """Print results grouped by model in a comparable table format."""
    
    # Group configs by model
    models_configs = {}
    for config_name, config_data in configs.items():
        model_name = config_data["huggingface_model_name"]
        if model_name not in models_configs:
            models_configs[model_name] = []
        models_configs[model_name].append((config_name, config_data))
    
    # Print results for each model
    for model_name, model_configs in models_configs.items():
        param_size = model_configs[0][1]["model_parameters_in_billions"]
        
        print("\n" + "=" * 130)
        print(f"MODEL: {model_name} ({param_size}B params)")
        print("=" * 130)
        
        # Table header
        header = f"{'Config':<25} {'Prompts':<10} {'MaxSeq':<8} {'AvgIn':<7} {'GPU(s)':<10} {'CPU(s)':<10} {'Speedup':<12} {'Offload':<20} {'Reload':<10}"
        print(header)
        print("-" * 130)
        
        # Sort configs by size tier then prompt set for consistent ordering
        size_order = {"xsmall": 0, "small": 1, "medium": 2, "large": 3}
        prompt_order = {"unique": 0, "shared_prefix": 1}
        
        sorted_configs = sorted(model_configs, key=lambda x: (
            size_order.get(x[0].split("_")[-2], 99),
            prompt_order.get(x[1]["prompt_set"], 99)
        ))
        
        for config_name, config_data in sorted_configs:
            # Get runtime data
            if config_data["runtime_gpu"] and not any(isinstance(r, Exception) for r in config_data["runtime_gpu"]):
                avg_gpu = sum(config_data["runtime_gpu"]) / len(config_data["runtime_gpu"])
                gpu_str = f"{avg_gpu:.2f}"
            else:
                avg_gpu = None
                gpu_str = "Error"
            
            if config_data["runtime_cpu"] and not any(isinstance(r, Exception) for r in config_data["runtime_cpu"]):
                avg_cpu = sum(config_data["runtime_cpu"]) / len(config_data["runtime_cpu"])
                cpu_str = f"{avg_cpu:.2f}"
            else:
                avg_cpu = None
                cpu_str = "Error"
            
            # Speedup
            if avg_gpu and avg_cpu:
                speedup = avg_cpu - avg_gpu
                speedup_pct = (speedup / avg_cpu) * 100 if avg_cpu > 0 else 0
                speedup_str = f"{speedup:+.2f}s ({speedup_pct:+.1f}%)"
            else:
                speedup_str = "N/A"
            
            # Offloading stats (GPU run)
            if config_data["gpu_offloading_stats"] and config_data["gpu_offloading_stats"][-1]:
                stats = config_data["gpu_offloading_stats"][-1]
                offload_str = f"{stats.get('blocks_offloaded', 0)}"
                reload_str = f"{stats.get('blocks_reloaded', 0)}"
            else:
                offload_str = "N/A"
                reload_str = "N/A"
            
            # Prompt info
            prompt_abbrev = "uniq" if config_data["prompt_set"] == "unique" else "shared"
            num_prompts = len(config_data["prompts"]) if config_data["prompts"] else 0
            
            row = f"{config_name:<25} {prompt_abbrev:<10} {config_data['max_num_sequences']:<8} {config_data['avg_input_length'] or 0:<7} {gpu_str:<10} {cpu_str:<10} {speedup_str:<12} {offload_str:<20} {reload_str:<10}"
            print(row)
        
        print("=" * 130)
    
    # Print summary
    print("\n" + "#" * 130)
    print("SUMMARY")
    print("#" * 130)
    print(f"Total configs tested: {len(configs)}")
    print(f"Models: {list(models_configs.keys())}")
    print(f"Size tiers: {list(SIZE_TIERS.keys())}")
    print(f"Prompt sets: {PROMPT_SET_NAMES}")
    print(f"Max tokens: {MAX_TOKENS}")
    print("#" * 130)

if __name__ == "__main__":
    # Define models to test: model_name -> param_size_billions
    models = {
        #"facebook/opt-125m": 0.125,
        "mistralai/Mistral-7B-Instruct-v0.1": 7.0,
        #"deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 71.0,
    }
    
    # Generate configs for all models (both prompt sets)
    configs.update(generate_configs(models))
    
    print(f"Generated {len(configs)} configs:")
    for name in sorted(configs.keys()):
        print(f"  - {name}")
    print()
    
    init_configs()
    run_tests()
    print_results()