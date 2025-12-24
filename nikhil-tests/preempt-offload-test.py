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
os.environ.setdefault("VLLM_SKIP_MEMORY_CHECK", "1")
huggingface_api_key = os.getenv("hf_token")
# ============================================================
# Customizable Test Parameters (CHECK BEFORE RUN!!)
# ============================================================
BLOCK_SIZE = 16 #vllm default block size
DEST_GPU_ID = 1
NUM_BLOCKS = 10000  # Default, will be updated per model
NUM_TRIALS = 1
NUM_PROMPTS = 1000 #number of prompts to take from the prompt set
MAX_TOKENS = 6000  # Global max tokens for all configs
PROMPT_SET_NAMES = ["unique"] # ["unique", "shared_prefix"] # Prompt sets to test
EVICTION_POLICIES = ["lru", 'arc'] #["lru", "arc"] List of eviction policies to test; each policy is run separately and reported separately.
TEST_GPU = True # Whether to run GPU offloading tests
TEST_CPU = True  # Whether to run CPU offloading tests
NUM_TOKENS_BEFORE_PREMPTION = 20

# Size tiers: name -> max_num_sequences
SIZE_TIERS = {
   #"xsmall": 50,
   #"small": 100,
    "medium": 200,
    "large": 300,
   "xlarge": 500,
}
# Define models to test: model_name -> param_size_billions
models = {
    #"facebook/opt-125m": 0.125,
    "mistralai/Mistral-7B-Instruct-v0.1": 7.0,
    #"deepseek-ai/DeepSeek-Coder-V2-Lite-Base": 16.0,
}

STAT_KEYS = [
    "blocks_offloaded",
    "blocks_reloaded",
    "blocks_allocated",
    "blocks_evicted",
    "blocks_freed",
]
# CSV output configuration
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_FIELDNAMES = [
    "model",
    "config",
    "size",
    "prompt_set",
    "policy",
    "backend",
    "trial_index",
    "runtime_s",
    "error",
    "throughput_tok_s",
    "compute_used_gb",
    "compute_total_gb",
    "dest_used_gb",
    "dest_total_gb",
    "gpu_mem_util",
    "offload_kv_capacity_gb",
    "max_seq",
    "avg_input_length",
    "max_tokens",
    "num_gpu_blocks",
    "dest_gpu_id",
    "gpu_mem_util",
    "preempt_after_tokens",
    "blocks_offloaded",
    "blocks_reloaded",
    "blocks_allocated",
    "blocks_evicted",
    "blocks_freed",
]
CSV_PATH: str | None = None
# ============================================================
# 
# ============================================================
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
    global _block_bytes_cache
    # Get model config to determine KV cache dimensions
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Calculate bytes per KV cache block
    # Each block stores: 2 (K+V) × num_kv_heads × head_dim × num_layers × dtype_bytes × block_size
    bytes_per_block = 2 * num_kv_heads * head_dim * num_layers * dtype_bytes * block_size
    _block_bytes_cache[model_name] = bytes_per_block
    
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

# ============================================================
# Prompt Sets
# ============================================================
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
PROMPT_SETS = {
    "unique": os.path.join(PROMPT_DIR, "prompts_unique.csv"),
    "shared_prefix": os.path.join(PROMPT_DIR, "prompts_shared_prefix.csv"),
    "default": os.path.join(PROMPT_DIR, "prompts.csv"),
}

def load_prompts(prompt_set: str, num_prompts: int = NUM_PROMPTS) -> list[str]:
    """Load prompts from a CSV file based on prompt set name."""
    if prompt_set in PROMPT_SETS:
        filepath = PROMPT_SETS[prompt_set]
    else:
        # Assume it's a direct file path; if relative, resolve against PROMPT_DIR
        filepath = prompt_set
        if not os.path.isabs(filepath):
            filepath = os.path.join(PROMPT_DIR, filepath)
    
    with open(filepath) as f:
        reader = csv.DictReader(f)
        res = [row["prompt"] for row in reader]
        return res[:num_prompts]

def calculate_gpu_mem_util(
    gpu_id: int, 
    model_weight_gb: float, 
    activation_gb: float = 0.0,
    min_kv_cache_gb: float = 0.1,
    cuda_overhead_gb: float = 0.25,
    model_name: str = "",
) -> float:
    
    assert model_name in models.keys(), f"Model {model_name} not in models dict"

    if model_name == "mistralai/Mistral-7B-Instruct-v0.1":
        return 0.3
    elif model_name == "facebook/opt-125m":
        return 0.032
    
    raise ValueError(f"GPU memory utilization not set for model {model_name}")
    
    # """
    # Calculate GPU memory utilization to force KV offloading.
    
    # We allocate just enough for:
    # - Model weights (required)
    # - Minimal KV cache working set (required for operation)
    # - Activation memory for forward pass (required)
    # - CUDA/PyTorch overhead (required)
    
    # This forces most KV cache to be offloaded to secondary storage.
    
    # Args:
    #     gpu_id: GPU device ID
    #     model_weight_gb: Model weights size in GB
    #     activation_gb: Activation memory needed during forward pass
    #     min_kv_cache_gb: Minimum KV cache on GPU (working set)
    #     cuda_overhead_gb: CUDA context and PyTorch allocator overhead
    
    # Returns:
    #     Memory utilization fraction (0.0 to 0.95)
    # """
    # torch.cuda.synchronize(gpu_id)
    # free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    # total_gb = total_bytes / (1024**3)
    
    # # Minimum memory needed to run at all
    # required_gb = (
    #     model_weight_gb +      # Model weights (non-negotiable)
    #     activation_gb +        # Forward pass activations  
    #     min_kv_cache_gb +      # Minimal KV cache working set
    #     cuda_overhead_gb       # CUDA context, allocator fragmentation
    # )
    
    # utilization = required_gb / total_gb
    
    # print(f"GPU {gpu_id}: {total_gb:.1f} GB total")
    # print(f"  Model weights: {model_weight_gb:.2f} GB")
    # print(f"  Activations:   {activation_gb:.2f} GB")
    # print(f"  Min KV cache:  {min_kv_cache_gb:.2f} GB")
    # print(f"  CUDA overhead: {cuda_overhead_gb:.2f} GB")
    # print(f"  Total needed:  {required_gb:.2f} GB → utilization: {utilization:.2%}")
    
    # return utilization

# ============================================================
# Config Generation
# ============================================================


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
                for policy in EVICTION_POLICIES:
                    config_name = f"{short_name}_{size_name}_{prompt_abbrev}_{policy}"
                    configs[config_name] = {
                        "huggingface_model_name": model_name,
                        "model_parameters_in_billions": param_size,
                        "prompt_set": prompt_set,
                        "prompts": None,
                        "avg_input_length": None,
                        "avg_output_length": MAX_TOKENS,
                        "max_num_sequences": max_seqs,
                        "size_tier": size_name,
                        "num_gpu_blocks": NUM_BLOCKS,
                        "max_tokens": MAX_TOKENS,
                        "dest_gpu_id": DEST_GPU_ID,
                        "gpu_mem_util": None,
                        "memory_usage": None,
                        "gpu_offloading_stats": [],
                        "cpu_offloading_stats": [],
                        "runtime_gpu": [],
                        "runtime_cpu": [],
                        "eviction_policy": policy,
                    }
    
    return configs

# Will be populated by generate_configs() in main
configs = {}

# Cache for max blocks per model (calculated once per model)
_max_blocks_cache: dict[str, int] = {}
_block_bytes_cache: dict[str, int] = {}
def init_csv_writer() -> None:
    """Initialize CSV file with header for incremental writes."""
    global CSV_PATH
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    CSV_PATH = os.path.join(RESULTS_DIR, f"{timestamp}.csv")
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()


def append_trial_row(row: dict) -> None:
    """Append a single trial row to the CSV file."""
    if CSV_PATH is None:
        raise RuntimeError("CSV writer not initialized")
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


def build_trial_row(
    config_name: str,
    config_data: dict,
    backend: str,
    trial_index: int,
    runtime: float | Exception,
    stats: dict,
    total_tokens: int,
    offload_kv_capacity_gb: float | None,
) -> dict:
    """Construct a per-trial CSV row."""
    model_name = config_data["huggingface_model_name"]
    size_tier = config_data.get("size_tier", config_name.split("_")[-2] if "_" in config_name else "unknown")
    prompt_set = config_data["prompt_set"]
    policy = config_data.get("eviction_policy", "lru")
    mem_snapshot = config_data.get("mem_snapshot", {})

    throughput = None
    runtime_val = None if isinstance(runtime, Exception) else runtime
    if runtime_val and runtime_val > 0 and total_tokens > 0:
        throughput = total_tokens / runtime_val

    row = {
        "model": model_name,
        "config": config_name,
        "size": size_tier,
        "prompt_set": prompt_set,
        "policy": policy,
        "backend": backend,
        "trial_index": trial_index,
        "runtime_s": runtime_val,
        "error": str(runtime) if isinstance(runtime, Exception) else "",
        "throughput_tok_s": throughput,
        "compute_used_gb": mem_snapshot.get("compute_used_gb"),
        "compute_total_gb": mem_snapshot.get("compute_total_gb"),
        "dest_used_gb": mem_snapshot.get("dest_used_gb"),
        "dest_total_gb": mem_snapshot.get("dest_total_gb"),
        "gpu_mem_util": config_data.get("gpu_mem_util"),
        "offload_kv_capacity_gb": offload_kv_capacity_gb,
        "max_seq": config_data["max_num_sequences"],
        "avg_input_length": config_data["avg_input_length"] or 0,
        "max_tokens": config_data["max_tokens"],
        "num_gpu_blocks": config_data["num_gpu_blocks"],
        "dest_gpu_id": config_data["dest_gpu_id"],
        "blocks_offloaded": stats.get("blocks_offloaded"),
        "blocks_reloaded": stats.get("blocks_reloaded"),
        "blocks_allocated": stats.get("blocks_allocated"),
        "blocks_evicted": stats.get("blocks_evicted"),
        "blocks_freed": stats.get("blocks_freed"),
        "preempt_after_tokens": NUM_TOKENS_BEFORE_PREMPTION,
    }
    return row

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
            min_kv_cache_gb=0.25,      # Minimal working set for KV cache on GPU
            cuda_overhead_gb=0.25,       # CUDA context + allocator overhead
            model_name=config["huggingface_model_name"],
        )
        print(f"Full KV cache would need: {mem_usage.kv_cache_memory:.2f} GB (will be offloaded)")
        print(f"Memory Usage for Config {config_name}: {config['memory_usage'].total_memory_usage:.2f} GB\n")

def run_tests() -> None:
    for config_name, config in configs.items():
        prompts = config["prompts"]
        num_gpu_blocks = config["num_gpu_blocks"]
        policy = config.get("eviction_policy", "lru")
        prompt_count = len(prompts)
        total_tokens = ((config.get("avg_input_length") or 0) + config.get("max_tokens", 0)) * prompt_count
        block_bytes = _block_bytes_cache.get(config["huggingface_model_name"])
        offload_kv_capacity_gb = None
        if block_bytes is not None:
            offload_kv_capacity_gb = (config["num_gpu_blocks"] * block_bytes) / 1e9
        print(f"\n{'='*60}")
        print(f"Running tests for: {config_name} ({len(prompts)} prompts)")
        print(f"{'='*60}")
        
        llm_gpu = None
        llm_cpu = None
        if TEST_GPU:
            try:
                print(f"GPU offloading with {num_gpu_blocks:,} blocks")
                llm_gpu = LLM(
                    model=config["huggingface_model_name"],
                    kv_transfer_config=KVTransferConfig(
                        kv_connector="OffloadingConnector",
                        kv_role="kv_both",
                        kv_connector_extra_config={
                            "spec_name": "GPUOffloadingSpec",
                            "num_gpu_blocks": num_gpu_blocks,
                            "dest_gpu_id": config["dest_gpu_id"],
                            "eviction_policy": policy,
                        },
                    ),
                    gpu_memory_utilization=config["gpu_mem_util"],
                    preempt_after_tokens=NUM_TOKENS_BEFORE_PREMPTION,
                )
                    # Capture memory snapshot for compute and dest GPUs
                compute_device = torch.cuda.current_device()
                comp_free, comp_total = torch.cuda.mem_get_info(compute_device)
                dest_free, dest_total = torch.cuda.mem_get_info(config["dest_gpu_id"])
                config["mem_snapshot"] = {
                    "compute_device": compute_device,
                    "compute_free_gb": comp_free / 1e9,
                    "compute_total_gb": comp_total / 1e9,
                    "compute_used_gb": (comp_total - comp_free) / 1e9,
                    "dest_device": config["dest_gpu_id"],
                    "dest_free_gb": dest_free / 1e9,
                    "dest_total_gb": dest_total / 1e9,
                    "dest_used_gb": (dest_total - dest_free) / 1e9,
                }

                for i in range(NUM_TRIALS):
                    start = time.perf_counter()
                    llm_gpu.generate(prompts, sampling_params=SamplingParams(max_tokens=config["max_tokens"]))
                    config["runtime_gpu"].append(time.perf_counter() - start)
                    gpu_stats = get_offloading_stats(llm_gpu) or {}
                    append_trial_row(
                        build_trial_row(
                            config_name=config_name,
                            config_data=config,
                            backend="gpu",
                            trial_index=i + 1,
                            runtime=config["runtime_gpu"][-1],
                            stats=gpu_stats,
                            total_tokens=total_tokens,
                            offload_kv_capacity_gb=offload_kv_capacity_gb,
                        )
                    )
                # Save the latest stats for aggregate reporting
                config["gpu_offloading_stats"].append(gpu_stats if gpu_stats else {})
            except Exception as e:
                print(f"Error running test for config {config_name}: {e}")
                if llm_gpu is not None:
                    shutdown_llm(llm_gpu)
                config["gpu_offloading_stats"].append({})
                config["runtime_gpu"].append(e)
        if TEST_CPU:
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
                            "eviction_policy": policy,
                        },
                    ),
                    gpu_memory_utilization=config["gpu_mem_util"],
                    preempt_after_tokens=NUM_TOKENS_BEFORE_PREMPTION,
                )

                for i in range(NUM_TRIALS):
                    start = time.perf_counter()
                    llm_cpu.generate(prompts, sampling_params=SamplingParams(max_tokens=config["max_tokens"]))
                    config["runtime_cpu"].append(time.perf_counter() - start)
                    cpu_stats = get_offloading_stats(llm_cpu) or {}
                    append_trial_row(
                        build_trial_row(
                            config_name=config_name,
                            config_data=config,
                            backend="cpu",
                            trial_index=i + 1,
                            runtime=config["runtime_cpu"][-1],
                            stats=cpu_stats,
                            total_tokens=total_tokens,
                            offload_kv_capacity_gb=offload_kv_capacity_gb,
                        )
                    )
                # Save the latest stats for aggregate reporting
                config["cpu_offloading_stats"].append(cpu_stats if cpu_stats else {})
            except Exception as e:
                    print(f"Error running test for config {config_name}: {e}")
                    if llm_cpu is not None:
                        shutdown_llm(llm_cpu)
                    config["cpu_offloading_stats"].append({})
                    config["runtime_cpu"].append(e)

        
def print_results() -> None:
    """Collect results into a table and print with nice alignment (pandas if available)."""

    # Prepare rows
    rows = []
    trial_rows = []
    size_order = {"xsmall": 0, "small": 1, "medium": 2, "large": 3}
    prompt_order = {"unique": 0, "shared_prefix": 1}
    policy_order = {p: i for i, p in enumerate(EVICTION_POLICIES)}

    for config_name, config_data in configs.items():
        model_name = config_data["huggingface_model_name"]
        size_tier = config_data.get("size_tier", config_name.split("_")[-2] if "_" in config_name else "unknown")
        prompt_set = config_data["prompt_set"]
        policy = config_data.get("eviction_policy", "lru")
        mem_snapshot = config_data.get("mem_snapshot", {})

        # Runtimes
        avg_gpu = None
        avg_cpu = None
        if config_data["runtime_gpu"] and not any(isinstance(r, Exception) for r in config_data["runtime_gpu"]):
            avg_gpu = sum(config_data["runtime_gpu"]) / len(config_data["runtime_gpu"])
        if config_data["runtime_cpu"] and not any(isinstance(r, Exception) for r in config_data["runtime_cpu"]):
            avg_cpu = sum(config_data["runtime_cpu"]) / len(config_data["runtime_cpu"])

        speedup_s = None
        speedup_pct = None
        if avg_gpu is not None and avg_cpu is not None:
            speedup_s = avg_cpu - avg_gpu
            speedup_pct = (speedup_s / avg_cpu) * 100 if avg_cpu > 0 else 0.0

        prompt_count = len(config_data.get("prompts", [])) if config_data.get("prompts") is not None else 0
        total_tokens = ((config_data.get("avg_input_length") or 0) + config_data.get("max_tokens", 0)) * prompt_count
        gpu_throughput = None
        cpu_throughput = None
        if avg_gpu is not None and avg_gpu > 0 and total_tokens > 0:
            gpu_throughput = total_tokens / avg_gpu
        if avg_cpu is not None and avg_cpu > 0 and total_tokens > 0:
            cpu_throughput = total_tokens / avg_cpu

        # Offload stats (GPU run)
        offload_blocks = None
        reload_blocks = None
        if config_data["gpu_offloading_stats"] and config_data["gpu_offloading_stats"][-1]:
            stats = config_data["gpu_offloading_stats"][-1]
            offload_blocks = stats.get("blocks_offloaded")
            reload_blocks = stats.get("blocks_reloaded")

        block_bytes = _block_bytes_cache.get(model_name)
        offload_kv_capacity_gb = None
        if block_bytes is not None:
            offload_kv_capacity_gb = (config_data["num_gpu_blocks"] * block_bytes) / 1e9

        # Collect per-trial rows for CSV output (GPU)
        gpu_stats = config_data["gpu_offloading_stats"][-1] if config_data["gpu_offloading_stats"] else {}
        for idx, runtime in enumerate(config_data["runtime_gpu"]):
            gpu_throughput_trial = None
            if not isinstance(runtime, Exception) and runtime > 0 and total_tokens > 0:
                gpu_throughput_trial = total_tokens / runtime
            trial_rows.append({
                "model": model_name,
                "config": config_name,
                "size": size_tier,
                "prompt_set": prompt_set,
                "policy": policy,
                "backend": "gpu",
                "trial_index": idx + 1,
                "runtime_s": None if isinstance(runtime, Exception) else runtime,
                "error": str(runtime) if isinstance(runtime, Exception) else "",
                "throughput_tok_s": gpu_throughput_trial,
                "compute_used_gb": mem_snapshot.get("compute_used_gb"),
                "compute_total_gb": mem_snapshot.get("compute_total_gb"),
                "dest_used_gb": mem_snapshot.get("dest_used_gb"),
                "dest_total_gb": mem_snapshot.get("dest_total_gb"),
                "gpu_mem_util": config_data.get("gpu_mem_util"),
                "offload_kv_capacity_gb": offload_kv_capacity_gb,
                "max_seq": config_data["max_num_sequences"],
                "avg_input_length": config_data["avg_input_length"] or 0,
                "max_tokens": config_data["max_tokens"],
                "num_gpu_blocks": config_data["num_gpu_blocks"],
                "dest_gpu_id": config_data["dest_gpu_id"],
                "gpu_mem_util": config_data["gpu_mem_util"],
                "blocks_offloaded": gpu_stats.get("blocks_offloaded"),
                "blocks_reloaded": gpu_stats.get("blocks_reloaded"),
                "blocks_allocated": gpu_stats.get("blocks_allocated"),
                "blocks_evicted": gpu_stats.get("blocks_evicted"),
                "blocks_freed": gpu_stats.get("blocks_freed"),
            })

        # Collect per-trial rows for CSV output (CPU)
        cpu_stats = config_data["cpu_offloading_stats"][-1] if config_data["cpu_offloading_stats"] else {}
        for idx, runtime in enumerate(config_data["runtime_cpu"]):
            cpu_throughput_trial = None
            if not isinstance(runtime, Exception) and runtime > 0 and total_tokens > 0:
                cpu_throughput_trial = total_tokens / runtime
            trial_rows.append({
                "model": model_name,
                "config": config_name,
                "size": size_tier,
                "prompt_set": prompt_set,
                "policy": policy,
                "backend": "cpu",
                "trial_index": idx + 1,
                "runtime_s": None if isinstance(runtime, Exception) else runtime,
                "error": str(runtime) if isinstance(runtime, Exception) else "",
                "throughput_tok_s": cpu_throughput_trial,
                "compute_used_gb": mem_snapshot.get("compute_used_gb"),
                "compute_total_gb": mem_snapshot.get("compute_total_gb"),
                "dest_used_gb": mem_snapshot.get("dest_used_gb"),
                "dest_total_gb": mem_snapshot.get("dest_total_gb"),
                "gpu_mem_util": config_data.get("gpu_mem_util"),
                "offload_kv_capacity_gb": offload_kv_capacity_gb,
                "max_seq": config_data["max_num_sequences"],
                "avg_input_length": config_data["avg_input_length"] or 0,
                "max_tokens": config_data["max_tokens"],
                "num_gpu_blocks": config_data["num_gpu_blocks"],
                "dest_gpu_id": config_data["dest_gpu_id"],
                "gpu_mem_util": config_data["gpu_mem_util"],
                "blocks_offloaded": cpu_stats.get("blocks_offloaded"),
                "blocks_reloaded": cpu_stats.get("blocks_reloaded"),
                "blocks_allocated": cpu_stats.get("blocks_allocated"),
                "blocks_evicted": cpu_stats.get("blocks_evicted"),
                "blocks_freed": cpu_stats.get("blocks_freed"),
            })

        rows.append({
            "Model": model_name,
            "Size": size_tier,
            "Prompts": "uniq" if prompt_set == "unique" else "shared",
            "Policy": policy,
            "MaxSeq": config_data["max_num_sequences"],
            "AvgIn": config_data["avg_input_length"] or 0,
            "GPU_s": None if avg_gpu is None else round(avg_gpu, 2),
            "CPU_s": None if avg_cpu is None else round(avg_cpu, 2),
            "GPU_tok_s": None if gpu_throughput is None else round(gpu_throughput, 2),
            "CPU_tok_s": None if cpu_throughput is None else round(cpu_throughput, 2),
            "Compute_used_GB": mem_snapshot.get("compute_used_gb"),
            "Dest_used_GB": mem_snapshot.get("dest_used_gb"),
            "Offload_kv_capacity_GB": None if offload_kv_capacity_gb is None else round(offload_kv_capacity_gb, 3),
            "GPU_mem_util": config_data.get("gpu_mem_util"),
            "Speedup_s": None if speedup_s is None else round(speedup_s, 2),
            "Speedup_pct": None if speedup_pct is None else round(speedup_pct, 1),
            "Offload": offload_blocks if offload_blocks is not None else "N/A",
            "Reload": reload_blocks if reload_blocks is not None else "N/A",
            "Trials": len(config_data["runtime_gpu"]) if config_data["runtime_gpu"] else len(config_data["runtime_cpu"]),
        })

    # Sort rows for stable output
    rows.sort(key=lambda r: (
        r["Model"],
        size_order.get(r["Size"], 99),
        prompt_order.get("unique" if r["Prompts"] == "uniq" else "shared_prefix", 99),
        policy_order.get(r["Policy"], 99),
    ))

    def _print_table(rows_subset: list[dict], title: str) -> None:
        if not rows_subset:
            print(f"\nResults for {title}: (no rows)")
            return
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame(rows_subset)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 200)
            print(f"\nResults for {title} (pandas):")
            print(df.to_string(index=False))
        except Exception:
            cols = ["Model", "Config", "Size", "Prompts", "Policy", "MaxSeq", "AvgIn", "GPU_s", "CPU_s", "GPU_tok_s", "CPU_tok_s", "Compute_used_GB", "Dest_used_GB", "Offload_kv_capacity_GB", "GPU_mem_util", "Speedup_s", "Speedup_pct", "Offload", "Reload", "Trials"]
            str_rows = [[str(row.get(c, "")) for c in cols] for row in rows_subset]
            widths = [max(len(col), max((len(r[i]) for r in str_rows), default=0)) for i, col in enumerate(cols)]
            header = " ".join(col.ljust(widths[i]) for i, col in enumerate(cols))
            print(f"\nResults for {title}:")
            print(header)
            print("-" * len(header))
            for r in str_rows:
                print(" ".join(r[i].ljust(widths[i]) for i in range(len(cols))))

    # Print a separate table per eviction policy
    policies_present = sorted({r["Policy"] for r in rows}, key=lambda p: policy_order.get(p, 99))
    for pol in policies_present:
        subset = [r for r in rows if r["Policy"] == pol]
        _print_table(subset, f"policy={pol}")

    # Summary
    print("\n" + "#" * 80)
    print("SUMMARY")
    print("#" * 80)
    print(f"Total configs tested: {len(rows)}")
    print(f"Models: {sorted(set(r['Model'] for r in rows))}")
    print(f"Size tiers: {list(SIZE_TIERS.keys())}")
    print(f"Prompt sets: {PROMPT_SET_NAMES}")
    print(f"Eviction policies: {EVICTION_POLICIES}")
    print(f"Trials per config: {NUM_TRIALS}")
    print(f"Max tokens: {MAX_TOKENS}")
    print("#" * 80)

    if CSV_PATH is not None:
        print(f"Per-trial rows streamed to {CSV_PATH}")
    else:
        # Fallback: write once at the end if streaming was not initialized
        try:
            init_csv_writer()
            assert CSV_PATH is not None
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                for row in trial_rows:
                    writer.writerow(row)
            print(f"Per-trial rows written to {CSV_PATH}")
        except Exception as e:
            print(f"Failed to write CSV results: {e}")

if __name__ == "__main__":
    # Generate configs for all models (both prompt sets)
    configs.update(generate_configs(models))
    
    print(f"Generated {len(configs)} configs:")
    for name in sorted(configs.keys()):
        print(f"  - {name}")
    print()
    
    init_csv_writer()
    init_configs()
    run_tests()
    print_results()
