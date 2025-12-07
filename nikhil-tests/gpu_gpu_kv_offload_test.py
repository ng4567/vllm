import gc
import time
from dataclasses import dataclass, field

import torch
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig

# ============================================================
# Configuration
# ============================================================

STAT_KEYS = [
    "blocks_offloaded",
    "blocks_reloaded",
    "blocks_allocated",
    "blocks_evicted",
    "blocks_freed",
]

MODELS = {
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.1",
        "gpu_memory_utilization": 0.195,
    },
    "opt": {
        "name": "facebook/opt-125m",
        "gpu_memory_utilization": 0.03,
    },
}

MODEL = "mistral"
MODEL_NAME = MODELS[MODEL]["name"]
GPU_MEMORY_UTILIZATION = MODELS[MODEL]["gpu_memory_utilization"]


DEST_GPU_ID = 1
BLOCK_SIZE = 16
EVICTION_POLICY = "lru"  # "arc" has a bug with touch()
NUM_BLOCKS = 10000

# Fixed prompts reused across runs to trigger prefix cache reloads
SHARED_PROMPTS = [
    (
        "In a distant future, humanity has colonized Mars. "
        "The following report summarizes daily operations of colony alpha. "
        "Please provide diagnostics and recommendations. "
        "Log entry: " + "status nominal. " * 100 +
        "Continue the mission report with next steps."
    ),
    (
        "The quantum computer completed its analysis of the stellar data. "
        "Here are the findings from sector 7G observations. "
        "Summary: " + "readings stable. " * 100 +
        "What conclusions can we draw from this data?"
    ),
    (
        "Captain's log, stardate 47634.44. We have entered the nebula. "
        "Sensor readings indicate unusual particle concentrations. "
        "Details: " + "anomaly detected. " * 100 +
        "Recommend next course of action."
    ),
]


# ============================================================
# Data Classes
# ============================================================

@dataclass
class RunResult:
    label: str
    num_requests: int
    duration: float
    tokens: int
    throughput: float
    delta: dict[str, int] = field(default_factory=dict)


# ============================================================
# Hardware Info
# ============================================================

def get_hardware_info() -> dict:
    """Gather hardware configuration info."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })
    return info


def print_hardware_info() -> None:
    """Print hardware configuration."""
    info = get_hardware_info()
    print("=" * 100)
    print("HARDWARE CONFIGURATION")
    print("=" * 100)
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"CUDA Version: {info['cuda_version']}")
    print(f"GPU Count: {info['gpu_count']}")
    print()
    for gpu in info["gpus"]:
        print(f"  GPU {gpu['id']}: {gpu['name']}")
        print(f"    Memory: {gpu['total_memory_gb']:.1f} GB")
        print(f"    Compute Capability: {gpu['compute_capability']}")
        print(f"    SM Count: {gpu['multi_processor_count']}")
    print()


def print_test_config() -> None:
    """Print test configuration."""
    print("=" * 100)
    print("TEST CONFIGURATION")
    print("=" * 100)
    print(f"Model: {MODEL_NAME}")
    print(f"GPU Memory Utilization: {GPU_MEMORY_UTILIZATION * 100:.0f}%")
    print(f"Offload Blocks: {NUM_BLOCKS}")
    print(f"Block Size: {BLOCK_SIZE} tokens")
    print(f"Eviction Policy: {EVICTION_POLICY}")
    print(f"Destination GPU ID (for GPU offload): {DEST_GPU_ID}")
    print()


# ============================================================
# LLM Setup
# ============================================================

def create_gpu_offloading_llm() -> LLM:
    config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "GPUOffloadingSpec",
            "num_gpu_blocks": NUM_BLOCKS,
            "dest_gpu_id": DEST_GPU_ID,
            "block_size": BLOCK_SIZE,
            "eviction_policy": EVICTION_POLICY,
        },
    )
    return LLM(
        model=MODEL_NAME,
        kv_transfer_config=config,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )


def create_cpu_offloading_llm() -> LLM:
    config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "xfloadingSpec",
            "num_cpu_blocks": NUM_BLOCKS,
            "block_size": BLOCK_SIZE,
            "eviction_policy": EVICTION_POLICY,
        },
    )
    return LLM(
        model=MODEL_NAME,
        kv_transfer_config=config,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )


def shutdown_llm(llm: LLM) -> None:
    """Shutdown LLM and free resources."""
    import time
    
    # Try to shutdown the engine core properly
    engine = getattr(llm, "llm_engine", None)
    if engine is not None:
        # Shutdown engine core client if available
        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            try:
                engine_core.shutdown()
            except Exception:
                pass
        
        executor = getattr(engine, "model_executor", None)
        if executor is not None and hasattr(executor, "shutdown"):
            try:
                executor.shutdown()
            except Exception:
                pass
    
    del llm
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Give time for processes to fully terminate and release memory
    time.sleep(3)
    
    # Force another cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Prompt Generation
# ============================================================

def generate_prompts(num_requests: int, reuse_prefix: bool = True) -> list[str]:
    """Generate prompts for inference."""
    assert num_requests > 0, "Number of requests must be greater than 0"
    if reuse_prefix:
        return [SHARED_PROMPTS[i % len(SHARED_PROMPTS)] for i in range(num_requests)]
    else:
        return [f"Unique request {i+1}: Tell me a story." for i in range(num_requests)]


# ============================================================
# Stats Collection
# ============================================================
def get_free_blocks(gpu_id: int, block_size: int) -> int:
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    return free_bytes // block_size


def get_offloading_stats(llm_instance: LLM) -> dict[str, int]:
    """Aggregate offloading stats from all workers."""
    stats_per_worker = llm_instance.llm_engine.collective_rpc("get_offloading_stats")
    aggregated: dict[str, int] = {key: 0 for key in STAT_KEYS}
    any_stats = False
    for worker_stats in stats_per_worker:
        if not worker_stats:
            continue
        any_stats = True
        for key in STAT_KEYS:
            aggregated[key] += worker_stats.get(key, 0)
    return aggregated if any_stats else {key: 0 for key in STAT_KEYS}


def compute_delta(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    """Compute the difference between two stat dictionaries."""
    return {key: current.get(key, 0) - previous.get(key, 0) for key in STAT_KEYS}


def count_tokens(outputs) -> int:
    """Count total tokens generated across all outputs."""
    total = 0
    for request_output in outputs:
        for seq_output in request_output.outputs:
            token_ids = getattr(seq_output, "token_ids", []) or []
            total += len(token_ids)
    return total


# ============================================================
# Inference
# ============================================================

def run_inference(
    llm: LLM,
    num_requests: int,
    reuse_prefix: bool = True,
    max_tokens: int = 100,
):
    """Run inference and return outputs."""
    prompts = generate_prompts(num_requests, reuse_prefix=reuse_prefix)
    outputs = llm.generate(
        prompts,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.8),
    )
    return outputs


def execute_single_run(
    llm: LLM,
    num_requests: int,
    reuse_prefix: bool,
    prev_stats: dict[str, int],
) -> tuple[RunResult, dict[str, int]]:
    """Execute a single inference run and return results."""
    start = time.perf_counter()
    outputs = run_inference(llm, num_requests, reuse_prefix=reuse_prefix)
    duration = time.perf_counter() - start

    total_tokens = count_tokens(outputs)
    throughput = total_tokens / duration if duration > 0 else float("inf")

    stats = get_offloading_stats(llm)
    delta = compute_delta(stats, prev_stats)

    result = RunResult(
        label="",
        num_requests=num_requests,
        duration=duration,
        tokens=total_tokens,
        throughput=throughput,
        delta=delta,
    )
    return result, stats


# ============================================================
# Printing / Reporting
# ============================================================

def print_section_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def print_table_header() -> None:
    """Print the table header for side-by-side comparison."""
    print("-" * 120)
    print(f"{'Run':<15} | {'GPU Offload':^45} | {'CPU Offload':^45} | {'Speedup':>8}")
    print(f"{'':<15} | {'time':>7} {'tok/s':>10} {'Δoffload':>9} {'Δreload':>8} | "
          f"{'time':>7} {'tok/s':>10} {'Δoffload':>9} {'Δreload':>8} | {'':<8}")
    print("-" * 120)


def compute_speedup(gpu_throughput: float, cpu_throughput: float) -> float:
    """Compute percentage speedup of GPU over CPU."""
    if cpu_throughput <= 0:
        return 0.0
    return ((gpu_throughput - cpu_throughput) / cpu_throughput) * 100


def print_comparison_row(label: str, gpu_result: RunResult, cpu_result: RunResult) -> None:
    """Print a single row comparing GPU and CPU results."""
    gpu_d = gpu_result.delta
    cpu_d = cpu_result.delta
    speedup = compute_speedup(gpu_result.throughput, cpu_result.throughput)
    speedup_str = f"{speedup:+.1f}%"
    
    print(
        f"{label:<15} | "
        f"{gpu_result.duration:>6.2f}s {gpu_result.throughput:>10.1f} {gpu_d.get('blocks_offloaded', 0):>9} {gpu_d.get('blocks_reloaded', 0):>8} | "
        f"{cpu_result.duration:>6.2f}s {cpu_result.throughput:>10.1f} {cpu_d.get('blocks_offloaded', 0):>9} {cpu_d.get('blocks_reloaded', 0):>8} | "
        f"{speedup_str:>8}"
    )


def print_cumulative_stats(
    gpu_stats: dict[str, int],
    cpu_stats: dict[str, int],
) -> None:
    """Print cumulative offloading statistics."""
    gpu_net = gpu_stats.get("blocks_offloaded", 0) - gpu_stats.get("blocks_reloaded", 0)
    cpu_net = cpu_stats.get("blocks_offloaded", 0) - cpu_stats.get("blocks_reloaded", 0)
    
    print()
    print(f"{'Metric':<25} | {'GPU Offload':>15} | {'CPU Offload':>15}")
    print("-" * 60)
    print(f"{'Total Blocks Offloaded':<25} | {gpu_stats.get('blocks_offloaded', 0):>15} | {cpu_stats.get('blocks_offloaded', 0):>15}")
    print(f"{'Total Blocks Reloaded':<25} | {gpu_stats.get('blocks_reloaded', 0):>15} | {cpu_stats.get('blocks_reloaded', 0):>15}")
    print(f"{'Total Blocks Evicted':<25} | {gpu_stats.get('blocks_evicted', 0):>15} | {cpu_stats.get('blocks_evicted', 0):>15}")
    print(f"{'Net Offloaded':<25} | {gpu_net:>15} | {cpu_net:>15}")


def print_final_summary(
    gpu_results: list[RunResult],
    cpu_results: list[RunResult],
    gpu_stats: dict[str, int],
    cpu_stats: dict[str, int],
) -> None:
    """Print final summary with overall speedup."""
    print_section_header("FINAL SUMMARY")
    
    # Calculate totals
    gpu_total_tokens = sum(r.tokens for r in gpu_results)
    cpu_total_tokens = sum(r.tokens for r in cpu_results)
    gpu_total_time = sum(r.duration for r in gpu_results)
    cpu_total_time = sum(r.duration for r in cpu_results)
    
    gpu_avg_throughput = gpu_total_tokens / gpu_total_time if gpu_total_time > 0 else 0
    cpu_avg_throughput = cpu_total_tokens / cpu_total_time if cpu_total_time > 0 else 0
    
    overall_speedup = compute_speedup(gpu_avg_throughput, cpu_avg_throughput)
    time_saved = cpu_total_time - gpu_total_time
    time_saved_pct = (time_saved / cpu_total_time * 100) if cpu_total_time > 0 else 0
    
    print(f"{'Metric':<30} | {'GPU Offload':>15} | {'CPU Offload':>15}")
    print("-" * 65)
    print(f"{'Total Tokens Generated':<30} | {gpu_total_tokens:>15,} | {cpu_total_tokens:>15,}")
    print(f"{'Total Time (s)':<30} | {gpu_total_time:>15.2f} | {cpu_total_time:>15.2f}")
    print(f"{'Average Throughput (tok/s)':<30} | {gpu_avg_throughput:>15.1f} | {cpu_avg_throughput:>15.1f}")
    
    # Print cumulative stats
    print_cumulative_stats(gpu_stats, cpu_stats)
    
    print()
    print("=" * 65)
    print("VERDICT")
    print("=" * 65)
    
    if overall_speedup > 0:
        print(f"🚀 GPU Offloading is {overall_speedup:.1f}% FASTER than CPU Offloading")
    elif overall_speedup < 0:
        print(f"🐢 GPU Offloading is {abs(overall_speedup):.1f}% SLOWER than CPU Offloading")
    else:
        print("⚖️  GPU and CPU Offloading have similar performance")
    
    if time_saved > 0:
        print(f"⏱️  Time saved with GPU offload: {time_saved:.2f}s ({time_saved_pct:.1f}% faster)")
    elif time_saved < 0:
        print(f"⏱️  Extra time with GPU offload: {abs(time_saved):.2f}s ({abs(time_saved_pct):.1f}% slower)")


# ============================================================
# Comparison Runner
# ============================================================

def run_gpu_benchmark(gpu_llm: LLM, request_counts: list[int]) -> tuple[list[RunResult], list[RunResult], dict[str, int]]:
    """Run all GPU offloading benchmarks and return results."""
    gpu_prev_stats: dict[str, int] = {key: 0 for key in STAT_KEYS}
    unique_results: list[RunResult] = []
    reload_results: list[RunResult] = []

    # UNIQUE PROMPTS
    for num_requests in request_counts:
        result, gpu_prev_stats = execute_single_run(
            gpu_llm, num_requests, reuse_prefix=False, prev_stats=gpu_prev_stats
        )
        result.label = f"UNIQUE {num_requests}"
        unique_results.append(result)

    # RELOAD PROMPTS
    for num_requests in request_counts:
        result, gpu_prev_stats = execute_single_run(
            gpu_llm, num_requests, reuse_prefix=True, prev_stats=gpu_prev_stats
        )
        result.label = f"RELOAD {num_requests}"
        reload_results.append(result)

    return unique_results, reload_results, gpu_prev_stats


def run_cpu_benchmark(cpu_llm: LLM, request_counts: list[int]) -> tuple[list[RunResult], list[RunResult], dict[str, int]]:
    """Run all CPU offloading benchmarks and return results."""
    cpu_prev_stats: dict[str, int] = {key: 0 for key in STAT_KEYS}
    unique_results: list[RunResult] = []
    reload_results: list[RunResult] = []

    # UNIQUE PROMPTS
    for num_requests in request_counts:
        result, cpu_prev_stats = execute_single_run(
            cpu_llm, num_requests, reuse_prefix=False, prev_stats=cpu_prev_stats
        )
        result.label = f"UNIQUE {num_requests}"
        unique_results.append(result)

    # RELOAD PROMPTS
    for num_requests in request_counts:
        result, cpu_prev_stats = execute_single_run(
            cpu_llm, num_requests, reuse_prefix=True, prev_stats=cpu_prev_stats
        )
        result.label = f"RELOAD {num_requests}"
        reload_results.append(result)

    return unique_results, reload_results, cpu_prev_stats


def print_results_table(
    gpu_unique: list[RunResult],
    gpu_reload: list[RunResult],
    cpu_unique: list[RunResult],
    cpu_reload: list[RunResult],
) -> None:
    """Print comparison tables for all results."""
    
    # ==================== UNIQUE PROMPTS ====================
    print_section_header("UNIQUE PROMPTS (no prefix cache reuse)")
    print_table_header()
    for gpu_r, cpu_r in zip(gpu_unique, cpu_unique):
        print_comparison_row(gpu_r.label, gpu_r, cpu_r)
    print("-" * 120)

    # ==================== RELOAD PROMPTS ====================
    print_section_header("RELOAD PROMPTS (reusing prefix cache)")
    print_table_header()
    for gpu_r, cpu_r in zip(gpu_reload, cpu_reload):
        print_comparison_row(gpu_r.label, gpu_r, cpu_r)
    print("-" * 120)


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("╔" + "═" * 98 + "╗")
    print("║" + " GPU vs CPU KV Cache Offloading Benchmark ".center(98) + "║")
    print("╚" + "═" * 98 + "╝")
    print()
    
    # Print hardware and config info
    print_hardware_info()
    print_test_config()
    
    request_counts = [10, 50, 100, 500, 1000, 2000]
    
    # ==================== GPU OFFLOADING ====================
    print("🔧 Initializing GPU Offloading LLM...")
    gpu_llm = create_gpu_offloading_llm()
    
    print("🏃 Running GPU offloading benchmarks...")
    gpu_unique, gpu_reload, gpu_stats = run_gpu_benchmark(gpu_llm, request_counts)
    
    print("🧹 Shutting down GPU Offloading LLM...")
    shutdown_llm(gpu_llm)
    
    # ==================== CPU OFFLOADING ====================
    print("\n🔧 Initializing CPU Offloading LLM...")
    cpu_llm = create_cpu_offloading_llm()
    
    print("🏃 Running CPU offloading benchmarks...")
    cpu_unique, cpu_reload, cpu_stats = run_cpu_benchmark(cpu_llm, request_counts)
    
    print("🧹 Shutting down CPU Offloading LLM...")
    shutdown_llm(cpu_llm)
    
    # ==================== PRINT RESULTS ====================
    print_results_table(gpu_unique, gpu_reload, cpu_unique, cpu_reload)
    
    # Combine results for final summary
    gpu_all = gpu_unique + gpu_reload
    cpu_all = cpu_unique + cpu_reload
    print_final_summary(gpu_all, cpu_all, gpu_stats, cpu_stats)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
