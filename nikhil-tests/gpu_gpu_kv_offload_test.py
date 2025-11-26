import time
from dataclasses import dataclass
import gc
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
import torch

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
        "gpu_memory_utilization": 0.9,
    },
    "opt": {
        "name": "facebook/opt-125m",
        "gpu_memory_utilization": 0.03,
    },
}

MODEL = "opt"
MODEL_NAME = MODELS[MODEL]["name"]
GPU_MEMORY_UTILIZATION = MODELS[MODEL]["gpu_memory_utilization"]
NUM_BLOCKS = 500
DEST_GPU_ID = 1
BLOCK_SIZE = 16
EVICTION_POLICY = "lru"

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
    delta: dict[str, int]


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
            "spec_name": "CPUOffloadingSpec",
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
    # Ask the underlying executor to shut down (what LLMEngine.__del__ does)
    engine = getattr(llm, "llm_engine", None)
    if engine is not None:
        executor = getattr(engine, "model_executor", None)
        if executor is not None and hasattr(executor, "shutdown"):
            executor.shutdown()

    # Drop Python references and clean up CUDA memory
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# ============================================================
# Prompt Generation
# ============================================================

def generate_prompts(num_requests: int, reuse_prefix: bool = True) -> list[str]:
    """
    Generate prompts. When reuse_prefix=True, uses the same fixed prompts
    so that later runs can hit the prefix cache and trigger reloads.
    """
    assert num_requests > 0, "Number of requests must be greater than 0"

    if reuse_prefix:
        return [SHARED_PROMPTS[i % len(SHARED_PROMPTS)] for i in range(num_requests)]
    else:
        return [f"Unique request {i+1}: Tell me a story." for i in range(num_requests)]


# ============================================================
# Stats Collection
# ============================================================

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
    verbose: bool = False,
):
    """Run inference and return outputs."""
    prompts = generate_prompts(num_requests, reuse_prefix=reuse_prefix)
    if verbose:
        print(f"  Running {num_requests} request(s) | reuse_prefix={reuse_prefix}")
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
        label="",  # Will be set by caller
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

def print_header(title: str) -> None:
    """Print a section header."""
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_comparison_header() -> None:
    """Print the table header for side-by-side comparison."""
    print()
    print("=" * 100)
    print(f"{'Run':<20} | {'GPU Offload':<35} | {'CPU Offload':<35}")
    print(f"{'':<20} | {'time':>8} {'tok/s':>10} {'Δoff':>6} {'Δrel':>6} | "
          f"{'time':>8} {'tok/s':>10} {'Δoff':>6} {'Δrel':>6}")
    print("-" * 100)


def print_comparison_row(label: str, gpu_result: RunResult, cpu_result: RunResult) -> None:
    """Print a single row comparing GPU and CPU results."""
    gpu_d = gpu_result.delta
    cpu_d = cpu_result.delta
    print(
        f"{label:<20} | "
        f"{gpu_result.duration:>7.2f}s {gpu_result.throughput:>10.1f} "
        f"{gpu_d.get('blocks_offloaded', 0):>6} {gpu_d.get('blocks_reloaded', 0):>6} | "
        f"{cpu_result.duration:>7.2f}s {cpu_result.throughput:>10.1f} "
        f"{cpu_d.get('blocks_offloaded', 0):>6} {cpu_d.get('blocks_reloaded', 0):>6}"
    )


def print_cumulative_stats_row(label: str, stats: dict[str, int]) -> None:
    """Print cumulative stats for one LLM type."""
    net = stats.get("blocks_offloaded", 0) - stats.get("blocks_reloaded", 0)
    print(f"  {label}: offloaded={stats.get('blocks_offloaded', 0)}, "
          f"reloaded={stats.get('blocks_reloaded', 0)}, "
          f"net={net}")


def print_final_summary(
    gpu_stats: dict[str, int],
    cpu_stats: dict[str, int],
) -> None:
    """Print final cumulative statistics."""
    print()
    print("=" * 100)
    print("Cumulative Offloading Stats")
    print("-" * 100)
    print_cumulative_stats_row("GPU Offload", gpu_stats)
    print_cumulative_stats_row("CPU Offload", cpu_stats)
    print("=" * 100)


# ============================================================
# Comparison Runner
# ============================================================

def run_comparison(gpu_llm: LLM, cpu_llm: LLM) -> None:
    """Run side-by-side comparison of GPU and CPU offloading."""
    
    # Test plan: (label, num_requests, reuse_prefix)
    run_plan = [
        ("Warmup", 10, True),
        ("Cached 50", 50, True),
        ("Cached 100", 100, True),
        ("Unique 50", 50, False),
        ("Cached again", 100, True),
    ]

    gpu_prev_stats: dict[str, int] = {key: 0 for key in STAT_KEYS}
    cpu_prev_stats: dict[str, int] = {key: 0 for key in STAT_KEYS}

    print_comparison_header()

    for label, num_requests, reuse_prefix in run_plan:
        # Run GPU offloading
        gpu_result, gpu_prev_stats = execute_single_run(
            gpu_llm, num_requests, reuse_prefix, gpu_prev_stats
        )
        gpu_result.label = label

        # Run CPU offloading
        cpu_result, cpu_prev_stats = execute_single_run(
            cpu_llm, num_requests, reuse_prefix, cpu_prev_stats
        )
        cpu_result.label = label

        # Print comparison row immediately
        print_comparison_row(label, gpu_result, cpu_result)

    # Print final summary
    print_final_summary(gpu_prev_stats, cpu_prev_stats)


# ============================================================
# Main
# ============================================================

def main():
    print_header("GPU vs CPU KV Cache Offloading Comparison")

    print("\nInitializing GPU Offloading LLM...")
    gpu_llm = create_gpu_offloading_llm()

    print("\nInitializing CPU Offloading LLM...")
    cpu_llm = create_cpu_offloading_llm()

    run_comparison(gpu_llm, cpu_llm)

    shutdown_llm(gpu_llm)
    shutdown_llm(cpu_llm)

if __name__ == "__main__":
    main()
    
