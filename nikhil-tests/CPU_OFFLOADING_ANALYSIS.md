# CPU KV Cache Offloading: Architecture Analysis & Limitations

## Executive Summary

CPU KV cache offloading in vLLM V1 is designed for **prefix caching** (storing completed KV blocks for future request reuse), NOT for **extending active context length** by swapping KV cache mid-request. This fundamental design choice creates limitations when trying to benchmark CPU vs GPU offloading with constrained GPU memory.

---

## How KV Cache Works in Transformers

During inference, each transformer layer computes:
```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

**Critical insight**: Every decode step requires access to ALL previous keys and values for that sequence. This is the "KV cache" - it grows linearly with sequence length.

```
Token 1:    K₁, V₁
Token 2:    K₁, V₁, K₂, V₂  
Token 3:    K₁, V₁, K₂, V₂, K₃, V₃
...
Token N:    K₁, V₁, K₂, V₂, ..., Kₙ, Vₙ  ← Need ALL of these on GPU
```

---

## vLLM's Offloading Architecture

### Design Intent: Prefix Caching
```
Request A finishes → KV blocks copied to CPU → GPU blocks freed
                                    ↓
Later, Request B with same prefix → Reload from CPU → Skip recomputation
```

### What Happens During Offloading
```
1. Request is RUNNING on GPU
2. Block becomes "full" (all tokens computed)
3. Block is COPIED to CPU (data now exists in both places)
4. GPU block remains allocated to the request
5. Request continues generating tokens
6. When request FINISHES, GPU blocks are freed
```

**Key point**: Step 4 - the GPU block stays allocated. The copy to CPU is for future prefix cache hits, not to free GPU memory immediately.

---

## The Hang at 81% - Root Cause Analysis

### Setup
- GPU memory utilization: 0.15 (very constrained)
- GPU KV cache: ~130 blocks (~2,080 tokens)
- 300 concurrent prompts generating tokens

### What Happened
```
Time T₀: Requests start, GPU blocks allocated
Time T₁: GPU KV cache fills up (130 blocks used)
Time T₂: Blocks are offloaded to CPU, but GPU blocks NOT freed
Time T₃: New blocks needed, but none available
Time T₄: Scheduler waits for free blocks... forever (HANG)
```

### Why Blocks Aren't Freed

The offloading connector only marks requests as `finished_sending` when:
1. The request has finished generating ALL tokens, AND
2. ALL store jobs for that request have completed

```python
# offloading_connector.py
if req_id in self._finished_reqs_waiting_for_store:
    self._finished_reqs_waiting_for_store.remove(req_id)
    finished_sending.add(req_id)  # Only then!
```

Until `finished_sending` is received, GPU blocks remain allocated.

---

## Why Incremental Freeing Fails

### Attempted Fix
Free GPU blocks immediately after they're copied to CPU.

### Why It Crashes
```
1. Block X allocated to Request A, tracked in req_to_blocks[A]
2. Block X offloaded to CPU
3. Block X freed from GPU (our attempted fix)
4. Block X reallocated to Request B
5. Request A continues, tries to use Block X for attention
6. Block X now has Request B's data → CORRUPTION
   OR
   Request A tries to cache Block X again → AssertionError
```

The request still has a reference to the block in its internal tracking (`req_to_blocks`). Simply freeing the GPU block without updating this tracking corrupts state.

### Proper Fix (Not Implemented)
Would require:
1. Remove block from request's `req_to_blocks`
2. Track that tokens [X, X+block_size] are now "virtual" (CPU only)
3. Before each forward pass, reload needed blocks from CPU
4. This is essentially "paging" - a major architectural change

---

## Can We Set GPU KV Cache to Zero?

**No.** Here's why:

### Attention Requires GPU Memory
```python
# Simplified attention computation
for layer in transformer_layers:
    Q = compute_query(hidden_states)      # On GPU
    K = kv_cache[layer].keys              # MUST be on GPU for matmul
    V = kv_cache[layer].values            # MUST be on GPU for matmul
    attn_output = softmax(Q @ K.T) @ V    # GPU operation
```

The `Q @ K.T` matrix multiplication requires both Q and K to be on the same device (GPU). You cannot perform attention with KV cache on CPU.

### Minimum GPU KV Cache Required
At minimum, you need enough GPU blocks to hold:
- Current batch's tokens being processed
- Enough context for meaningful attention

With zero GPU KV cache, the model literally cannot compute attention.

### What Would Happen
```
gpu_memory_utilization=0.0 (or num_gpu_blocks=0)
    ↓
No KV cache allocated on GPU
    ↓
First token generation fails - nowhere to store K, V
    ↓
Crash or immediate failure
```

---

## Workarounds for Benchmarking

### Option 1: Increase GPU Memory Utilization
```python
llm_cpu = LLM(
    model=model,
    kv_transfer_config=kv_transfer_config,
    gpu_memory_utilization=0.5,  # More GPU KV cache
    ...
)
```
**Pros**: Avoids hang, offloading still happens for overflow
**Cons**: Not a fair comparison if GPU test uses different value

### Option 2: Reduce Concurrent Requests
```python
llm_cpu = LLM(
    ...
    max_num_seqs=8,  # Fewer simultaneous requests
)
```
**Pros**: Less GPU memory pressure
**Cons**: Lower throughput

### Option 3: Use Shorter Sequences
Reduce `MAX_TOKENS` or use shorter prompts.

### Option 4: Match GPU Offloading Configuration
For GPU→GPU offloading, if it works with `gpu_memory_utilization=0.15`, use the same for CPU. If GPU offloading also hangs at 81%, the issue is workload-related, not CPU-specific.

---

## Comparison: GPU vs CPU Offloading

| Aspect | GPU Offloading | CPU Offloading |
|--------|---------------|----------------|
| Transfer speed | ~600 GB/s (NVLink) | ~25 GB/s (PCIe) |
| Latency | ~1-5 μs | ~10-50 μs |
| Capacity | Limited by 2nd GPU | Limited by system RAM |
| Use case | High-perf overflow | Large capacity overflow |

Both use the same offloading connector architecture and have the same limitation: blocks aren't freed until requests complete.

---

## Future Work

To properly support "KV cache paging" (free GPU blocks mid-request):

1. **Track virtual blocks**: Know which blocks are CPU-only
2. **Reload on demand**: Before attention, reload needed blocks
3. **Block table indirection**: Map virtual→physical blocks dynamically
4. **Prefetching**: Predict and preload blocks before needed

This would be similar to OS virtual memory paging, but for KV cache.

---

## Conclusion

The current CPU offloading hang is not a bug but a **design limitation**. The offloading system assumes requests complete before their blocks need to be reused. With severely constrained GPU memory, this assumption breaks down.

For benchmarking, use configurations where the GPU KV cache can hold at least one full request's worth of blocks, with some headroom for concurrent requests.
