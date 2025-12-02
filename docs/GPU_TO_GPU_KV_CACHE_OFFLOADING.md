# GPU-to-GPU KV Cache Offloading in vLLM

## Overview

This document describes the GPU-to-GPU KV cache offloading feature added to vLLM. This feature extends the existing CPU offloading capability to allow offloading KV cache blocks from a primary GPU (running the model) to a secondary GPU's memory, enabling higher throughput and lower latency compared to CPU offloading due to faster GPU-to-GPU transfer speeds.

---

## Architecture Overview

The KV cache offloading system in vLLM v1 uses a modular architecture with clear separation between:

1. **Scheduler-side components** - Track which blocks are offloaded and manage eviction policies
2. **Worker-side components** - Handle the actual data transfers between devices

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Scheduler Side                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │ OffloadingSpec  │───▶│ OffloadingManager│───▶│     Backend        │  │
│  │ (GPUOffloading  │    │ (LRU/ARC)        │    │  (GPUBackend/      │  │
│  │     Spec)       │    │                  │    │   CPUBackend)      │  │
│  └─────────────────┘    └──────────────────┘    └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ TransferSpec
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Worker Side                                   │
│  ┌─────────────────┐    ┌──────────────────────┐                        │
│  │ OffloadingWorker│───▶│  OffloadingHandler   │                        │
│  │                 │    │ (GpuGpuOffloading    │                        │
│  │                 │    │     Handler)         │                        │
│  └─────────────────┘    └──────────────────────┘                        │
│                                    │                                     │
│                         ┌──────────┴──────────┐                         │
│                         ▼                     ▼                         │
│              ┌─────────────────┐    ┌─────────────────┐                │
│              │  Primary GPU    │    │ Secondary GPU   │                │
│              │  (cuda:0)       │◀──▶│  (cuda:1)       │                │
│              │  KV Cache       │    │  KV Cache       │                │
│              └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Load/Store Specifications (`vllm/v1/kv_offload/mediums.py`)

These classes define the metadata for KV block transfers:

| Class | Description | Medium |
|-------|-------------|--------|
| `GPULoadStoreSpec` | Spec for blocks on the primary GPU | `"GPU"` |
| `DestGPULoadStoreSpec` | Spec for blocks on the secondary/destination GPU | `"SECONDARY_GPU"` |
| `CPULoadStoreSpec` | Spec for blocks in CPU memory | `"CPU"` |

All specs inherit from `BlockIDsLoadStoreSpec` which wraps a numpy array of block IDs (`block_ids: np.ndarray`).

```python
class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "GPU"

class DestGPULoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "SECONDARY_GPU"
```

---

### 2. GPU Backend (`vllm/v1/kv_offload/backends/gpu.py`)

The `GPUBackend` class manages block allocation on the secondary GPU:

```python
class GPUBackend(Backend):
    def __init__(self, block_size: int, num_blocks: int):
        # Tracks total and allocated blocks on secondary GPU
        self.num_blocks: int = num_blocks
        self.num_allocated_blocks: int = 0
        self.allocated_blocks_free_list: list[int] = []

    def get_num_free_blocks(self) -> int:
        # Returns available blocks for allocation
        
    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        # Allocates blocks, reusing from free list when possible
        
    def free(self, block: BlockStatus):
        # Returns block to free list
```

**Key Data Structure:**
- `GPUBlockStatus` - Extends `BlockStatus` with a `block_id` field to track the block's position in the secondary GPU tensor

---

### 3. GPU Offloading Spec (`vllm/v1/kv_offload/gpu.py`)

The `GPUOffloadingSpec` is the main entry point for configuring GPU-to-GPU offloading:

```python
class GPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig):
        # Configuration from kv_connector_extra_config:
        self.num_gpu_blocks: int      # Number of blocks on secondary GPU
        self.dest_gpu_id: int         # Secondary GPU device ID (default: 1)
        self.eviction_policy: str     # "lru" or "arc"
        
    def get_manager(self) -> OffloadingManager:
        # Returns LRUOffloadingManager or ARCOffloadingManager
        # with GPUBackend
        
    def get_handlers(self, kv_caches, attn_backends):
        # Returns GpuGpuOffloadingHandler for bidirectional transfers
        yield GPULoadStoreSpec, DestGPULoadStoreSpec, handler  # Offload
        yield DestGPULoadStoreSpec, GPULoadStoreSpec, handler  # Reload
```

---

### 4. GPU-GPU Offloading Handler (`vllm/v1/kv_offload/worker/gpu_gpu.py`)

This is the core component that performs async KV transfers between GPUs:

```python
class GpuGpuOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        src_block_size: int,
        dest_block_size: int,
        num_gpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        src_gpu_id: int = 0,
        dest_gpu_id: int = 1,
    ):
        # Block size conversion factor
        self.block_size_factor = dest_block_size // src_block_size
        
        # CUDA streams for async transfers
        self.gpu_src_to_dest_stream = torch.cuda.Stream()  # Offload stream
        self.gpu_dest_to_src_stream = torch.cuda.Stream()  # Reload stream
        
        # Event tracking for async completion
        self.transfer_events: dict[int, tuple[torch.cuda.Event, int]] = {}
        self.events_pool: dict[int, list[torch.cuda.Event]] = {}
        
        # Allocate KV cache tensors on secondary GPU
        self.src_gpu_tensors: list[torch.Tensor]   # Primary GPU tensors
        self.dest_gpu_tensors: list[torch.Tensor]  # Secondary GPU tensors
        
        # Statistics counters
        self.blocks_offloaded: int = 0
        self.blocks_reloaded: int = 0
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `transfer_async(job_id, spec)` | Initiates async transfer, returns immediately |
| `get_finished()` | Returns list of completed transfers |
| `get_offloading_stats()` | Returns transfer statistics |

**Transfer Logic:**
```python
def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
    src_spec, dst_spec = spec
    
    # Determine direction
    if isinstance(src_spec, GPULoadStoreSpec):
        direction = "OFFLOAD"  # GPU → Secondary GPU
        stream = self.gpu_src_to_dest_stream
    else:
        direction = "RELOAD"   # Secondary GPU → GPU
        stream = self.gpu_dest_to_src_stream
    
    # Perform async copy using non-blocking operations
    with torch.cuda.stream(stream):
        for src_tensor, dst_tensor in zip(src_tensors, dst_tensors):
            for src_block_id, dst_block_id in src_to_dst:
                dst_tensor[dst_block_id].copy_(
                    src_tensor[src_block_id], 
                    non_blocking=True
                )
        event.record(stream)
    
    self.transfer_events[job_id] = (event, device)
```

---

### 5. Offloading Worker (`vllm/v1/kv_offload/worker/worker.py`)

Coordinates multiple handlers and routes transfers based on type:

```python
class OffloadingWorker:
    def __init__(self):
        self.handlers: set[OffloadingHandler] = set()
        self.transfer_type_to_handler: dict[TransferType, OffloadingHandler] = {}
    
    def register_handler(self, src_cls, dst_cls, handler):
        # Maps (src_medium, dst_medium) → handler
        transfer_type = (src_cls.medium(), dst_cls.medium())
        self.transfer_type_to_handler[transfer_type] = handler
    
    def transfer_async(self, job_id, spec):
        # Routes to appropriate handler based on transfer type
        src, dst = spec
        transfer_type = (src.medium(), dst.medium())
        handler = self.transfer_type_to_handler[transfer_type]
        return handler.transfer_async(job_id, spec)
```

---

### 6. Offloading Manager (`vllm/v1/kv_offload/lru_manager.py`)

Scheduler-side component that tracks offloaded blocks and manages eviction:

```python
class LRUOffloadingManager(OffloadingManager):
    def __init__(self, backend: Backend):
        self.backend = backend
        # OrderedDict for LRU ordering
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
    
    def lookup(self, block_hashes) -> int:
        # Count consecutive offloaded blocks from start
        
    def prepare_load(self, block_hashes) -> LoadStoreSpec:
        # Increment ref counts, return load spec
        
    def touch(self, block_hashes):
        # Move blocks to end (most recently used)
        
    def prepare_store(self, block_hashes) -> PrepareStoreOutput:
        # Evict blocks if needed, allocate new, return store spec
```

---

### 7. Factory (`vllm/v1/kv_offload/factory.py`)

Registers and creates offloading specs:

```python
OffloadingSpecFactory.register_spec(
    "CPUOffloadingSpec", "vllm.v1.kv_offload.cpu", "CPUOffloadingSpec",
)
OffloadingSpecFactory.register_spec(
    "GPUOffloadingSpec", "vllm.v1.kv_offload.gpu", "GPUOffloadingSpec"
)
```

---

## Usage

### Configuration

To use GPU-to-GPU offloading, configure the `KVTransferConfig`:

```python
from vllm import LLM
from vllm.config.kv_transfer import KVTransferConfig

config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "spec_name": "GPUOffloadingSpec",
        "num_gpu_blocks": 10000,       # Blocks on secondary GPU
        "dest_gpu_id": 1,              # Secondary GPU device ID
        "block_size": 16,              # Tokens per block
        "eviction_policy": "lru",      # or "arc"
    },
)

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    kv_transfer_config=config,
    gpu_memory_utilization=0.2,
)
```

### Comparison with CPU Offloading

For CPU offloading, use `CPUOffloadingSpec`:

```python
config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "spec_name": "CPUOffloadingSpec",
        "num_cpu_blocks": 10000,
        "block_size": 16,
        "eviction_policy": "lru",
    },
)
```

---

## Key Differences: GPU vs CPU Offloading

| Aspect | GPU Offloading | CPU Offloading |
|--------|----------------|----------------|
| **Handler** | `GpuGpuOffloadingHandler` | `CpuGpuOffloadingHandler` |
| **Backend** | `GPUBackend` | `CPUBackend` |
| **Spec Classes** | `GPULoadStoreSpec` ↔ `DestGPULoadStoreSpec` | `GPULoadStoreSpec` ↔ `CPULoadStoreSpec` |
| **Storage** | Secondary GPU VRAM | Pinned CPU memory |
| **Streams** | `gpu_src_to_dest_stream`, `gpu_dest_to_src_stream` | `d2h_stream`, `h2d_stream` |
| **Transfer Method** | `tensor.copy_(non_blocking=True)` | `ops.swap_blocks()` |
| **Expected Speed** | Faster (GPU↔GPU bandwidth) | Slower (PCIe bandwidth) |

---

## Data Structures Summary

### Transfer Specifications

| Type | Description |
|------|-------------|
| `TransferSpec` | `tuple[LoadStoreSpec, LoadStoreSpec]` - (src, dst) specs |
| `TransferType` | `tuple[str, str]` - (src_medium, dst_medium) |
| `TransferResult` | `tuple[int, bool]` - (job_id, success) |

### Block Management

| Type | Description |
|------|-------------|
| `BlockHash` | Unique identifier for KV block content |
| `BlockStatus` | Tracks ref_cnt and readiness |
| `GPUBlockStatus` | Extends BlockStatus with block_id |

### Output Types

| Type | Description |
|------|-------------|
| `PrepareStoreOutput` | Contains hashes to store, store spec, evicted hashes |
| `OffloadingEvent` | Event log for block storage/removal |

---

## Integration Points

### 1. GPU Worker (`vllm/v1/worker/gpu_worker.py`)

The GPU worker initializes KV transfer during `initialize_from_config()`:

```python
def initialize_from_config(self, kv_cache_config: KVCacheConfig):
    ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
    self.model_runner.initialize_kv_cache(kv_cache_config)
```

### 2. Model Runner Mixin (`vllm/v1/worker/kv_connector_model_runner_mixin.py`)

Provides helper methods for KV connector operations:

```python
class KVConnectorModelRunnerMixin:
    @staticmethod
    def get_kv_offloading_stats() -> dict | None:
        """Return offloading stats exposed by the KV connector."""
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            if hasattr(kv_connector, "get_offloading_stats"):
                return kv_connector.get_offloading_stats()
        return None
```

---

## Statistics and Monitoring

The handler tracks the following metrics:

| Metric | Description |
|--------|-------------|
| `blocks_offloaded` | Total blocks transferred to secondary GPU |
| `blocks_reloaded` | Total blocks transferred back to primary GPU |
| `blocks_allocated` | Total blocks allocated on secondary GPU |
| `blocks_evicted` | Total blocks evicted from secondary GPU |
| `blocks_freed` | Total blocks freed on secondary GPU |

Access via:
```python
stats = llm.llm_engine.collective_rpc("get_offloading_stats")
```

---

## File Summary

| File | Purpose |
|------|---------|
| `vllm/v1/kv_offload/gpu.py` | `GPUOffloadingSpec` - Main GPU offloading configuration |
| `vllm/v1/kv_offload/cpu.py` | `CPUOffloadingSpec` - CPU offloading (for comparison) |
| `vllm/v1/kv_offload/worker/gpu_gpu.py` | `GpuGpuOffloadingHandler` - GPU↔GPU transfer logic |
| `vllm/v1/kv_offload/worker/cpu_gpu.py` | `CpuGpuOffloadingHandler` - CPU↔GPU transfer logic |
| `vllm/v1/kv_offload/backends/gpu.py` | `GPUBackend` - Block allocation on secondary GPU |
| `vllm/v1/kv_offload/mediums.py` | LoadStoreSpec classes for different mediums |
| `vllm/v1/kv_offload/factory.py` | Spec registration and creation |
| `vllm/v1/kv_offload/spec.py` | Base `OffloadingSpec` class |
| `vllm/v1/kv_offload/abstract.py` | Abstract classes for managers and specs |
| `vllm/v1/kv_offload/lru_manager.py` | LRU eviction policy implementation |
| `vllm/v1/kv_offload/worker/worker.py` | `OffloadingWorker` and `OffloadingHandler` base classes |
| `nikhil-tests/gpu_gpu_kv_offload_test.py` | Benchmark comparing GPU vs CPU offloading |

---

## Extending the Feature

To add a new offloading backend:

1. **Create a new LoadStoreSpec** in `mediums.py`:
   ```python
   class NewMediumLoadStoreSpec(BlockIDsLoadStoreSpec):
       @staticmethod
       def medium() -> str:
           return "NEW_MEDIUM"
   ```

2. **Create a Backend** in `backends/`:
   ```python
   class NewBackend(Backend):
       def allocate_blocks(self, block_hashes): ...
       def free(self, block): ...
       def get_load_store_spec(self, block_hashes, blocks): ...
   ```

3. **Create a Handler** in `worker/`:
   ```python
   class NewOffloadingHandler(OffloadingHandler):
       def transfer_async(self, job_id, spec): ...
       def get_finished(self): ...
   ```

4. **Create an OffloadingSpec** and register it:
   ```python
   OffloadingSpecFactory.register_spec(
       "NewOffloadingSpec", "vllm.v1.kv_offload.new", "NewOffloadingSpec"
   )
   ```

---

## Performance Considerations

- **GPU-to-GPU transfers** use NVLink or PCIe peer-to-peer when available, providing higher bandwidth than CPU transfers
- **Async transfers** allow overlap with model execution
- **Block size factor** allows coalescing multiple small blocks into larger transfers
- **LRU eviction** ensures frequently used blocks stay in cache
- **Event pooling** reduces CUDA event allocation overhead

---

## Testing

The benchmark script `nikhil-tests/gpu_gpu_kv_offload_test.py` provides:
- Side-by-side comparison of GPU vs CPU offloading
- Throughput measurements (tokens/second)
- Block transfer statistics
- Speedup calculations

Run with:
```bash
python nikhil-tests/gpu_gpu_kv_offload_test.py
```
