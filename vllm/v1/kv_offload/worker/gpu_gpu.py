# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention import AttentionBackend
from vllm.logger import init_logger
from vllm.v1.kv_offload.mediums import DestGPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

# Global variable to store handler instance
_global_handler_instance = None

logger = init_logger(__name__)


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


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
        # Store this instance globally
        global _global_handler_instance
        _global_handler_instance = self
        
        # Initialize counters
        self.blocks_offloaded: int = 0
        self.blocks_reloaded: int = 0
        self.blocks_evicted: int = 0
        self.blocks_freed: int = 0
        self.blocks_allocated: int = 0

        assert dest_block_size % src_block_size == 0
        self.block_size_factor = dest_block_size // src_block_size

        self.src_gpu_id = src_gpu_id
        self.dest_gpu_id = dest_gpu_id

        # cuda streams for gpu_src->gpu_dest and gpu_dest->gpu_src
        with torch.cuda.device(src_gpu_id):
            self.gpu_src_to_dest_stream = torch.cuda.Stream()
        with torch.cuda.device(dest_gpu_id):
            self.gpu_dest_to_src_stream = torch.cuda.Stream()

        # job_id -> (event, device_id)
        self.transfer_events: dict[int, tuple[torch.cuda.Event, int]] = {}
        # list of cuda events available for re-use, keyed by device
        self.events_pool: dict[int, list[torch.cuda.Event]] = {
            src_gpu_id: [],
            dest_gpu_id: [],
        }

        # allocate dest gpu tensors
        logger.info("Allocating %d secondary GPU tensors on cuda:%d...", 
                   len(gpu_caches), dest_gpu_id)
        
        self.src_gpu_tensors: list[torch.Tensor] = []
        self.dest_gpu_tensors: list[torch.Tensor] = []
        self.kv_dim_before_num_blocks: list[bool] = []
        
        for layer_name, gpu_tensor in gpu_caches.items():
            self.src_gpu_tensors.append(gpu_tensor)

            src_gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            if len(src_gpu_shape) != len(test_shape):
                # cross-layers tensor
                # shape is (num_blocks, ...)
                assert len(src_gpu_shape) == len(test_shape) + 1
                num_blocks_idx = 0
                self.kv_dim_before_num_blocks.append(False)
            elif test_shape[0] == 1234:
                # shape is (num_blocks, ...)
                num_blocks_idx = 0
                self.kv_dim_before_num_blocks.append(False)
            else:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert src_gpu_shape[0] == 2

                num_blocks_idx = 1
                self.kv_dim_before_num_blocks.append(True)

            dest_gpu_shape = list(src_gpu_shape)
            dest_gpu_shape[num_blocks_idx] = num_gpu_blocks * self.block_size_factor

            logger.debug("Allocating secondary GPU tensor of shape %r on cuda:%d", 
                        dest_gpu_shape, dest_gpu_id)

            self.dest_gpu_tensors.append(
                torch.zeros(
                    dest_gpu_shape,
                    dtype=gpu_tensor.dtype,
                    device=f"cuda:{dest_gpu_id}",
                )
            )

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_spec, dst_spec = spec
        
        # Determine direction and update counters
        if isinstance(src_spec, GPULoadStoreSpec):
            assert isinstance(dst_spec, DestGPULoadStoreSpec)
            direction = "OFFLOAD"
            stream = self.gpu_src_to_dest_stream
            event_device = self.src_gpu_id  # event on source device
            src_tensors = self.src_gpu_tensors
            dst_tensors = self.dest_gpu_tensors
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor
            # Update offload counter
            self.blocks_offloaded += len(src_spec.block_ids)
        else:
            assert isinstance(src_spec, DestGPULoadStoreSpec)
            assert isinstance(dst_spec, GPULoadStoreSpec)
            direction = "RELOAD"
            stream = self.gpu_dest_to_src_stream
            event_device = self.dest_gpu_id  # event on source device (dest GPU)
            src_tensors = self.dest_gpu_tensors
            dst_tensors = self.src_gpu_tensors
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
            # Update reload counter
            self.blocks_reloaded += len(src_spec.block_ids)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids

        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        if direction == "OFFLOAD":
            print(f"[KV TRANSFER] {direction} GPU{self.src_gpu_id}→GPU{self.dest_gpu_id} | "
                  f"Job {job_id} | Blocks: {len(src_blocks)} | "
                  f"IDs: {src_blocks[:5].tolist()}{'...' if len(src_blocks) > 5 else ''}")
        else:
            print(f"[KV TRANSFER] {direction} GPU{self.dest_gpu_id}→GPU{self.src_gpu_id} | "
                  f"Job {job_id} | Blocks: {len(src_blocks)} | "
                  f"IDs: {src_blocks[:5].tolist()}{'...' if len(src_blocks) > 5 else ''}")

        src_sub_block_count = src_blocks.size * src_block_size_factor
        dst_sub_block_count = dst_blocks.size * dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % src_block_size_factor

        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

        src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(
            src_blocks,
            src_block_size_factor,
            src_to_dst[:, 0],
            skip_count=src_sub_blocks_to_skip,
        )
        expand_block_ids(dst_blocks, dst_block_size_factor, src_to_dst[:, 1])
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        # Get or create event on the correct device
        pool = self.events_pool[event_device]
        if pool:
            event = pool.pop()
        else:
            with torch.cuda.device(event_device):
                event = torch.cuda.Event()
        
        with torch.cuda.stream(stream):
            for src_tensor, dst_tensor, kv_dim in zip(
                src_tensors, dst_tensors, self.kv_dim_before_num_blocks
            ):
                if kv_dim:
                    for kv_idx in range(2):  # 0=key, 1=value
                        src_cache = src_tensor[kv_idx]
                        dst_cache = dst_tensor[kv_idx]

                        for src_block_id, dst_block_id in src_to_dst:
                            dst_cache[dst_block_id].copy_(src_cache[src_block_id], non_blocking=True)
                else:
                    for src_block_id, dst_block_id in src_to_dst:
                        dst_tensor[dst_block_id].copy_(src_tensor[src_block_id], non_blocking=True)
            event.record(stream)

        self.transfer_events[job_id] = (event, event_device)

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id, (event, device) in self.transfer_events.items():
            if event.query():
                results.append((job_id, True))
                self.events_pool[device].append(event)
        for job_id, _ in results:
            del self.transfer_events[job_id]
        return results

    def get_offloading_stats(self) -> dict[str, int]:
        return {
            "blocks_offloaded": self.blocks_offloaded,
            "blocks_reloaded": self.blocks_reloaded,
            "blocks_allocated": self.blocks_allocated,
            "blocks_evicted": self.blocks_evicted,
            "blocks_freed": self.blocks_freed,
        }


# Module-level function to access stats
def get_global_stats() -> dict | None:
    """Get stats from the global handler instance."""
    global _global_handler_instance
    if _global_handler_instance:
        return _global_handler_instance.get_offloading_stats()
    return None