# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

import torch

from vllm.attention import AttentionBackend
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.gpu import GPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import DestGPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.gpu_gpu import GpuGpuOffloadingHandler
from vllm.v1.kv_offload.worker.worker import OffloadingHandler



class GPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        num_gpu_blocks = self.extra_config.get("num_gpu_blocks")
        if not num_gpu_blocks:
            raise Exception(
                "num_gpu_blocks must be specified in kv_connector_extra_config"
            )
        self.num_gpu_blocks: int = num_gpu_blocks
        self.dest_gpu_id: int = self.extra_config.get("dest_gpu_id", 1)

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handler: OffloadingHandler | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            gpu_backend = GPUBackend(
                block_size=self.offloaded_block_size, num_blocks=self.num_gpu_blocks
            )

            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=gpu_backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=gpu_backend, enable_events=enable_events
                )
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    f"Supported policies: lru, arc"
                )
        return self._manager
    
    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handler:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "GPU Offloading is currently only supported on CUDA-alike GPUs"
                )

            self._handler = GpuGpuOffloadingHandler(
                src_block_size=self.gpu_block_size,
                dest_block_size=self.offloaded_block_size,
                attn_backends=attn_backends,
                gpu_caches=kv_caches,
                num_gpu_blocks=self.num_gpu_blocks,
                src_gpu_id=0,
                dest_gpu_id=self.dest_gpu_id,
            )

        assert self._handler is not None
        yield GPULoadStoreSpec, DestGPULoadStoreSpec, self._handler
        yield DestGPULoadStoreSpec, GPULoadStoreSpec, self._handler

    def get_handler_stats(self) -> dict | None:
        """Get statistics from the handler if it exists.
    
        Returns:
        A dictionary containing:
            - blocks_offloaded: Number of blocks moved to secondary GPU
            - blocks_reloaded: Number of blocks brought back from secondary GPU
            - blocks_allocated: Total blocks allocated
            - blocks_evicted: Total blocks evicted
            - blocks_freed: Total blocks freed
        Returns None if handler is not initialized.
    """
        if self._handler and hasattr(self._handler, 'get_offloading_stats'):
            return self._handler.get_offloading_stats()
        return None
