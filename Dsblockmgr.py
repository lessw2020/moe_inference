"""Module for managing GPU/CPU memory blocks for key-value cache."""
from typing import List, Callable, Dict, Set, Optional
from enum import Enum, auto
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod

from distserve.config import ModelConfig, ParallelConfig, CacheConfig
from distserve.request import Request, BatchedRequests
from distserve.utils import Stage

logger = logging.getLogger(__name__)


class BlockLocation(Enum):
    """Location of memory blocks."""
    GPU = auto()
    CPU = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class BlockPool:
    """Manages a pool of memory blocks for either GPU or CPU."""
    
    location: BlockLocation
    max_blocks: int
    free_blocks: List[int] = field(default_factory=list)
    swapping_blocks: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        self.free_blocks = list(range(self.max_blocks))
    
    @property
    def available_blocks(self) -> int:
        """Total number of available blocks including those being swapped."""
        return len(self.free_blocks) + len(self.swapping_blocks)
        
    def get_free_blocks(self, num_blocks: int) -> List[int]:
        """Get specified number of free blocks, flushing swap buffer if needed."""
        if len(self.free_blocks) < num_blocks:
            if self.available_blocks < num_blocks:
                raise ValueError(
                    f"Not enough blocks on {self.location}: "
                    f"requested {num_blocks}, available {self.available_blocks}")
            return None  # Indicates need to flush swap buffer
        
        blocks = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        return blocks

    def add_free_blocks(self, blocks: List[int]):
        """Return blocks to free pool."""
        self.free_blocks.extend(blocks)
        
    def add_swapping_blocks(self, blocks: List[int]):
        """Add blocks to swapping pool."""
        self.swapping_blocks.extend(blocks)
        
    def flush_swapping_blocks(self):
        """Move all swapping blocks to free pool."""
        self.free_blocks.extend(self.swapping_blocks)
        self.swapping_blocks = []
        

@dataclass
class BlockAllocation:
    """Tracks block allocation for a single request."""
    location: BlockLocation
    block_ids: List[int]


class BlockManager:
    """Manages key-value cache memory blocks across GPU and CPU.
    
    Handles allocation, deallocation, and swapping of memory blocks between
    GPU and CPU for efficient cache management.
    """

    def __init__(
        self,
        stage: Stage,
        max_num_gpu_blocks: int,
        max_num_cpu_blocks: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        engine_remote_call_all_workers_async: Callable,
    ):
        """Initialize block manager.
        
        Args:
            stage: Processing stage (context or decoding)
            max_num_gpu_blocks: Maximum GPU blocks available
            max_num_cpu_blocks: Maximum CPU blocks available
            model_config: Model configuration
            parallel_config: Parallel processing configuration
            cache_config: Cache configuration
            engine_remote_call_all_workers_async: Callback for async worker calls
        """
        self.stage = stage
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.engine_remote_call_all_workers_async = engine_remote_call_all_workers_async

        # Initialize block pools
        self.gpu_pool = BlockPool(BlockLocation.GPU, max_num_gpu_blocks)
        self.cpu_pool = BlockPool(BlockLocation.CPU, max_num_cpu_blocks)
        
        # Track block allocations by request
        self.allocations: Dict[int, BlockAllocation] = {}

    def _get_pool(self, location: BlockLocation) -> BlockPool:
        """Get block pool for specified location."""
        return self.gpu_pool if location == BlockLocation.GPU else self.cpu_pool

    def _calculate_blocks_needed(self, request: Request) -> int:
        """Calculate number of blocks needed for a request."""
        total_tokens = request.get_input_len() + request.get_output_len()
        return (total_tokens + self.cache_config.block_size - 1) // self.cache_config.block_size

    def _ensure_blocks_available(self, pool: BlockPool, num_blocks: int):
        """Ensure required blocks are available in pool."""
        blocks = pool.get_free_blocks(num_blocks)
        if blocks is None:
            # Need to flush swapping blocks
            if pool.location == BlockLocation.GPU:
                self.engine_remote_call_all_workers_async("wait_for_all_swap_out")
            else:
                self.engine_remote_call_all_workers_async("wait_for_all_swap_in")
            pool.flush_swapping_blocks()
            blocks = pool.get_free_blocks(num_blocks)
        return blocks

    def allocate_blocks(self, request: Request):
        """Allocate blocks for a request."""
        request_id = request.request_id
        allocation = self.allocations.get(request_id)
        
        # Verify request state
        if allocation and allocation.location == BlockLocation.CPU:
            raise ValueError(
                f"Request {request_id} blocks are on CPU. "
                "Migrate to GPU before allocating more blocks.")

        num_blocks_needed = self._calculate_blocks_needed(request)
        
        if not allocation:
            # New allocation
            blocks = self._ensure_blocks_available(self.gpu_pool, num_blocks_needed)
            self.allocations[request_id] = BlockAllocation(BlockLocation.GPU, blocks)
        else:
            # Extend existing allocation if needed
            current_blocks = len(allocation.block_ids)
            if current_blocks < num_blocks_needed:
                additional_blocks = self._ensure_blocks_available(
                    self.gpu_pool, num_blocks_needed - current_blocks)
                allocation.block_ids.extend(additional_blocks)

    def allocate_blocks_batched(self, batch_requests: BatchedRequests):
        """Allocate blocks for multiple requests."""
        for request in batch_requests.requests:
            self.allocate_blocks(request)

    def free_blocks(self, request_id: int):
        """Free blocks allocated to a request."""
        allocation = self.allocations.get(request_id)
        if not allocation:
            raise ValueError(f"Request {request_id} has no block allocation")
            
        pool = self._get_pool(allocation.location)
        pool.add_free_blocks(allocation.block_ids)
        del self.allocations[request_id]

    def free_blocks_batched(self, requests: List[Request]):
        """Free blocks for multiple requests."""
        for request in requests:
            self.free_blocks(request.request_id)

    def get_block_table(self, request_id: int) -> List[int]:
        """Get block IDs allocated to a request."""
        allocation = self.allocations.get(request_id)
        if not allocation:
            raise ValueError(f"Request {request_id} has no block allocation")
        return allocation.block_ids

    def get_partial_block_table(self, request_ids: List[int]) -> List[List[int]]:
        """Get block tables for multiple requests."""
        return [self.get_block_table(rid) for rid in request_ids]

    def get_location(self, request_id: int) -> Optional[BlockLocation]:
        """Get location of blocks for a request."""
        allocation = self.allocations.get(request_id)
        return allocation.location if allocation else None

    def swap_requests(self, requests: List[Request], is_swap_in: bool):
        """Swap blocks between CPU and GPU for requests."""
        source_location = BlockLocation.CPU if is_swap_in else BlockLocation.GPU
        target_location = BlockLocation.GPU if is_swap_in else BlockLocation.CPU
        
        source_pool = self._get_pool(source_location)
        target_pool = self._get_pool(target_location)
        
        source_blocks = []
        target_blocks = []
        
        for request in requests:
            request_id = request.request_id
            allocation = self.allocations.get(request_id)
            
            if not allocation:
                raise ValueError(f"Request {request_id} has no block allocation")
            if allocation.location != source_location:
                raise ValueError(
                    f"Request {request_id} blocks are on {allocation.location}, "
                    f"expected {source_location}")

            # Allocate new blocks in target location
            new_blocks = self._ensure_blocks_available(
                target_pool, len(allocation.block_ids))
                
            # Track blocks for swapping
            source_blocks.extend(allocation.block_ids)
            target_blocks.extend(new_blocks)
            
            # Update allocation
            source_pool.add_swapping_blocks(allocation.block_ids)
            allocation.block_ids = new_blocks
            allocation.location = target_location

        # Trigger async swap operation
        self.engine_remote_call_all_workers_async(
            "swap_blocks", requests, source_blocks, target_blocks, is_swap_in)

    def swap_in_requests(self, requests: List[Request]):
        """Swap blocks from CPU to GPU for requests."""
        self.swap_requests(requests, is_swap_in=True)

    def swap_out_requests(self, requests: List[Request]):
        """Swap blocks from GPU to CPU for requests."""
        self.swap_requests(requests, is_swap_in=False)

    def is_all_requests_on_gpu(self, batch_requests: BatchedRequests) -> bool:
        """Check if all requests in batch have blocks on GPU."""
        return all(
            self.get_location(r.request_id) == BlockLocation.GPU 
            for r in batch_requests.requests
        )

    def print_block_usage(self):
        """Print current block usage statistics."""
        for pool in [self.cpu_pool, self.gpu_pool]:
            used_blocks = pool.max_blocks - pool.available_blocks
            usage_pct = (used_blocks / pool.max_blocks) * 100
            swapping_msg = (
                "swapping in" if pool.location == BlockLocation.CPU else "swapping out"
            )
            
            logger.info(
                f"({self.stage}) {pool.location} blocks: "
                f"{used_blocks} / {pool.max_blocks} ({usage_pct:.2f}%) used, "
                f"({len(pool.swapping_blocks)} {swapping_msg})"
            )

    def __repr__(self) -> str:
        return (
            f"BlockManager(max_gpu_blocks={self.gpu_pool.max_blocks}, "
            f"max_cpu_blocks={self.cpu_pool.max_blocks}, "
            f"block_size={self.cache_config.block_size})"
        )
