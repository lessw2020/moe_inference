from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from distserve.config import ContextStageSchedConfig, ParallelConfig
from distserve.logger import init_logger
from distserve.request import Request, BatchedRequests, MigratingRequest
from distserve.block_manager import BlockManager

logger = init_logger(__name__)


class SchedulerPolicy(Enum):
    """Enumeration of supported scheduler policies."""
    FCFS = "fcfs"
    # Easy to add new policies here


@dataclass
class BatchConstraints:
    """Encapsulates batch processing constraints."""
    max_batch_size: int
    max_tokens_per_batch: int
    max_gpu_blocks: int


class ContextStageScheduler(ABC):
    """
    Abstract base class for context stage scheduling.
    
    Maintains and manages requests in the system, supporting basic operations for
    request management and batch processing. Simpler than DecodingStageScheduler
    as each request is processed by a single context stage.
    """

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue."""
        pass

    @abstractmethod
    def abort_request(self, request_id: int) -> None:
        """Remove a request from the scheduler."""
        pass

    @abstractmethod
    def get_next_batch_and_pop(self) -> BatchedRequests:
        """
        Get and remove the next batch of requests for processing.
        
        Returns:
            BatchedRequests: The next batch of requests to process.
        """
        pass

    @abstractmethod
    def get_num_waiting_requests(self) -> int:
        """Return the number of requests waiting for processing."""
        pass

    @abstractmethod
    def print_status(self) -> None:
        """Print current scheduler status."""
        pass

    def on_finish_requests(self, batch: BatchedRequests) -> None:
        """Handle completion of a batch of requests."""
        pass

    def on_request_migrated(self, migrated_request: MigratingRequest) -> None:
        """Handle request migration to the decoding stage."""
        pass

    def post_process(self) -> None:
        """Perform post-iteration processing."""
        pass


class ContextStageFCFSScheduler(ContextStageScheduler):
    """
    First-come-first-serve (FCFS) scheduler implementation.
    
    Manages request scheduling using FCFS policy while respecting batch size,
    token count, and GPU block constraints.
    """

    def __init__(
        self,
        sched_config: ContextStageSchedConfig,
        parallel_config: ParallelConfig,
        block_manager: BlockManager
    ) -> None:
        if sched_config.policy != SchedulerPolicy.FCFS.value:
            raise ValueError(f"Invalid policy for FCFS scheduler: {sched_config.policy}")

        self.constraints = BatchConstraints(
            max_batch_size=sched_config.max_batch_size,
            max_tokens_per_batch=sched_config.max_tokens_per_batch,
            max_gpu_blocks=block_manager.max_num_gpu_blocks
        )
        self.block_manager = block_manager
        self.block_size = block_manager.cache_config.block_size
        
        self.waiting_queue: List[Request] = []
        self.unaccepted_queue: List[Request] = []
        self.num_on_fly_request_blocks: int = 0

    def _calculate_blocks_needed(self, length: int) -> int:
        """Calculate number of blocks needed for given length."""
        return (length + self.block_size - 1) // self.block_size

    def _can_add_to_batch(self, request: Request, current_batch: BatchedRequests) -> bool:
        """
        Check if request can be added to current batch based on constraints.
        
        Args:
            request: Request to potentially add
            current_batch: Current batch being built
            
        Returns:
            bool: True if request can be added, False otherwise
        """
        if len(current_batch) >= self.constraints.max_batch_size:
            return False

        new_token_count = current_batch.get_num_input_tokens() + request.get_num_input_tokens()
        if new_token_count > self.constraints.max_tokens_per_batch:
            return False

        # Calculate total blocks needed including current request
        total_blocks = sum(
            self._calculate_blocks_needed(len(req.prompt_token_ids))
            for req in current_batch.requests + [request]
        )
        
        # Add blocks from unaccepted and in-flight requests
        total_blocks += sum(
            self._calculate_blocks_needed(len(req.prompt_token_ids))
            for req in self.unaccepted_queue
        )
        total_blocks += self.num_on_fly_request_blocks

        return total_blocks <= self.constraints.max_gpu_blocks

    def add_request(self, request: Request) -> None:
        self.waiting_queue.append(request)

    def abort_request(self, request_id: int) -> None:
        self.waiting_queue = [req for req in self.waiting_queue 
                            if req.request_id != request_id]

    def get_next_batch_and_pop(self) -> BatchedRequests:
        next_batch = BatchedRequests()

        while self.waiting_queue and self._can_add_to_batch(self.waiting_queue[0], next_batch):
            request = self.waiting_queue.pop(0)
            next_batch.add_request(request)

        self.num_on_fly_request_blocks += sum(
            self._calculate_blocks_needed(req.get_input_len())
            for req in next_batch.requests
        )

        return next_batch

    def on_finish_requests(self, batch: BatchedRequests) -> None:
        self.unaccepted_queue.extend(
            request for request in batch.requests if not request.is_finished
        )
        
        self.num_on_fly_request_blocks -= sum(
            self._calculate_blocks_needed(req.get_input_len())
            for req in batch.requests
        )

    def on_request_migrated(self, migrated_request: MigratingRequest) -> None:
        self.unaccepted_queue = [req for req in self.unaccepted_queue 
                                if req.request_id != migrated_request.req.request_id]

    def get_num_waiting_requests(self) -> int:
        return len(self.waiting_queue)

    def print_status(self) -> None:
        logger.info(
            f"(context) {len(self.waiting_queue)} waiting, "
            f"{len(self.unaccepted_queue)} finished but unaccepted, "
            f"{self.num_on_fly_request_blocks} blocks occupied by on-the-fly requests"
        )

    def __repr__(self) -> str:
        return (
            f"FCFS(max_batch_size={self.constraints.max_batch_size}, "
            f"max_tokens_per_batch={self.constraints.max_tokens_per_batch})"
        )


def get_context_stage_scheduler(
    sched_config: ContextStageSchedConfig,
    parallel_config: ParallelConfig,
    block_manager: BlockManager
) -> ContextStageScheduler:
    """
    Factory function to create appropriate scheduler based on policy.
    
    Args:
        sched_config: Scheduling configuration
        parallel_config: Parallel processing configuration
        block_manager: Block management system
        
    Returns:
        ContextStageScheduler: Configured scheduler instance
        
    Raises:
        ValueError: If unsupported scheduler policy is specified
    """
    if sched_config.policy == SchedulerPolicy.FCFS.value:
        return ContextStageFCFSScheduler(sched_config, parallel_config, block_manager)
    
    raise ValueError(f"Unsupported scheduler policy: {sched_config.policy}")
