"""LLM Engine implementation for distributed text generation."""
import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, List, Dict, Protocol, TypeVar, Generic
import time
import logging

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup
import torch

from distserve.logger import init_logger
from distserve.config import ModelConfig, ParallelConfig, CacheConfig
from distserve.request import Request, BatchedRequests, MigratingRequest
from distserve.utils import cudaMemoryIpcHandle, Stage
from distserve.lifetime import LifetimeEvent, LifetimeEventType
from distserve.tokenizer import get_tokenizer
from distserve.block_manager import BlockManager
from distserve.worker import ParaWorker
from distserve.context_stage_scheduler import ContextStageSchedConfig, ContextStageScheduler
from distserve.decoding_stage_scheduler import DecodingStageSchedConfig, DecodingStageScheduler

logger = logging.getLogger(__name__)

# Configuration constants
class EngineConfig:
    """Engine configuration constants."""
    CONTEXT_SLEEP_NO_REQUEST: float = 0.003
    DECODING_SLEEP_NO_REQUEST: float = 0.003
    EVENT_LOOP_SLEEP: float = 0
    STATUS_PRINT_INTERVAL: float = 1


@dataclass
class StepOutput:
    """Output from a single inference step."""
    request: Request
    request_id: int = field(init=False)
    prompt: str = field(init=False)
    new_token: str
    new_token_id: int
    is_finished: bool = field(init=False)

    def __post_init__(self):
        self.request_id = self.request.request_id
        self.prompt = self.request.prompt
        self.is_finished = self.request.is_finished


class EngineCallbacks(Protocol):
    """Protocol defining engine callback interfaces."""
    def on_step_output(self, request_id: int, output: StepOutput) -> None: ...
    def on_lifetime_event(self, request_id: int, event: LifetimeEvent, is_decoding: bool = False) -> None: ...


class WorkerPool:
    """Manages a pool of parallel workers."""
    
    def __init__(self, 
                stage: Stage,
                parallel_config: ParallelConfig,
                placement_groups: List[PlacementGroup],
                model_config: ModelConfig):
        self.stage = stage
        self.parallel_config = parallel_config
        self.placement_groups = placement_groups
        self.model_config = model_config
        self.workers: List[List[ParaWorker]] = []

    async def initialize(self) -> None:
        """Initialize worker pool."""
        pp_id = torch.ops.nccl_ops.generate_nccl_id()
        layer_per_pg = self.model_config.get_num_layers() // len(self.placement_groups)
        layer_per_pp = self.model_config.get_num_layers(self.parallel_config)
        pp_per_pg = layer_per_placement_group // layer_per_pp

        init_handlers = []
        for i in range(self.parallel_config.pipeline_parallel_size):
            workers = []
            pg_index = i // pp_per_pg
            tp_id = torch.ops.nccl_ops.generate_nccl_id()
            cur_pg = self.placement_groups[pg_index]
            
            for j in range(self.parallel_config.tensor_parallel_size):
                config = copy.deepcopy(self.parallel_config)
                config.pipeline_parallel_rank = i
                config.tensor_parallel_rank = j
                
                worker = self._create_worker(i, j, config, cur_pg, pp_id, tp_id)
                workers.append(worker)
                init_handlers.append(worker.ready.remote())
                
            self.workers.append(workers)
        
        await asyncio.wait(init_handlers)

    def _create_worker(self, pp_rank: int, tp_rank: int, 
                      config: ParallelConfig, placement_group: PlacementGroup,
                      pp_id: bytes, tp_id: bytes) -> ParaWorker:
        """Create a single worker."""
        worker_id = pp_rank * self.parallel_config.tensor_parallel_size + tp_rank
        return ParaWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group
            )
        ).remote(
            worker_id=worker_id,
            stage=self.stage,
            model_config=self.model_config,
            cache_config=self.cache_config,
            parallel_config=config,
            pipeline_parallel_id=pp_id,
            tensor_parallel_id=tp_id,
        )

    async def call_all_workers(self, func_name: str, *args) -> List:
        """Call a function on all workers and wait for results."""
        futures = []
        for stage in self.workers:
            for worker in stage:
                futures.append(getattr(worker, func_name).remote(*args))
        return await asyncio.gather(*futures)


class EnginePipeline:
    """Manages the pipeline of batched requests."""

    def __init__(self, pipeline_size: int):
        self.pipeline_size = pipeline_size
        self.batches: List[BatchedRequests] = []
        self.futures: List[Optional[asyncio.Future]] = []

    def push(self, batch: BatchedRequests, future: Optional[asyncio.Future] = None):
        """Push a batch into the pipeline."""
        self.batches.append(batch)
        self.futures.append(future)

    def is_full(self) -> bool:
        """Check if pipeline is full."""
        return len(self.batches) == self.pipeline_size

    async def process_completed(self) -> Optional[tuple]:
        """Process the oldest completed batch."""
        if not self.batches or self.futures[0] is None:
            self.pop()
            return None

        result = await self.futures[0]
        return (self.batches[0], result)

    def pop(self) -> None:
        """Remove oldest batch from pipeline."""
        if self.batches:
            self.batches.pop(0)
            self.futures.pop(0)


class BaseLLMEngine(ABC):
    """Base class for LLM engines."""

    def __init__(self,
                 stage: Stage,
                 model_config: ModelConfig,
                 parallel_config: ParallelConfig,
                 cache_config: CacheConfig,
                 placement_groups: List[PlacementGroup],
                 callbacks: EngineCallbacks):
        self.stage = stage
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.callbacks = callbacks

        self.worker_pool = WorkerPool(stage, parallel_config, placement_groups, model_config)
        self.pipeline = EnginePipeline(parallel_config.pipeline_parallel_size)
        self.tokenizer = get_tokenizer(model_config)
        self.block_manager = None

    async def initialize(self) -> None:
        """Initialize engine components."""
        await self.worker_pool.initialize()
        await self._init_model()
        await self._init_cache()
        self._init_scheduler()

    async def _init_model(self) -> None:
        """Initialize model on all workers."""
        await self.worker_pool.call_all_workers("init_model")

    async def _init_cache(self) -> None:
        """Initialize cache on all workers."""
        num_gpu_blocks, num_cpu_blocks = await self._profile_blocks()
        if self.stage == Stage.CONTEXT:
            num_cpu_blocks = 1

        handles = await self.worker_pool.call_all_workers(
            "init_kvcache_and_swap", num_gpu_blocks, num_cpu_blocks)
        
        self.block_manager = BlockManager(
            self.stage, num_gpu_blocks, num_cpu_blocks,
            self.model_config, self.parallel_config, self.cache_config,
            self.worker_pool.call_all_workers)

    @abstractmethod
    def _init_scheduler(self) -> None:
        """Initialize scheduler."""
        pass

    async def _profile_blocks(self) -> tuple[int, int]:
        """Profile available GPU and CPU blocks."""
        return await self.worker_pool.workers[0][0]._profile_num_available_blocks.remote(
            self.cache_config.block_size,
            self.cache_config.gpu_memory_utilization,
            self.cache_config.cpu_swap_space)

    async def step(self) -> None:
        """Execute one step of inference."""
        batch = self._get_next_batch()
        if not batch:
            await self._handle_empty_batch()
            return

        await self._process_batch(batch)
        
        if self.pipeline.is_full():
            await self._handle_pipeline_output()

    @abstractmethod
    def _get_next_batch(self) -> Optional[BatchedRequests]:
        """Get next batch from scheduler."""
        pass

    async def _handle_empty_batch(self) -> None:
        """Handle case when no batch is available."""
        self.pipeline.push(BatchedRequests())
        await asyncio.sleep(self._get_sleep_time())

    def _get_sleep_time(self) -> float:
        """Get appropriate sleep time."""
        return (EngineConfig.CONTEXT_SLEEP_NO_REQUEST 
                if self.stage == Stage.CONTEXT 
                else EngineConfig.DECODING_SLEEP_NO_REQUEST)

    async def _process_batch(self, batch: BatchedRequests) -> None:
        """Process a batch of requests."""
        self._allocate_blocks(batch)
        self._log_batch_start(batch)
        
        futures = await self._execute_batch(batch)
        self.pipeline.push(batch, futures[(self.parallel_config.pipeline_parallel_size - 1) * 
                                        self.parallel_config.tensor_parallel_size])

    async def _execute_batch(self, batch: BatchedRequests) -> List:
        """Execute batch on workers."""
        return await self.worker_pool.call_all_workers(
            "step",
            batch.get_request_ids(),
            batch.get_input_tokens_batched(),
            batch.get_first_token_indexes(),
            self.block_manager.get_partial_block_table(batch.get_request_ids()))

    @abstractmethod
    async def _handle_pipeline_output(self) -> None:
        """Handle output from pipeline."""
        pass

    async def run(self) -> None:
        """Run engine event loop."""
        async def step_loop():
            while True:
                await self.step()
                await asyncio.sleep(EngineConfig.EVENT_LOOP_SLEEP)

        async def status_loop():
            while True:
                self.print_status()
                await asyncio.sleep(EngineConfig.STATUS_PRINT_INTERVAL)

        await asyncio.gather(step_loop(), status_loop())

    @abstractmethod
    def print_status(self) -> None:
        """Print engine status."""
        pass


class ContextLLMEngine(BaseLLMEngine):
    """Context stage LLM engine implementation."""

    def __init__(self, *args, bridge_queue: asyncio.Queue[MigratingRequest], **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge_queue = bridge_queue

    def _init_scheduler(self) -> None:
        self.scheduler = get_context_stage_scheduler(
            self.sched_config, self.parallel_config, self.block_manager)

    async def _handle_pipeline_output(self) -> None:
        output = await self.pipeline.process_completed()
        if output:
            batch, token_ids = output
            await self._process_output(batch, token_ids)
            
    async def _process_output(self, batch: BatchedRequests, token_ids: List[int]) -> None:
        tokens = self._decode_tokens(token_ids)
        batch.finish_one_iteration(tokens, token_ids, time.time())
        
        for request, token, token_id in zip(batch.requests, tokens, token_ids):
            self._handle_request_output(request, token, token_id)
            
        if not request.is_finished:
            await self._migrate_request(request)
        else:
            self._cleanup_request(request)

    def _cleanup_request(self, request: Request) -> None:
        """Clean up resources for finished request."""
        self.block_manager.free_blocks(request.request_id)
        self.worker_pool.call_all_workers("clear_request_resource", request.request_id)

    async def _migrate_request(self, request: Request) -> None:
        """Migrate request to decoding stage."""
        migrating_req = MigratingRequest(
            request,
            self.block_manager.get_block_table(request.request_id),
            self.parallel_config)
        await self.bridge_queue.put(migrating_req)


class DecodingLLMEngine(BaseLLMEngine):
    """Decoding stage LLM engine implementation."""

    def __init__(self, *args, bridge_queue: asyncio.Queue[MigratingRequest], 
                 context_callback: Callable[[MigratingRequest], None], **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge_queue = bridge_queue
        self.context_callback = context_callback

    async def run(self) -> None:
        """Run decoding engine event loops."""
        async def bridge_loop():
            while True:
                req = await self.bridge_queue.get()
                await self.scheduler.add_request(req)
                self.bridge_queue.task_done()

        await asyncio.gather(bridge_loop(), super().run())

    async def _migrate_blocks(self, req: MigratingRequest) -> None:
        """Migrate blocks from context to decoding stage."""
        self._backup_and_allocate_blocks(req)
        await self._transfer_blocks(req)
        self.context_callback(req)

    def _backup_and_allocate_blocks(self, req: MigratingRequest) -> None:
        """Backup request state and allocate new blocks."""
        tokens_backup = (req.req.generated_tokens, req.req.generated_token_ids)
        req.req.generated_tokens = []
        req.req.generated_token_ids = []
        
        self.block_manager.allocate_blocks(req.req)
        
        req.req.generated_tokens, req.req.generated_token_ids = tokens_backup
