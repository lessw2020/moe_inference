import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.nn as nn
import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import queue
import threading
from contextlib import asynccontextmanager

@dataclass
class KVCache:
    key_states: torch.Tensor
    value_states: torch.Tensor
    sequence_ids: torch.Tensor
    
class DummyTransformer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states, past_key_values=None):
        batch_size = hidden_states.shape[0]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        if past_key_values is not None:
            k = torch.cat([past_key_values.key_states, k], dim=1)
            v = torch.cat([past_key_values.value_states, v], dim=1)
            
        return hidden_states, KVCache(k, v, torch.arange(k.shape[1]))

class PrefillEngine:
    def __init__(self, model_path: str, world_size: int):
        self.model = DummyTransformer().cuda()
        self.kv_cache_queue = queue.Queue()
        self.world_size = world_size
        self.running = True
        
        # Initialize distributed
        dist.init_process_group(backend="nccl")
        rpc.init_rpc(
            f"prefill_worker_{dist.get_rank()}",
            rank=dist.get_rank(),
            world_size=world_size
        )
        
    async def process_batch(self, input_ids: torch.Tensor) -> None:
        # Simulate prefill computation
        hidden_states = torch.randn(
            input_ids.shape[0],
            input_ids.shape[1],
            self.model.hidden_size,
            device="cuda"
        )
        
        with torch.no_grad():
            output, kv_cache = self.model(hidden_states)
            
        # Store KV cache for decoding engine
        self.kv_cache_queue.put(kv_cache)
        
        # Notify decoding engine via RPC
        rpc.rpc_sync(
            f"decode_worker_0",
            self._notify_kv_cache_ready,
            args=(dist.get_rank(),)
        )
    
    @staticmethod
    def _notify_kv_cache_ready(prefill_rank: int) -> None:
        pass  # Decoding engine will handle this callback
        
    async def run(self):
        while self.running:
            # Simulate receiving batches
            batch_size = 32
            seq_length = 512
            input_ids = torch.randint(0, 50000, (batch_size, seq_length)).cuda()
            
            await self.process_batch(input_ids)
            await asyncio.sleep(0.1)  # Simulate batch arrival time
            
    def shutdown(self):
        self.running = False
        dist.destroy_process_group()
        rpc.shutdown()

class DecodingEngine:
    def __init__(self, model_path: str, world_size: int):
        self.model = DummyTransformer().cuda()
        self.kv_cache_map: Dict[int, KVCache] = {}
        self.world_size = world_size
        self.running = True
        
        # Initialize distributed
        dist.init_process_group(backend="nccl")
        rpc.init_process_group("decode_worker", rank=dist.get_rank(), world_size=world_size)
        
    @asynccontextmanager
    async def get_kv_cache(self, prefill_rank: int) -> KVCache:
        # Wait for notification from prefill engine
        while prefill_rank not in self.kv_cache_map:
            await asyncio.sleep(0.01)
            
        try:
            yield self.kv_cache_map[prefill_rank]
        finally:
            del self.kv_cache_map[prefill_rank]
            
    async def decode_step(self, input_ids: torch.Tensor, prefill_rank: int) -> torch.Tensor:
        async with self.get_kv_cache(prefill_rank) as kv_cache:
            # Simulate decoding computation
            hidden_states = torch.randn(
                input_ids.shape[0],
                1,  # Only decode one token at a time
                self.model.hidden_size,
                device="cuda"
            )
            
            with torch.no_grad():
                output, new_kv_cache = self.model(hidden_states, kv_cache)
                
            # Simulate token generation
            next_token = torch.argmax(output[:, -1:], dim=-1)
            return next_token
            
    def handle_kv_cache_ready(self, prefill_rank: int) -> None:
        # Receive KV cache from prefill engine via NCCL
        kv_cache = torch.zeros(1)  # Placeholder for NCCL receive
        dist.recv(kv_cache, src=prefill_rank)
        self.kv_cache_map[prefill_rank] = kv_cache
        
    async def run(self):
        while self.running:
            # Simulate receiving requests
            batch_size = 32
            input_ids = torch.randint(0, 50000, (batch_size, 1)).cuda()
            prefill_rank = dist.get_rank() % (self.world_size - 1)
            
            next_token = await self.decode_step(input_ids, prefill_rank)
            await asyncio.sleep(0.05)  # Simulate request arrival time
            
    def shutdown(self):
        self.running = False
        dist.destroy_process_group()
        rpc.shutdown()

# Example usage
async def main():
    world_size = 4  # Total number of processes
    model_path = "dummy_model.pt"
    
    # Start prefill engine in processes 0-2
    prefill_engines = []
    for rank in range(world_size - 1):
        engine = PrefillEngine(model_path, world_size)
        prefill_engines.append(engine)
        asyncio.create_task(engine.run())
        
    # Start decoding engine in process 3
    decode_engine = DecodingEngine(model_path, world_size)
    decode_task = asyncio.create_task(decode_engine.run())
    
    # Run for some time
    await asyncio.sleep(60)
    
    # Cleanup
    for engine in prefill_engines:
        engine.shutdown()
    decode_engine.shutdown()
    
if __name__ == "__main__":
    asyncio.run(main())
