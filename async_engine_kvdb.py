import torch
import torch.distributed as dist
import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import threading
import uuid
from datetime import datetime, timedelta
import numpy as np
from contextlib import asynccontextmanager
import pickle
import lmdb
import os

@dataclass
class KVCache:
    key_states: torch.Tensor
    value_states: torch.Tensor
    sequence_ids: torch.Tensor
    created_at: datetime
    
    def to_bytes(self) -> bytes:
        return pickle.dumps({
            'key_states': self.key_states.cpu(),
            'value_states': self.value_states.cpu(),
            'sequence_ids': self.sequence_ids.cpu(),
            'created_at': self.created_at
        })
    
    @staticmethod
    def from_bytes(data: bytes) -> 'KVCache':
        cache_dict = pickle.loads(data)
        return KVCache(
            key_states=cache_dict['key_states'].cuda(),
            value_states=cache_dict['value_states'].cuda(),
            sequence_ids=cache_dict['sequence_ids'].cuda(),
            created_at=cache_dict['created_at']
        )

class KVCacheStore:
    def __init__(self, db_path: str, max_cache_size_gb: float = 32.0):
        self.db_path = db_path
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.current_size_bytes = 0
        self.lock = threading.Lock()
        
        # Initialize LMDB environment
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.env = lmdb.open(
            db_path,
            map_size=self.max_cache_size_bytes,
            max_dbs=2
        )
        
        # Create separate databases for cache and metadata
        with self.env.begin(write=True) as txn:
            self.cache_db = self.env.open_db(b'cache', txn=txn)
            self.metadata_db = self.env.open_db(b'metadata', txn=txn)
    
    def _estimate_cache_size(self, cache: KVCache) -> int:
        return (
            cache.key_states.element_size() * cache.key_states.nelement() +
            cache.value_states.element_size() * cache.value_states.nelement() +
            cache.sequence_ids.element_size() * cache.sequence_ids.nelement()
        )
    
    async def store_cache(self, cache: KVCache) -> str:
        cache_id = str(uuid.uuid4())
        cache_size = self._estimate_cache_size(cache)
        
        with self.lock:
            with self.env.begin(write=True) as txn:
                # Check if we need to evict old caches
                while self.current_size_bytes + cache_size > self.max_cache_size_bytes:
                    # Get oldest cache
                    cursor = txn.cursor(self.metadata_db)
                    if not cursor.first():
                        break
                    
                    oldest_id = cursor.key().decode()
                    oldest_size = int(txn.get(cursor.key(), db=self.metadata_db))
                    
                    # Remove oldest cache
                    txn.delete(oldest_id.encode(), db=self.cache_db)
                    txn.delete(cursor.key(), db=self.metadata_db)
                    self.current_size_bytes -= oldest_size
                
                # Store new cache
                txn.put(
                    cache_id.encode(),
                    cache.to_bytes(),
                    db=self.cache_db
                )
                txn.put(
                    f"{cache.created_at.timestamp()}_{cache_id}".encode(),
                    str(cache_size).encode(),
                    db=self.metadata_db
                )
                self.current_size_bytes += cache_size
        
        return cache_id
    
    async def get_cache(self, cache_id: str) -> Optional[KVCache]:
        with self.env.begin() as txn:
            cache_data = txn.get(cache_id.encode(), db=self.cache_db)
            if cache_data is None:
                return None
            return KVCache.from_bytes(cache_data)
    
    async def delete_cache(self, cache_id: str) -> None:
        with self.lock:
            with self.env.begin(write=True) as txn:
                # Find and delete metadata
                cursor = txn.cursor(self.metadata_db)
                for key, value in cursor:
                    if cache_id in key.decode():
                        size = int(value)
                        txn.delete(key, db=self.metadata_db)
                        self.current_size_bytes -= size
                        break
                
                # Delete cache
                txn.delete(cache_id.encode(), db=self.cache_db)
    
    def close(self):
        self.env.close()

class PrefillEngine:
    def __init__(self, model, kv_store: KVCacheStore):
        self.model = model
        self.kv_store = kv_store
        self.running = True
    
    async def process_batch(self, input_ids: torch.Tensor) -> str:
        # Simulate prefill computation
        hidden_states = torch.randn(
            input_ids.shape[0],
            input_ids.shape[1],
            self.model.hidden_size,
            device="cuda"
        )
        
        with torch.no_grad():
            output, kv_cache = self.model(hidden_states)
            
        # Store KV cache in database
        kv_cache.created_at = datetime.now()
        cache_id = await self.kv_store.store_cache(kv_cache)
        return cache_id
    
    async def run(self):
        while self.running:
            # Simulate receiving batches
            batch_size = 32
            seq_length = 512
            input_ids = torch.randint(0, 50000, (batch_size, seq_length)).cuda()
            
            cache_id = await self.process_batch(input_ids)
            print(f"Stored KV cache with ID: {cache_id}")
            await asyncio.sleep(0.1)

class DecodingEngine:
    def __init__(self, model, kv_store: KVCacheStore):
        self.model = model
        self.kv_store = kv_store
        self.running = True
    
    async def decode_step(self, input_ids: torch.Tensor, cache_id: str) -> torch.Tensor:
        # Retrieve KV cache from database
        kv_cache = await self.kv_store.get_cache(cache_id)
        if kv_cache is None:
            raise ValueError(f"No KV cache found for ID: {cache_id}")
        
        # Simulate decoding computation
        hidden_states = torch.randn(
            input_ids.shape[0],
            1,
            self.model.hidden_size,
            device="cuda"
        )
        
        with torch.no_grad():
            output, new_kv_cache = self.model(hidden_states, kv_cache)
            
        # Store updated KV cache
        new_kv_cache.created_at = datetime.now()
        new_cache_id = await self.kv_store.store_cache(new_kv_cache)
        
        # Clean up old cache
        await self.kv_store.delete_cache(cache_id)
        
        # Simulate token generation
        next_token = torch.argmax(output[:, -1:], dim=-1)
        return next_token, new_cache_id
    
    async def run(self, cache_id: str):
        while self.running:
            # Simulate receiving requests
            batch_size = 32
            input_ids = torch.randint(0, 50000, (batch_size, 1)).cuda()
            
            next_token, new_cache_id = await self.decode_step(input_ids, cache_id)
            cache_id = new_cache_id  # Use updated cache for next iteration
            print(f"Generated token using cache ID: {cache_id}")
            await asyncio.sleep(0.05)

# Example usage
async def main():
    from dummy_transformer import DummyTransformer  # Using previous implementation
    
    # Initialize KV cache store
    kv_store = KVCacheStore(
        db_path="./kv_cache.lmdb",
        max_cache_size_gb=32.0
    )
    
    # Initialize model
    model = DummyTransformer().cuda()
    
    # Start prefill engine
    prefill_engine = PrefillEngine(model, kv_store)
    prefill_task = asyncio.create_task(prefill_engine.run())
    
    # Wait for first cache to be generated
    await asyncio.sleep(1)
    
    # Get a cache ID from the store (in practice, this would be communicated between engines)
    with kv_store.env.begin() as txn:
        cursor = txn.cursor(kv_store.cache_db)
        first_cache_id = cursor.first()[0].decode()
    
    # Start decoding engine
    decode_engine = DecodingEngine(model, kv_store)
    decode_task = asyncio.create_task(decode_engine.run(first_cache_id))
    
    # Run for some time
    await asyncio.sleep(60)
    
    # Cleanup
    prefill_engine.running = False
    decode_engine.running = False
    await prefill_task
    await decode_task
    kv_store.close()

if __name__ == "__main__":
    asyncio.run(main())
