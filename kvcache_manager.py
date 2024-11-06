import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import gc

@dataclass
class CacheBlock:
    """Represents a block in the KV cache"""
    block_id: int
    sequence_ids: List[int]  # Sequence IDs stored in this block
    last_access_time: int
    is_pinned: bool = False
    
    # PyTorch tensors for keys and values
    keys: Optional[torch.Tensor] = None    # Shape: [block_size, num_heads, head_dim]
    values: Optional[torch.Tensor] = None  # Shape: [block_size, num_heads, head_dim]

class KVCacheManager:
    def __init__(
        self,
        block_size: int = 1024,
        max_blocks: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        eviction_policy: str = "lru",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.eviction_policy = eviction_policy
        self.blocks: Dict[int, CacheBlock] = {}
        self.access_counter = 0
        
        # Calculate memory requirements per block
        self.block_memory = (
            2 * block_size * num_heads * head_dim * torch.finfo(dtype).bits // 8
        )  # in bytes, *2 for both K and V
        
    def _create_empty_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create empty key and value tensors for a new block"""
        shape = (self.block_size, self.num_heads, self.head_dim)
        keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        values = torch.empty(shape, dtype=self.dtype, device=self.device)
        return keys, values
        
    def allocate_block(self, sequence_ids: List[int]) -> Optional[CacheBlock]:
        """Allocate a new block, evicting if necessary"""
        if len(self.blocks) >= self.max_blocks:
            evicted_block_id = self._choose_block_to_evict()
            if evicted_block_id is None:
                return None  # Cannot evict any blocks
            self._evict_block(evicted_block_id)
            
        # Try to allocate new tensors
        try:
            keys, values = self._create_empty_tensors()
        except torch.cuda.OutOfMemoryError:
            # Emergency eviction if we're out of GPU memory
            self._emergency_memory_cleanup()
            try:
                keys, values = self._create_empty_tensors()
            except torch.cuda.OutOfMemoryError:
                return None
            
        block_id = len(self.blocks)
        block = CacheBlock(
            block_id=block_id,
            sequence_ids=sequence_ids,
            last_access_time=self.access_counter,
            keys=keys,
            values=values
        )
        self.blocks[block_id] = block
        return block
    
    def _emergency_memory_cleanup(self):
        """Perform emergency cleanup when running out of GPU memory"""
        # Clear any unused cached allocators
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # If still needed, evict unpinned blocks until we free enough memory
        if len(self.blocks) > 0:
            blocks_to_evict = sorted(
                [(b.last_access_time, bid) for bid, b in self.blocks.items() if not b.is_pinned]
            )
            
            # Evict up to half of unpinned blocks
            for _, block_id in blocks_to_evict[:len(blocks_to_evict)//2]:
                self._evict_block(block_id)
    
    def _choose_block_to_evict(self) -> Optional[int]:
        """Choose a block to evict based on the configured policy"""
        if self.eviction_policy == "lru":
            # Find least recently used non-pinned block
            lru_block_id = None
            lru_time = float('inf')
            
            for block_id, block in self.blocks.items():
                if not block.is_pinned and block.last_access_time < lru_time:
                    lru_time = block.last_access_time
                    lru_block_id = block_id
            return lru_block_id
            
        elif self.eviction_policy == "fifo":
            # Find first non-pinned block
            for block_id, block in self.blocks.items():
                if not block.is_pinned:
                    return block_id
                    
        return None  # No blocks available for eviction
    
    def _evict_block(self, block_id: int):
        """Evict a block from the cache"""
        if block_id in self.blocks:
            block = self.blocks[block_id]
            # Explicitly free GPU memory
            if block.keys is not None:
                del block.keys
            if block.values is not None:
                del block.values
            del self.blocks[block_id]
            
            # Optional: force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def access_block(self, block_id: int):
        """Update access time for a block"""
        if block_id in self.blocks:
            self.blocks[block_id].last_access_time = self.access_counter
            self.access_counter += 1
            
    def pin_block(self, block_id: int):
        """Pin a block to prevent eviction"""
        if block_id in self.blocks:
            self.blocks[block_id].is_pinned = True
            
    def unpin_block(self, block_id: int):
        """Unpin a block to allow eviction"""
        if block_id in self.blocks:
            self.blocks[block_id].is_pinned = False
            
    def get_total_memory_used(self) -> int:
        """Get total GPU memory used by the cache in bytes"""
        return len(self.blocks) * self.block_memory
    
    def append_to_block(self, block_id: int, position: int, 
                       new_keys: torch.Tensor, new_values: torch.Tensor):
        """Append new key-value pairs to a specific position in a block"""
        if block_id not in self.blocks:
            return
            
        block = self.blocks[block_id]
        if position < 0 or position >= self.block_size:
            return
            
        # Ensure tensors are on the correct device and have correct dtype
        new_keys = new_keys.to(device=self.device, dtype=self.dtype)
        new_values = new_values.to(device=self.device, dtype=self.dtype)
        
        # Update the tensors at the specified position
        block.keys[position:position + new_keys.size(0)] = new_keys
        block.values[position:position + new_values.size(0)] = new_values
        
    def get_block_kv(self, block_id: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get the key-value tensors for a block"""
        if block_id not in self.blocks:
            return None, None
        block = self.blocks[block_id]
        return block.keys, block.values
