# Initialize the cache manager
cache_manager = KVCacheManager(
    block_size=1024,
    max_blocks=32,
    num_heads=32,
    head_dim=128,
    dtype=torch.float16
)

# Allocate a new block
block = cache_manager.allocate_block(sequence_ids=[1, 2, 3])

# Add some key-value pairs
new_keys = torch.randn(10, 32, 128, dtype=torch.float16, device="cuda")
new_values = torch.randn(10, 32, 128, dtype=torch.float16, device="cuda")
cache_manager.append_to_block(block.block_id, position=0, new_keys=new_keys, new_values=new_values)
