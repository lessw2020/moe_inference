import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from collections import deque

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_linear(attn_output)

class ExpertLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 4 * input_size)
        self.fc2 = nn.Linear(4 * input_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class FineGrainedPipelinedMoELayer(nn.Module):
    def __init__(self, num_experts, input_size, output_size, num_tokens_per_device, world_size, rank, pipeline_depth=3):
        super().__init__()
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.num_tokens_per_device = num_tokens_per_device
        self.input_size = input_size
        self.output_size = output_size
        self.pipeline_depth = pipeline_depth

        self.gate = nn.Linear(input_size, num_experts)
        self.experts = nn.ModuleList([ExpertLayer(input_size, output_size) for _ in range(num_experts // world_size)])
        
        self.events = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(6)]

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        chunk_size = seq_len // self.pipeline_depth
        input_chunks = list(x.split(chunk_size, dim=1))
        
        output_chunks = []
        pipeline = deque()

        for i, chunk in enumerate(input_chunks):
            chunk = chunk.reshape(-1, self.input_size)

            # Stage 1: Gate computation
            self.events[0][0].record()
            with autocast():
                gate_logits = self.gate(chunk)
                gates = F.softmax(gate_logits, dim=-1)
                selected_experts = torch.argmax(gates, dim=-1)
            self.events[0][1].record()

            # Stage 2: Expert counting and all-reduce
            self.events[1][0].record()
            expert_counts = torch.zeros(self.world_size, self.num_experts, dtype=torch.long, device=chunk.device)
            expert_counts[self.rank] = torch.bincount(selected_experts, minlength=self.num_experts)
            dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM, group=dist.group.WORLD, async_op=True)
            self.events[1][1].record()

            # Stage 3: Prepare for all-to-all
            self.events[2][0].record()
            expert_cumsum = torch.cumsum(expert_counts, dim=1)
            input_tensors = [torch.zeros(count.sum(), self.input_size, device=chunk.device) for count in expert_counts]
            for j in range(self.world_size):
                for k in range(self.num_experts):
                    if j == self.rank:
                        idx = (selected_experts == k).nonzero().squeeze(1)
                        input_tensors[j][expert_cumsum[j, k] - expert_counts[j, k]:expert_cumsum[j, k]] = chunk[idx]
            input_tensor = torch.cat(input_tensors)
            self.events[2][1].record()

            # Stage 4: All-to-all communication
            self.events[3][0].record()
            output_tensor = torch.empty_like(input_tensor)
            dist.all_to_all_single(output_tensor, input_tensor, group=dist.group.WORLD, async_op=True)
            self.events[3][1].record()

            # Stage 5: Expert computation
            self.events[4][0].record()
            local_outputs = []
            start_idx = 0
            for j, expert in enumerate(self.experts):
                end_idx = start_idx + expert_counts[self.rank, j*self.world_size:(j+1)*self.world_size].sum()
                expert_input = output_tensor[start_idx:end_idx]
                with autocast():
                    local_outputs.append(expert(expert_input))
                start_idx = end_idx
            local_output = torch.cat(local_outputs)
            self.events[4][1].record()

            # Stage 6: Reverse all-to-all
            self.events[5][0].record()
            dist.all_to_all_single(input_tensor, local_output, group=dist.group.WORLD, async_op=True)
            self.events[5][1].record()

            pipeline.append((expert_counts, expert_cumsum, input_tensor, chunk.size(0)))

            if len(pipeline) == self.pipeline_depth:
                expert_counts, expert_cumsum, input_tensor, chunk_size = pipeline.popleft()
                torch.cuda.current_stream().synchronize()

                # Combine outputs
                expert_destinations = torch.searchsorted(expert_cumsum, torch.arange(1, chunk_size + 1, device=chunk.device).unsqueeze(0).expand(self.world_size, -1))
                combined_output = torch.zeros(chunk_size, self.output_size, device=chunk.device)
                start_idx = 0
                for j in range(self.world_size):
                    end_idx = start_idx + expert_counts[j].sum()
                    combined_output[expert_destinations[j] == self.rank] = input_tensor[start_idx:end_idx]
                    start_idx = end_idx

                output_chunks.append(combined_output)

        # Process remaining items in the pipeline
        while pipeline:
            expert_counts, expert_cumsum, input_tensor, chunk_size = pipeline.popleft()
            torch.cuda.current_stream().synchronize()

            expert_destinations = torch.searchsorted(expert_cumsum, torch.arange(1, chunk_size + 1, device=chunk.device).unsqueeze(0).expand(self.world_size, -1))
            combined_output = torch.zeros(chunk_size, self.output_size, device=chunk.device)
            start_idx = 0
            for j in range(self.world_size):
                end_idx = start_idx + expert_counts[j].sum()
                combined_output[expert_destinations[j] == self.rank] = input_tensor[start_idx:end_idx]
                start_idx = end_idx

            output_chunks.append(combined_output)

        # Combine all output chunks
        final_output = torch.cat(output_chunks)
        final_output = final_output.view(batch_size, seq_len, self.output_size)

        # Print timing information
        torch.cuda.synchronize()
        stage_names = ["Gate", "Count", "Prepare", "All-to-All", "Expert", "Reverse"]
        for i, (start, end) in enumerate(self.events):
            print(f"Rank {self.rank}, MoE {stage_names[i]} time: {start.elapsed_time(end):.2f} ms")

        return final_output

class HybridParallelTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_experts, num_tokens_per_device, world_size, rank, pipeline_depth):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = FineGrainedPipelinedMoELayer(num_experts, d_model, d_model, num_tokens_per_device, world_size, rank, pipeline_depth)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Data parallel attention
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # Expert parallel MoE
        moe_output = self.moe(x)
        x = self.norm2(x + moe_output)
        
        return x

class HybridParallelTransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, num_experts, num_tokens_per_device, world_size, rank, pipeline_depth):
        super().__init__()
        self.layers = nn.ModuleList([
            HybridParallelTransformerLayer(d_model, num_heads, num_experts, num_tokens_per_device, world_size, rank, pipeline_depth)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_distributed_inference(rank, world_size, num_layers, d_model, num_heads, num_experts, pipeline_depth):
    setup(rank, world_size)

    batch_size = 32
    seq_len = 64
    num_tokens_per_device = batch_size * seq_len // world_size

    model = HybridParallelTransformerModel(
        num_layers, d_model, num_heads, num_experts, num_tokens_per_device, world_size, rank, pipeline_depth
    ).to(rank)
    
    # Ensure all processes have the same initial model parameters
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Generate input data
    if rank == 0:
        input_tensor = torch.randn(batch_size, seq_len, d_model, device=rank)
    else:
        input_tensor = torch.empty(batch_size, seq_len, d_model, device=rank)
    dist.broadcast(input_tensor, src=0)

    # Perform inference
    with torch.no_grad(), autocast():
        output = model(input_tensor)

    print(f"Rank {rank}, Output shape: {output.shape}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    num_layers = 4
    d_model = 1024
    num_heads = 16
    num_experts = 32
    pipeline_depth = 3

    torch.multiprocessing.spawn(
        run_distributed_inference,
        args=(world_size, num_layers, d_model, num_heads, num_experts, pipeline_depth),
        nprocs=world_size,
        join=True
    )
