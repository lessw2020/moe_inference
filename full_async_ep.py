import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from collections import deque

# ... [Previous code for MultiHeadAttention and ExpertLayer remains unchanged]

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
        
        self.streams = [torch.cuda.Stream() for _ in range(pipeline_depth)]
        self.events = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(6)]

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        chunk_size = seq_len // self.pipeline_depth
        input_chunks = list(x.split(chunk_size, dim=1))
        
        output_chunks = [None] * self.pipeline_depth
        pipeline = deque()

        for i, chunk in enumerate(input_chunks):
            stream = self.streams[i % self.pipeline_depth]
            with torch.cuda.stream(stream):
                chunk = chunk.reshape(-1, self.input_size)

                # Stage 1: Gate computation
                self.events[0][0].record(stream)
                with autocast():
                    gate_logits = self.gate(chunk)
                    gates = F.softmax(gate_logits, dim=-1)
                    selected_experts = torch.argmax(gates, dim=-1)
                self.events[0][1].record(stream)

                # Stage 2: Expert counting and all-reduce
                self.events[1][0].record(stream)
                expert_counts = torch.zeros(self.world_size, self.num_experts, dtype=torch.long, device=chunk.device)
                expert_counts[self.rank] = torch.bincount(selected_experts, minlength=self.num_experts)
                all_reduce_work = dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM, group=dist.group.WORLD, async_op=True)
                self.events[1][1].record(stream)

                # Stage 3: Prepare for all-to-all (can start before all-reduce completes)
                self.events[2][0].record(stream)
                expert_cumsum = torch.cumsum(expert_counts, dim=1)
                input_tensors = [torch.zeros(count.sum(), self.input_size, device=chunk.device) for count in expert_counts]
                for j in range(self.world_size):
                    for k in range(self.num_experts):
                        if j == self.rank:
                            idx = (selected_experts == k).nonzero().squeeze(1)
                            input_tensors[j][expert_cumsum[j, k] - expert_counts[j, k]:expert_cumsum[j, k]] = chunk[idx]
                input_tensor = torch.cat(input_tensors)
                self.events[2][1].record(stream)

                # Ensure all-reduce is complete before all-to-all
                all_reduce_work.wait()

                # Stage 4: All-to-all communication
                self.events[3][0].record(stream)
                output_tensor = torch.empty_like(input_tensor)
                all_to_all_work = dist.all_to_all_single(output_tensor, input_tensor, group=dist.group.WORLD, async_op=True)
                self.events[3][1].record(stream)

                # Stage 5: Expert computation (can start as soon as partial data is available)
                self.events[4][0].record(stream)
                local_outputs = []
                start_idx = 0
                for j, expert in enumerate(self.experts):
                    end_idx = start_idx + expert_counts[self.rank, j*self.world_size:(j+1)*self.world_size].sum()
                    expert_input = output_tensor[start_idx:end_idx]
                    with autocast():
                        local_outputs.append(expert(expert_input))
                    start_idx = end_idx
                local_output = torch.cat(local_outputs)
                self.events[4][1].record(stream)

                # Ensure all-to-all is complete before reverse all-to-all
                all_to_all_work.wait()

                # Stage 6: Reverse all-to-all
                self.events[5][0].record(stream)
                reverse_all_to_all_work = dist.all_to_all_single(input_tensor, local_output, group=dist.group.WORLD, async_op=True)
                self.events[5][1].record(stream)

                # Combine outputs (can start as soon as partial data is available)
                expert_destinations = torch.searchsorted(expert_cumsum, torch.arange(1, chunk.size(0) + 1, device=chunk.device).unsqueeze(0).expand(self.world_size, -1))
                combined_output = torch.zeros(chunk.size(0), self.output_size, device=chunk.device)
                start_idx = 0
                for j in range(self.world_size):
                    end_idx = start_idx + expert_counts[j].sum()
                    combined_output[expert_destinations[j] == self.rank] = input_tensor[start_idx:end_idx]
                    start_idx = end_idx

                # Ensure reverse all-to-all is complete
                reverse_all_to_all_work.wait()

                # Reshape output
                output_chunks[i] = combined_output.view(batch_size, -1, self.output_size)

        # Synchronize all streams
        for stream in self.streams:
            stream.synchronize()

        # Combine all output chunks
        final_output = torch.cat(output_chunks, dim=1)

        # Print timing information
        for i, (start, end) in enumerate(self.events):
            print(f"Rank {self.rank}, MoE Stage {i} time: {start.elapsed_time(end):.2f} ms")

        return final_output

# ... [Rest of the code for HybridParallelTransformerLayer, HybridParallelTransformerModel, etc. remains unchanged]
