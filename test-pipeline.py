import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast

class ExpertLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

class PipelinedMoELayer(nn.Module):
    def __init__(self, num_experts, input_size, output_size, num_tokens_per_device, world_size, rank):
        super().__init__()
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.num_tokens_per_device = num_tokens_per_device
        self.input_size = input_size
        self.output_size = output_size

        self.gate = nn.Linear(input_size, num_experts)
        self.experts = nn.ModuleList([ExpertLayer(input_size, output_size) for _ in range(num_experts // world_size)])
        
        # Create CUDA events for synchronization
        self.start_events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        self.end_events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, self.input_size)

        # Stage 1: Gate computation and expert selection
        self.start_events[0].record()
        with autocast():
            gate_logits = self.gate(x)
            gates = F.softmax(gate_logits, dim=-1)
            selected_experts = torch.argmax(gates, dim=-1)

        expert_counts = torch.zeros(self.world_size, self.num_experts, dtype=torch.long, device=x.device)
        expert_counts[self.rank] = torch.bincount(selected_experts, minlength=self.num_experts)
        dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        self.end_events[0].record()

        # Stage 2: Prepare for all-to-all communication
        self.start_events[1].record()
        expert_cumsum = torch.cumsum(expert_counts, dim=1)
        input_tensors = [torch.zeros(count.sum(), self.input_size, device=x.device) for count in expert_counts]
        for i in range(self.world_size):
            for j in range(self.num_experts):
                if i == self.rank:
                    idx = (selected_experts == j).nonzero().squeeze(1)
                    input_tensors[i][expert_cumsum[i, j] - expert_counts[i, j]:expert_cumsum[i, j]] = x[idx]
        input_tensor = torch.cat(input_tensors)
        output_tensor = torch.empty_like(input_tensor)
        self.end_events[1].record()

        # Stage 3: All-to-all communication and expert computation
        self.start_events[2].record()
        dist.all_to_all_single(output_tensor, input_tensor, group=dist.group.WORLD)

        # Process data with local experts
        local_outputs = []
        start_idx = 0
        for i, expert in enumerate(self.experts):
            end_idx = start_idx + expert_counts[self.rank, i*self.world_size:(i+1)*self.world_size].sum()
            expert_input = output_tensor[start_idx:end_idx]
            with autocast():
                local_outputs.append(expert(expert_input))
            start_idx = end_idx
        local_output = torch.cat(local_outputs)

        # Reverse all-to-all communication
        dist.all_to_all_single(input_tensor, local_output, group=dist.group.WORLD)
        self.end_events[2].record()

        # Wait for all stages to complete
        torch.cuda.synchronize()

        # Combine outputs
        expert_destinations = torch.searchsorted(expert_cumsum, torch.arange(1, self.num_tokens_per_device + 1, device=x.device).unsqueeze(0).expand(self.world_size, -1))
        combined_output = torch.zeros(x.shape[0], self.output_size, device=x.device)
        start_idx = 0
        for i in range(self.world_size):
            end_idx = start_idx + expert_counts[i].sum()
            combined_output[expert_destinations[i] == self.rank] = input_tensor[start_idx:end_idx]
            start_idx = end_idx

        # Print timing information
        for i in range(3):
            print(f"Rank {self.rank}, Stage {i+1} time: {self.start_events[i].elapsed_time(self.end_events[i]):.2f} ms")

        return combined_output.view(*original_shape[:-1], self.output_size)

class PipelinedMoEModel(nn.Module):
    def __init__(self, num_experts, input_size, hidden_size, output_size, num_layers, num_tokens_per_device, world_size, rank):
        super().__init__()
        self.layers = nn.ModuleList([PipelinedMoELayer(num_experts, input_size if i == 0 else hidden_size, 
                                                       hidden_size, num_tokens_per_device, world_size, rank) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_distributed_inference(rank, world_size, num_experts, input_size, hidden_size, output_size, num_layers):
    setup(rank, world_size)

    batch_size = 32
    seq_len = 64
    num_tokens_per_device = batch_size * seq_len // world_size

    model = PipelinedMoEModel(num_experts, input_size, hidden_size, output_size, num_layers, num_tokens_per_device, world_size, rank).to(rank)
    
    # Ensure all processes have the same initial model parameters
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Generate input data
    if rank == 0:
        input_tensor = torch.randn(batch_size, seq_len, input_size, device=rank)
    else:
        input_tensor = torch.empty(batch_size, seq_len, input_size, device=rank)
    dist.broadcast(input_tensor, src=0)

    # Perform inference
    with torch.no_grad(), autocast():
        output = model(input_tensor)

    print(f"Rank {rank}, Output shape: {output.shape}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    num_experts = 32
    input_size = 512
    hidden_size = 1024
    output_size = 256
    num_layers = 4

    torch.multiprocessing.spawn(
        run_distributed_inference,
        args=(world_size, num_experts, input_size, hidden_size, output_size, num_layers),
        nprocs=world_size,
        join=True
    )
