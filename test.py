import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

class MoELayer(nn.Module):
    def __init__(self, num_experts, input_size, output_size, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_size, num_experts)
        self.experts = nn.ModuleList([ExpertLayer(input_size, output_size) for _ in range(num_experts)])

    def forward(self, x):
        # Gate computation
        gate_logits = self.gate(x)
        gates = F.softmax(gate_logits, dim=-1)

        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)])
        
        # Gather outputs from selected experts
        batch_size, seq_len, _ = x.size()
        expert_outputs = expert_outputs.transpose(0, 1)  # [batch_size, num_experts, seq_len, output_size]
        output = torch.einsum('bne,bnec->bc', top_k_gates, expert_outputs[torch.arange(batch_size).unsqueeze(1), top_k_indices])
        
        return output

class MoEModel(nn.Module):
    def __init__(self, num_experts, input_size, hidden_size, output_size, num_layers, top_k=2):
        super().__init__()
        self.layers = nn.ModuleList([MoELayer(num_experts, input_size if i == 0 else hidden_size, 
                                              hidden_size, top_k) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# Example usage
input_size = 512
hidden_size = 1024
output_size = 256
num_experts = 8
num_layers = 4
batch_size = 32
seq_len = 64

model = MoEModel(num_experts, input_size, hidden_size, output_size, num_layers)
input_tensor = torch.randn(batch_size, seq_len, input_size)

output = model(input_tensor)
print(f"Output shape: {output.shape}")

# Demonstrating expert parallel inference
def expert_parallel_inference(model, input_tensor, num_expert_parallel=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Split the batch across multiple GPUs
    split_size = input_tensor.size(0) // num_expert_parallel
    input_splits = torch.split(input_tensor, split_size)

    # Perform inference on each split
    outputs = []
    for i, split in enumerate(input_splits):
        with torch.cuda.device(i % torch.cuda.device_count()):
            outputs.append(model(split.to(f"cuda:{i % torch.cuda.device_count()}")))

    # Concatenate the results
    return torch.cat(outputs, dim=0)

# Run expert parallel inference
output = expert_parallel_inference(model, input_tensor, num_expert_parallel=4)
print(f"Expert parallel output shape: {output.shape}")
