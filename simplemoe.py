import logging
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_logger():
    logger = logging.getLogger("MoE_Training")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler("moe_training.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class ExpertUsageTracker:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.reset_stats()

    def reset_stats(self):
        self.expert_counts = defaultdict(int)
        self.total_samples = 0
        self.routing_distribution = torch.zeros(self.num_experts)
        self.token_expert_history = []  # Track token assignments per iteration
        self.capacity_factor = torch.zeros(self.num_experts)  # Track expert load

    def update(self, expert_indices, routing_probs, batch_idx):
        # Track which tokens went to which experts
        expert_assignment = defaultdict(list)
        for token_idx, expert_idx in enumerate(expert_indices.cpu().numpy()):
            expert_assignment[int(expert_idx)].append(token_idx)

        # Calculate capacity utilization
        unique_experts, counts = torch.unique(expert_indices, return_counts=True)
        batch_size = len(expert_indices)
        capacity = defaultdict(float)
        for expert, count in zip(unique_experts.cpu().numpy(), counts.cpu().numpy()):
            capacity[expert] = (count / batch_size) * 100
            self.expert_counts[expert] += count
            self.capacity_factor[expert] += count

        # Store this iteration's routing pattern
        self.token_expert_history.append(
            {
                "batch_idx": batch_idx,
                "expert_assignments": dict(expert_assignment),
                "expert_capacity": dict(capacity),
            }
        )

        self.total_samples += len(expert_indices)
        self.routing_distribution += routing_probs.sum(0).detach().cpu()

    def get_stats(self):
        usage_percentages = {
            expert: (count / self.total_samples) * 100
            for expert, count in self.expert_counts.items()
        }
        routing_dist = (
            (self.routing_distribution / self.routing_distribution.sum())
            .detach()
            .numpy()
        )

        # Calculate average capacity factor per expert
        avg_capacity = (self.capacity_factor / len(self.token_expert_history)).tolist()

        return {
            "expert_usage_percentages": usage_percentages,
            "routing_distribution": routing_dist,
            "total_samples": self.total_samples,
            "token_routing_history": self.token_expert_history,
            "average_capacity_factor": avg_capacity,
        }


def cleanup():
    dist.destroy_process_group()

    # Clean up logging handlers if we're rank 0
    try:
        logger = logging.getLogger("MoE_Training")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    except:
        pass


class DistributedMoE(nn.Module):
    def __init__(self, num_experts, input_size, hidden_size, output_size, world_size):
        super().__init__()
        self.num_experts = num_experts
        self.world_size = world_size
        self.input_size = input_size
        self.output_size = output_size

        self.router = nn.Linear(input_size, num_experts)

        experts_per_gpu = num_experts // world_size
        start_idx = dist.get_rank() * experts_per_gpu
        end_idx = start_idx + experts_per_gpu

        self.local_experts = nn.ModuleList(
            [
                Expert(input_size, hidden_size, output_size)
                for _ in range(start_idx, end_idx)
            ]
        )

        self.aux_loss = None
        self.usage_tracker = ExpertUsageTracker(num_experts)
        self.logger = (
            logging.getLogger("MoE_Training") if dist.get_rank() == 0 else None
        )
        self.current_batch = 0

    def _compute_load_balancing_loss(
        self, router_probs
    ):  # Method is defined with underscore
        expert_loads = router_probs.mean(dim=0)
        target_loads = torch.ones_like(expert_loads) / self.num_experts
        aux_loss = torch.mean(torch.sum(expert_loads * expert_loads)) * self.num_experts
        aux_loss += (
            torch.mean(torch.sum(target_loads * target_loads)) * self.num_experts
        )
        return aux_loss

    def forward(self, x):
        batch_size = x.shape[0]
        start_time = time.time()

        # Routing
        routing_logits = self.router(x)
        routing_probs = torch.softmax(routing_logits, dim=-1)
        self.aux_loss = self._compute_load_balancing_loss(routing_probs)
        expert_indices = torch.argmax(routing_probs, dim=-1)

        if self.logger:
            self.usage_tracker.update(expert_indices, routing_probs, self.current_batch)
            self.current_batch += 1  # Increment batch counter
            router_time = time.time() - start_time
            self.logger.debug(f"Router processing time: {router_time:.4f}s")

        # Prepare for all-to-all
        dispatch_start = time.time()
        chunks_per_gpu = batch_size // self.world_size
        input_chunks = list(x.chunk(self.world_size))

        max_chunk_size = max(chunk.size(0) for chunk in input_chunks)
        for i in range(len(input_chunks)):
            if input_chunks[i].size(0) < max_chunk_size:
                pad_size = max_chunk_size - input_chunks[i].size(0)
                pad = torch.zeros(pad_size, self.input_size, device=x.device)
                input_chunks[i] = torch.cat([input_chunks[i], pad])

        output_chunks = [torch.zeros_like(chunk) for chunk in input_chunks]

        if self.logger:
            prep_time = time.time() - dispatch_start
            self.logger.debug(f"Dispatch preparation time: {prep_time:.4f}s")

        # All-to-all communication
        comm_start = time.time()
        dist.all_to_all(output_chunks, input_chunks)

        if self.logger:
            comm_time = time.time() - comm_start
            self.logger.debug(f"All-to-all communication time: {comm_time:.4f}s")

        # Expert processing
        expert_start = time.time()
        received_inputs = torch.cat(output_chunks)
        combined_output = torch.zeros_like(received_inputs)

        experts_per_gpu = self.num_experts // self.world_size
        local_start_idx = dist.get_rank() * experts_per_gpu

        for i, expert in enumerate(self.local_experts):
            expert_idx = local_start_idx + i
            mask = expert_indices == expert_idx

            if not mask.any():
                dummy_input = torch.zeros(
                    (1,) + received_inputs.shape[1:], device=received_inputs.device
                )
                dummy_output = expert(dummy_input)
                combined_output += dummy_output.mean() * 1e-8
            else:
                expert_input = received_inputs[mask]
                expert_output = expert(expert_input)
                combined_output[mask] = expert_output

        if self.logger:
            expert_time = time.time() - expert_start
            self.logger.debug(f"Expert processing time: {expert_time:.4f}s")

        # Return communication
        return_start = time.time()
        output_chunks_to_send = list(combined_output.chunk(self.world_size))

        for i in range(len(output_chunks_to_send)):
            if output_chunks_to_send[i].size(0) < max_chunk_size:
                pad_size = max_chunk_size - output_chunks_to_send[i].size(0)
                pad = torch.zeros(pad_size, self.output_size, device=x.device)
                output_chunks_to_send[i] = torch.cat([output_chunks_to_send[i], pad])

        received_chunks = [torch.zeros_like(chunk) for chunk in output_chunks_to_send]
        dist.all_to_all(received_chunks, output_chunks_to_send)

        final_output = torch.cat(received_chunks)[:batch_size]

        if self.logger:
            return_time = time.time() - return_start
            total_time = time.time() - start_time
            self.logger.debug(f"Return communication time: {return_time:.4f}s")
            self.logger.debug(f"Total forward pass time: {total_time:.4f}s")

        return final_output


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:
        setup_logger()


def run_training(rank, world_size):
    setup(rank, world_size)
    logger = logging.getLogger("MoE_Training") if rank == 0 else None

    if logger:
        logger.info(f"Starting training with {world_size} GPUs")

    model = DistributedMoE(
        num_experts=8,
        input_size=256,
        hidden_size=512,
        output_size=256,
        world_size=world_size,
    ).to(rank)

    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float("inf")
    running_loss = 0.0

    for epoch in range(10):
        epoch_start = time.time()
        if logger:
            logger.info(f"\nStarting epoch {epoch}")
            model.module.usage_tracker.reset_stats()

        for batch_idx in range(10):
            inputs = torch.randn(32, 256, device=rank)
            targets = torch.randn(32, 256, device=rank)

            optimizer.zero_grad()

            batch_start = time.time()
            outputs = model(inputs)

            main_loss = F.mse_loss(outputs, targets)
            aux_loss = model.module.aux_loss
            loss = main_loss + 0.01 * aux_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if logger and batch_idx % 2 == 0:  # Increased logging frequency
                batch_time = time.time() - batch_start
                logger.info(f"\nEpoch {epoch}, Batch {batch_idx}")
                logger.info(
                    f"Main Loss: {main_loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}"
                )
                logger.info(f"Batch processing time: {batch_time:.4f}s")

                # Log token-expert routing for this batch
                stats = model.module.usage_tracker.get_stats()
                current_batch = stats["token_routing_history"][-1]

                logger.info("\nToken-Expert Routing Summary for current batch:")
                logger.info("Expert | Token Count | Tokens | Capacity")
                logger.info("-" * 50)
                for expert in range(model.module.num_experts):
                    tokens = current_batch["expert_assignments"].get(expert, [])
                    capacity = current_batch["expert_capacity"].get(expert, 0)
                    logger.info(
                        f"E{expert:2d}   | {len(tokens):11d} | {tokens[:5]}{'...' if len(tokens)>5 else ''} | {capacity:.1f}%"
                    )

                # Show load balancing metrics
                logger.info("\nLoad Balancing Metrics:")
                for expert in range(model.module.num_experts):
                    avg_capacity = stats["average_capacity_factor"][expert]
                    logger.info(f"Expert {expert}: Avg Capacity: {avg_capacity:.1f}%")

        if logger:
            epoch_time = time.time() - epoch_start
            avg_loss = running_loss / (batch_idx + 1)
            logger.info(f"\nEpoch {epoch} Summary:")
            logger.info(f"Time: {epoch_time:.2f}s, Average loss: {avg_loss:.4f}")

            # Log overall epoch expert usage statistics
            stats = model.module.usage_tracker.get_stats()
            logger.info("\nEpoch Expert Usage Statistics:")
            logger.info("Expert | Usage % | Routing Weight")
            logger.info("-" * 40)
            for expert in range(model.module.num_experts):
                usage = stats["expert_usage_percentages"].get(expert, 0)
                routing_weight = stats["routing_distribution"][expert]
                logger.info(f"E{expert:2d}   | {usage:6.2f}% | {routing_weight:.4f}")

            running_loss = 0.0

    if logger:
        logger.info("\nTraining completed")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)
