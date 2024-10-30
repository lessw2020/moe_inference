"""Configuration system for distributed model serving."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Final, ClassVar
import torch
from transformers import AutoConfig

from distserve.utils import GB


class TokenizerMode(Enum):
    """Tokenizer processing modes."""
    AUTO = "auto"
    SLOW = "slow"


class ModelDType(Enum):
    """Supported model data types."""
    FP16 = "fp16"
    FP32 = "fp32"

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get corresponding PyTorch dtype."""
        return {
            ModelDType.FP16: torch.half,
            ModelDType.FP32: torch.float32,
        }[self]
    
    @property
    def size_in_bytes(self) -> int:
        """Get size in bytes for this dtype."""
        return {
            ModelDType.FP16: 2,
            ModelDType.FP32: 4,
        }[self]


class SchedulerPolicy(Enum):
    """Available scheduling policies."""
    FCFS = "fcfs"
    SRPT = "srpt" 
    MLFQ = "mlfq"
    SJ_MLFQ = "sj-mlfq"


@dataclass
class CacheConfig:
    """Configuration for the key-value cache."""
    block_size: int
    max_num_blocks_per_req: int
    gpu_memory_utilization: float = 0.9
    cpu_swap_space_gb: int = 0

    def __post_init__(self):
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f"GPU memory utilization must be in (0, 1], "
                f"got {self.gpu_memory_utilization}")
        self.cpu_swap_space = self.cpu_swap_space_gb * GB


@dataclass
class ParallelConfig:
    """Configuration for distributed execution."""
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    pipeline_parallel_size: int = 1
    pipeline_parallel_rank: int = 0

    def __post_init__(self):
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size
        self.use_parallel = self.world_size > 1

    def to_list(self) -> List[int]:
        """Convert config to list format."""
        return [
            self.tensor_parallel_size,
            self.tensor_parallel_rank,
            self.pipeline_parallel_size,
            self.pipeline_parallel_rank,
        ]

    def is_last_stage(self) -> bool:
        """Check if this is the last pipeline stage."""
        return self.pipeline_parallel_rank == self.pipeline_parallel_size - 1


@dataclass
class DisaggParallelConfig:
    """Configuration for disaggregated execution."""
    context: ParallelConfig
    decoding: ParallelConfig

    def get_num_workers(self) -> int:
        """Get total number of required workers."""
        return self.context.world_size + self.decoding.world_size


@dataclass
class SchedulerConfigBase:
    """Base configuration for schedulers."""
    policy: SchedulerPolicy
    max_batch_size: int
    max_tokens_per_batch: int

    def __post_init__(self):
        if isinstance(self.policy, str):
            try:
                self.policy = SchedulerPolicy(self.policy)
            except ValueError:
                raise ValueError(f"Unsupported policy: {self.policy}")


@dataclass
class ContextStageSchedConfig(SchedulerConfigBase):
    """Configuration for context stage scheduler."""
    parallel_config: Optional[ParallelConfig] = None

    def __post_init__(self):
        super().__post_init__()
        if self.policy != SchedulerPolicy.FCFS:
            raise ValueError(
                f"Context stage only supports FCFS policy, got {self.policy}")


@dataclass
class DecodingStageSchedConfig(SchedulerConfigBase):
    """Configuration for decoding stage scheduler."""
    model_name: Optional[str] = None
    waiting_block_prop_threshold: float = 0.05


@dataclass
class ModelConfig:
    """Configuration for the model.
    
    Handles model configuration including architecture details,
    tokenizer settings, and hardware-specific parameters.
    """
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: TokenizerMode = TokenizerMode.AUTO
    trust_remote_code: bool = False
    dtype: ModelDType = ModelDType.FP16
    seed: int = 1
    use_dummy_weights: bool = False

    # Constants for model architecture handling
    POSITION_EMBEDDING_KEYS: ClassVar[List[str]] = [
        "max_position_embeddings",  # OPT
        "n_positions",              # GPT-2
        "max_seq_len",             # MPT
        "max_sequence_length",     # Others
        "max_seq_length",
        "seq_len",
    ]

    def __post_init__(self):
        """Initialize and validate configuration."""
        self.tokenizer = self.tokenizer or self.model
        if isinstance(self.tokenizer_mode, str):
            self.tokenizer_mode = TokenizerMode(self.tokenizer_mode)
        if isinstance(self.dtype, str):
            self.dtype = ModelDType(self.dtype)
        self.hf_config = self._load_hf_config()

    def _load_hf_config(self):
        """Load HuggingFace model configuration."""
        try:
            return AutoConfig.from_pretrained(
                self.model, trust_remote_code=self.trust_remote_code)
        except Exception as e:
            raise ValueError(
                f"Failed to load model config for {self.model}: {str(e)}")

    @property
    def hidden_size(self) -> int:
        """Get model hidden dimension size."""
        return self.hf_config.hidden_size

    @property
    def head_size(self) -> int:
        """Get attention head dimension size."""
        return self.hidden_size // self.hf_config.num_attention_heads

    @property
    def ffn_intermediate_size(self) -> int:
        """Get FFN intermediate dimension size."""
        return self.hf_config.intermediate_size

    def get_attention_heads(self, parallel_config: Optional[ParallelConfig] = None) -> Dict[str, int]:
        """Get number of attention heads for different components."""
        parallel_config = parallel_config or ParallelConfig()
        tp_size = parallel_config.tensor_parallel_size

        # Handle query heads
        q_heads = self.hf_config.num_attention_heads // tp_size

        # Handle key/value heads
        if self._is_multi_query_attention():
            kv_heads = 1
        elif hasattr(self.hf_config, "n_head_kv"):
            kv_heads = self.hf_config.n_head_kv // tp_size
        elif hasattr(self.hf_config, "num_key_value_heads"):
            kv_heads = self.hf_config.num_key_value_heads // tp_size
        else:
            kv_heads = q_heads

        if not all(h > 0 for h in [q_heads, kv_heads]):
            raise ValueError(
                f"Invalid head configuration: q_heads={q_heads}, kv_heads={kv_heads}")

        return {"query": q_heads, "kv": kv_heads}

    def _is_multi_query_attention(self) -> bool:
        """Check if model uses multi-query attention."""
        is_falcon = self.hf_config.model_type == "falcon"
        new_decoder = getattr(self.hf_config, "new_decoder_architecture", False)
        multi_query = getattr(self.hf_config, "multi_query", False)
        return not (is_falcon and new_decoder) and multi_query

    def get_max_sequence_length(self) -> int:
        """Get maximum supported sequence length."""
        for key in self.POSITION_EMBEDDING_KEYS:
            if hasattr(self.hf_config, key):
                return getattr(self.hf_config, key)
        return float("inf")

    def get_layers_per_stage(self, parallel_config: Optional[ParallelConfig] = None) -> int:
        """Get number of layers per pipeline stage."""
        parallel_config = parallel_config or ParallelConfig()
        total_layers = self.hf_config.num_hidden_layers
        pp_size = parallel_config.pipeline_parallel_size
        
        if total_layers % pp_size != 0:
            raise ValueError(
                f"Number of layers ({total_layers}) not divisible by pipeline "
                f"parallel size ({pp_size})")
        
        return total_layers // pp_size

    def calculate_model_size_bytes(self, parallel_config: Optional[ParallelConfig] = None) -> int:
        """Calculate model size in bytes."""
        parallel_config = parallel_config or ParallelConfig()
        tp_size = parallel_config.tensor_parallel_size
        
        # Embedding layers
        params = (
            self.hf_config.vocab_size * self.hidden_size +  # vocab embeddings
            self.get_max_sequence_length() * self.hidden_size  # position embeddings
        )
        
        # Attention layers
        params += (
            4 * self.get_layers_per_stage(parallel_config) * 
            (self.hidden_size ** 2) / tp_size
        )
        
        # FFN layers
        params += (
            8 * self.get_layers_per_stage(parallel_config) * 
            (self.hidden_size ** 2) / tp_size
        )
        
        # Biases
        params += (
            5 * self.get_layers_per_stage(parallel_config) * 
            self.hidden_size
        )
        
        return int(params * self.dtype.size_in_bytes)
