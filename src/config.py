"""Configuration for the small language model"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    """Model hyperparameters"""
    vocab_size: int = 10000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    max_seq_length: int = 256
    dropout: float = 0.1
    attention_dropout: float = 0.1
    ffn_hidden_size: int = 1024
    layer_norm_eps: float = 1e-6
    

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 5
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    use_mixed_precision: bool = True
    num_workers: int = 4
    

@dataclass
class DataConfig:
    """Data loading parameters"""
    dataset_name: str = "tiny_stories"  # or "wikitext" or "openwebtext"
    dataset_split: str = "train"
    validation_split: float = 0.05
    max_samples: int = None  # None for all samples
    cache_dir: Path = Path("./data")
    

@dataclass
class Config:
    """Complete configuration"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    
    # Paths
    output_dir: Path = Path("./outputs")
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")
    
    # Distributed training
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Misc
    seed: int = 42
    device: str = "cuda"
    use_wandb: bool = False
    
    def create_dirs(self):
        """Create necessary directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
config = Config()
