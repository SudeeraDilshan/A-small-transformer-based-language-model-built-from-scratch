"""Example training script"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.config import Config, ModelConfig, TrainingConfig, DataConfig
from slm.train import Trainer

# Create custom config for quick experimentation
config = Config(
    model=ModelConfig(
        vocab_size=5000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_length=256,
    ),
    training=TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=3,
    ),
    data=DataConfig(
        dataset_name="tiny_stories",
        max_samples=1000,  # Limit for quick training
    ),
    device="cuda",
)

if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train()
