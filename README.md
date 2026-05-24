# Small Language Model from Scratch

A complete implementation of a transformer-based language model built from scratch using PyTorch. This project features a proper production-ready folder structure with clean separation of concerns.

## 📁 Project Structure

```
src/                  # Core package code
  ├── model.py        # Transformer architecture
  ├── tokenizer.py    # Tokenization implementations
  ├── data.py         # Data loading and preprocessing
  ├── train.py        # Training pipeline
  ├── evaluate.py     # Inference and evaluation
  ├── config.py       # Configuration management
  ├── utils.py        # Utility functions
  └── __init__.py     # Package exports

scripts/              # Executable entry points
  └── main.py         # CLI interface

examples/             # Example scripts
  ├── train_example.py      # Training example
  └── inference_example.py  # Inference example

notebooks/            # Jupyter notebooks
tests/                # Unit tests
configs/              # Configuration files
checkpoints/          # Model checkpoints
outputs/              # Training outputs
logs/                 # TensorBoard logs
data/                 # Datasets
```

For detailed structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## ✨ Key Features

- **Transformer with RoPE**: Rotary position embeddings for better efficiency
- **SwiGLU FFN**: Enhanced feed-forward network
- **Multiple Tokenizers**: Character-level and BPE encoding
- **Advanced Sampling**: Top-k and top-p (nucleus) sampling
- **Mixed Precision**: FP16 training for faster convergence
- **Checkpointing**: Save and resume training
- **Dataset Support**: TinyStories and WikiText built-in
- **Interactive Generation**: Real-time text generation loop
- **Production-Ready**: Proper folder structure and best practices

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a development package
pip install -e .
```

### 2. View Model Info

```bash
python scripts/main.py info
```

### 3. Train the Model

```bash
# Default training
python scripts/main.py train

# Custom parameters
python scripts/main.py train --epochs 10 --batch-size 64 --lr 1e-4 --dataset tiny_stories
```

### 4. Generate Text

```bash
# Single generation
python scripts/main.py generate --prompt "Once upon a time" --length 200

# Interactive mode
python scripts/main.py generate --interactive

# With custom sampling
python scripts/main.py generate --prompt "Hello" --temperature 0.8 --top-k 50 --top-p 0.95
```

### 5. Tokenizer

```bash
# Build tokenizer
python scripts/main.py tokenizer --action build

# Test tokenizer
python scripts/main.py tokenizer --action test --test-text "Your text here"
```

## 📚 Usage Examples

### Train with Custom Config

```python
from src.config import Config, ModelConfig, TrainingConfig
from src.train import Trainer

config = Config(
    model=ModelConfig(hidden_size=256, num_layers=4),
    training=TrainingConfig(batch_size=32, num_epochs=5),
)

trainer = Trainer(config)
trainer.train()
```

### Generate Text from Code

```python
from src.evaluate import Generator
from src.config import config
from pathlib import Path

checkpoint = config.checkpoint_dir / "best_model.pt"
generator = Generator(checkpoint, config)

text = generator.generate(
    "Once upon a time",
    max_length=200,
    temperature=0.8,
    top_p=0.95
)
print(text)
```

### Use as Package

```python
from src.model import LanguageModel
from src.tokenizer import get_tokenizer
from src.config import config

# Create model
model = LanguageModel(
    vocab_size=10000,
    hidden_size=256,
    num_layers=4,
    num_heads=8
)

# Get tokenizer
tokenizer = get_tokenizer("char")
```

## 🔧 Configuration

Edit configuration in code or pass parameters to CLI:

```python
from src.config import Config, ModelConfig, TrainingConfig, DataConfig

config = Config(
    model=ModelConfig(
        vocab_size=10000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        max_seq_length=256,
    ),
    training=TrainingConfig(
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=5,
    ),
    data=DataConfig(
        dataset_name="tiny_stories",
        validation_split=0.05,
    ),
)
```

## 📊 Model Architecture

### Transformer Block
- **Multi-head Self-Attention**: 8 heads with rotary position embeddings
- **Feed-Forward Network**: SwiGLU activation
- **Layer Normalization**: Pre-norm residual connections
- **Dropout**: For regularization

### Positional Encoding
- **Rotary Position Embeddings (RoPE)**: More efficient than traditional sinusoidal embeddings

### Decoding Strategies
- **Top-k Sampling**: Sample from top-k most likely tokens
- **Top-p (Nucleus) Sampling**: Sample from smallest set of tokens with cumulative probability ≥ p
- **Temperature**: Control randomness of sampling

## 🎯 Training Tips

1. **Start small**: Use smaller model and dataset for development
2. **Monitor loss**: Watch validation loss to avoid overfitting
3. **Learning rate**: Adjust based on your hardware and batch size
4. **Data**: More and better data = better model performance
5. **Sampling**: Adjust temperature and top-p for generation quality

## 📈 Monitoring Training

```bash
# In a separate terminal, visualize training with TensorBoard
tensorboard --logdir ./logs
```

Then open http://localhost:6006 in your browser.

## ⚙️ Advanced Usage

### Distributed Training (Multi-GPU)

```python
from src.config import Config
config = Config(use_distributed=True, world_size=4)
```

### Mixed Precision Training

```python
from src.config import TrainingConfig
config.training.use_mixed_precision = True
```

### Custom Dataset

```python
from src.data import TextDataset
from src.tokenizer import get_tokenizer

tokenizer = get_tokenizer("char")
dataset = TextDataset(
    texts=your_texts,
    tokenizer=tokenizer,
    max_seq_length=256,
)
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Run `pip install -r requirements.txt` |
| "CUDA out of memory" | Reduce `batch_size` or `max_seq_length` in config |
| Slow first run | First run downloads dataset (~1-2 GB) |
| Poor generation | Train for more epochs or use more data |
| Import errors | Make sure you're in the project root directory |

## 📁 File Descriptions

| File | Purpose |
|------|---------|
| `src/config.py` | Configuration classes and defaults |
| `src/model.py` | Transformer model implementation |
| `src/tokenizer.py` | Character and BPE tokenizers |
| `src/data.py` | Dataset loading and preprocessing |
| `src/train.py` | Training loop implementation |
| `src/evaluate.py` | Inference and evaluation |
| `src/utils.py` | Helper functions and utilities |
| `scripts/main.py` | CLI entry point |
| `examples/` | Example scripts |
| `setup.py` | Package installation |
| `requirements.txt` | Python dependencies |

## 🔄 Workflow

1. **Setup**: Install dependencies and review structure
2. **Configure**: Adjust model/training settings in `src/config.py`
3. **Train**: Run training script and monitor progress
4. **Evaluate**: Generate text and assess quality
5. **Iterate**: Fine-tune hyperparameters based on results
6. **Deploy**: Use the trained model for inference

## 🚀 Next Steps

- Experiment with different model sizes
- Try different datasets and preprocessing
- Implement custom training strategies
- Fine-tune on domain-specific data
- Add evaluation metrics
- Deploy to production

## 📖 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

## 📝 License

MIT

---

**Ready to build? Start with `python scripts/main.py info` to see your model!** 🎉
