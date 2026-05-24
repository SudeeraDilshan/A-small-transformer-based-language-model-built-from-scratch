# Small Language Model from Scratch

A complete implementation of a transformer-based language model built from scratch using PyTorch. This project features a proper production-ready folder structure with clean separation of concerns.

## 📁 Project Structure

```
src/
  └── slm/            # Core package code
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
# Small Language Model from Scratch

A minimal, self-contained transformer-based language model built from scratch using PyTorch. The repository is organized as a small reusable package and includes training, evaluation, tokenization, and example scripts.

This README highlights how to get started and where to find the main components. For a detailed developer-oriented layout, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Quick overview

- Core package: `src/slm/` (model, tokenizer, data, train, evaluate, utils)
- CLI entrypoint: `scripts/main.py` (train, generate, tokenizer, info)
- Examples: `examples/` (train and inference examples)
- Docs: `README.md`, `QUICKSTART.md`, `PROJECT_STRUCTURE.md`

## Installation

Activate your environment and install dependencies:

```bash
# Activate virtualenv (Windows PowerShell)
.\\.venv\Scripts\\Activate.ps1

# Install requirements
pip install -r requirements.txt

# (optional) Install package in editable mode
pip install -e .
```

## Common commands

- Show model and config info:

```bash
python scripts/main.py info
```

- Train the model (default config):

```bash
python scripts/main.py train
```

- Generate text:

```bash
python scripts/main.py generate --prompt "Once upon a time" --length 200
```

- Build/test tokenizer:

```bash
python scripts.main.py tokenizer --action build
python scripts.main.py tokenizer --action test --test-text "Hello world"
```

## Examples

Run the provided examples:

```bash
python examples/train_example.py
python examples/inference_example.py
```

## Notes and tips

- If you encounter "Module not found" errors, ensure you ran `pip install -r requirements.txt` or `pip install -e .` and that you run commands from the repository root.
- If CUDA memory is insufficient, reduce `src/slm/config.py` batch size or max sequence length.
- First dataset downloads may take time and use disk space; datasets are cached under `data/`.

## License

MIT
tokenizer = get_tokenizer("char")
```

## 🔧 Configuration

Edit configuration in code or pass parameters to CLI:

```python
from slm.config import Config, ModelConfig, TrainingConfig, DataConfig

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
from slm.config import Config
config = Config(use_distributed=True, world_size=4)
```

### Mixed Precision Training

```python
from slm.config import TrainingConfig
config.training.use_mixed_precision = True
```

### Custom Dataset

```python
from slm.data import TextDataset
from slm.tokenizer import get_tokenizer

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
| `src/slm/config.py` | Configuration classes and defaults |
| `src/slm/model.py` | Transformer model implementation |
| `src/slm/tokenizer.py` | Character and BPE tokenizers |
| `src/slm/data.py` | Dataset loading and preprocessing |
| `src/slm/train.py` | Training loop implementation |
| `src/slm/evaluate.py` | Inference and evaluation |
| `src/slm/utils.py` | Helper functions and utilities |
| `scripts/main.py` | CLI entry point |
| `examples/` | Example scripts |
| `setup.py` | Package installation |
| `requirements.txt` | Python dependencies |

## 🔄 Workflow

1. **Setup**: Install dependencies and review structure
2. **Configure**: Adjust model/training settings in `src/slm/config.py`
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
