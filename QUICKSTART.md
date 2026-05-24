# Quick Start Guide

## 1. Setup Environment

Your PyTorch is already installed in `.venv`. Activate it and install remaining dependencies:

```bash
# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate

# Install remaining packages
pip install -r requirements.txt
```

## 2. Project Structure

The project is now organized for production use:

```
src/              # All core code
scripts/          # Entry points
examples/         # Example scripts
notebooks/        # Jupyter notebooks
tests/            # Unit tests
configs/          # Config files
checkpoints/      # Model checkpoints (auto-created)
outputs/          # Outputs (auto-created)
logs/             # TensorBoard logs (auto-created)
data/             # Datasets (auto-created)
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details.

## 3. View Model Info

```bash
python scripts/main.py info
```

This shows your model configuration.

## 4. Train the Model

```bash
# Start training
python scripts/main.py train
```

Or with custom parameters:
```bash
python scripts/main.py train --epochs 5 --batch-size 32 --lr 1e-4
```

Training will:
- Automatically download the dataset (first time only)
- Create checkpoints directory
- Save progress periodically
- Show training/validation loss

## 5. Generate Text

Once trained, generate text:

```bash
# Single generation
python scripts/main.py generate --prompt "Once upon a time" --length 200

# Interactive mode
python scripts/main.py generate --interactive
```

## 6. Monitor Training (Optional)

In a new terminal:
```bash
tensorboard --logdir ./logs
```

Open http://localhost:6006 in your browser.

## 7. Use Examples

Run example scripts:

```bash
# Training example
python examples/train_example.py

# Inference example
python examples/inference_example.py
```

## 8. Use as Package

You can now import from `src`:

```python
from src.model import LanguageModel
from src.config import config
from src.train import Trainer

trainer = Trainer(config)
trainer.train()
```

## Commands Reference

| Command | Purpose |
|---------|---------|
| `python scripts/main.py info` | Show model configuration |
| `python scripts/main.py train` | Train the model |
| `python scripts/main.py generate` | Generate text |
| `python scripts/main.py tokenizer` | Build/test tokenizer |

## Configuration

Edit `src/config.py` for quick changes:

```python
# Smaller model (faster):
hidden_size = 128
num_layers = 2

# Larger model (better quality):
hidden_size = 512
num_layers = 6

# Different dataset:
dataset_name = "wikitext"  # or "tiny_stories"
```

## Troubleshooting

**"Module not found" error:**
```bash
pip install -r requirements.txt
```

**"CUDA out of memory":**
- Reduce batch_size in src/config.py
- Reduce max_seq_length

**Slow first run:**
- First download takes time (~1-2 GB)
- Subsequent runs are much faster

## Next Steps

1. **Train**: `python scripts/main.py train`
2. **Generate**: `python scripts/main.py generate --interactive`
3. **Experiment**: Try different model sizes
4. **Optimize**: Adjust hyperparameters based on results
5. **Deploy**: Use trained model in production

## File Locations

- **Main entry point**: `scripts/main.py`
- **Configuration**: `src/config.py`
- **Model code**: `src/model.py`
- **Training**: `src/train.py`
- **Inference**: `src/evaluate.py`
- **Examples**: `examples/`

## Folder Structure

```
project/
├── src/              ← Core package
│   ├── model.py
│   ├── config.py
│   ├── train.py
│   └── ...
├── scripts/          ← Executables
│   └── main.py
├── examples/         ← Examples
├── checkpoints/      ← Auto-created on first train
├── outputs/          ← Auto-created on first train
└── logs/             ← Auto-created on first train
```

---

**You're all set! Start with:** `python scripts/main.py info` 🎉

See [README.md](README.md) for full documentation and [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for structure details.
