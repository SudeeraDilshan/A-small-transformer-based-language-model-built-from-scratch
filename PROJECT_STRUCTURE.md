# Project Structure

Small Language Model - Proper folder organization for a production-ready project.

## Directory Structure

```
.
├── src/                          # Source code root
│   └── slm/                      # Main package
│       ├── __init__.py              # Package initialization
│       ├── config.py                # Configuration classes
│       ├── model.py                 # Transformer model architecture
│       ├── tokenizer.py             # Tokenizer implementations
│       ├── data.py                  # Data loading and preprocessing
│       ├── train.py                 # Training loop
│       ├── evaluate.py              # Evaluation and inference
│       └── utils.py                 # Utility functions
│
├── scripts/                      # Executable scripts
│   └── main.py                  # CLI entry point
│
├── examples/                     # Example scripts
│   ├── train_example.py         # Training example
│   └── inference_example.py     # Inference example
│
├── notebooks/                    # Jupyter notebooks
│   └── (add your notebooks here)
│
├── tests/                        # Unit tests
│   └── (add your tests here)
│
├── configs/                      # Configuration files (YAML/JSON)
│   └── (add custom configs here)
│
├── checkpoints/                  # Model checkpoints (git ignored)
├── outputs/                      # Output directory (git ignored)
├── logs/                         # TensorBoard logs (git ignored)
├── data/                         # Dataset directory (git ignored)
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup script
├── README.md                     # Main documentation
├── QUICKSTART.md                 # Quick start guide
├── .gitignore                    # Git ignore file
└── PROJECT_STRUCTURE.md          # This file
```

## Directory Purpose

### `src/slm/`
Core package code. All reusable modules go here.
- **config.py**: Centralized configuration management
- **model.py**: Neural network architecture
- **tokenizer.py**: Text tokenization implementations
- **data.py**: Dataset loading and preprocessing
- **train.py**: Training pipeline with Trainer class
- **evaluate.py**: Inference and evaluation
- **utils.py**: Helper functions and utilities

### `scripts/`
Executable entry points for the project.
- **main.py**: CLI interface with all commands (train, generate, etc.)

### `examples/`
Example scripts showing how to use the library.
- **train_example.py**: Custom training example
- **inference_example.py**: Text generation example

### `notebooks/`
Jupyter notebooks for exploration and analysis.

### `tests/`
Unit tests for the codebase (pytest recommended).

### `configs/`
Configuration files for different experiments (YAML/JSON).

### `checkpoints/`, `outputs/`, `logs/`, `data/`
Runtime directories (created automatically, git ignored).

## Usage

### Install for development
```bash
pip install -e .
pip install -r requirements.txt
```

### Run from anywhere
```bash
# Using the CLI
python scripts/main.py train
python scripts/main.py generate --prompt "Hello"

# Or use examples
python examples/train_example.py
python examples/inference_example.py
```

### Use as a package in your code
```python
from slm.model import LanguageModel
from slm.config import config
from slm.train import Trainer

trainer = Trainer(config)
trainer.train()
```

## Benefits of This Structure

1. **Modularity**: Clean separation of concerns
2. **Reusability**: Easy to import and use components
3. **Scalability**: Ready for package distribution
4. **Testing**: Easy to add unit tests
5. **Documentation**: Clear organization aids understanding
6. **Collaboration**: Standard structure familiar to developers
7. **Git**: Proper .gitignore keeps repo clean
