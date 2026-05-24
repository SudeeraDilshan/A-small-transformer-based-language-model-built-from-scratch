"""Example inference script"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.evaluate import Generator

# Generate text from checkpoint
checkpoint_path = config.checkpoint_dir / "best_model.pt"
generator = Generator(checkpoint_path, config)

prompts = [
    "Once upon a time",
    "The quick brown fox",
    "In a galaxy far away",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    text = generator.generate(
        prompt,
        max_length=150,
        temperature=0.8,
        top_k=40,
        top_p=0.9
    )
    print(f"Generated:\n{text}\n")
    print("-" * 80)
