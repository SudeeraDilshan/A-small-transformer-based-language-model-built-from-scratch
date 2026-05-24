"""Main entry point for the project"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config import config
from src.train import Trainer
from src.evaluate import Generator, interactive_generation
from src.tokenizer import get_tokenizer
from src.data import load_dataset
from src.model import LanguageModel
from src.utils import count_parameters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Small Language Model Training and Inference")
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--dataset", type=str, default="tiny_stories", 
                            help="Dataset name (tiny_stories, wikitext)")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--prompt", type=str, default="The quick brown",
                          help="Text prompt")
    gen_parser.add_argument("--checkpoint", type=str, default=None,
                          help="Checkpoint path")
    gen_parser.add_argument("--length", type=int, default=200,
                          help="Max length to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8,
                          help="Sampling temperature")
    gen_parser.add_argument("--top-k", type=int, default=50,
                          help="Top-k sampling")
    gen_parser.add_argument("--top-p", type=float, default=0.95,
                          help="Top-p (nucleus) sampling")
    gen_parser.add_argument("--interactive", action="store_true",
                          help="Interactive mode")
    
    # Build tokenizer command
    tok_parser = subparsers.add_parser("tokenizer", help="Build/test tokenizer")
    tok_parser.add_argument("--action", choices=["build", "test"], default="build")
    tok_parser.add_argument("--test-text", type=str, default="Hello, world!",
                          help="Text to test tokenizer on")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Print model info")
    
    args = parser.parse_args()
    
    if args.command == "train":
        logger.info("Starting training...")
        
        # Update config
        config.training.num_epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.lr
        config.data.dataset_name = args.dataset
        
        # Train
        trainer = Trainer(config)
        trainer.train()
    
    elif args.command == "generate":
        logger.info("Starting text generation...")
        
        if args.interactive:
            interactive_generation()
        else:
            checkpoint_path = Path(args.checkpoint) if args.checkpoint \
                            else config.checkpoint_dir / "best_model.pt"
            
            generator = Generator(checkpoint_path, config)
            
            text = generator.generate(
                args.prompt,
                max_length=args.length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            print(f"\nPrompt: {args.prompt}")
            print(f"\nGenerated text:\n{text}\n")
    
    elif args.command == "tokenizer":
        logger.info("Tokenizer mode")
        
        if args.action == "build":
            logger.info("Building tokenizer...")
            train_texts, _ = load_dataset(config.data.dataset_name,
                                         config.data.cache_dir,
                                         max_samples=100)
            
            if train_texts:
                tokenizer = get_tokenizer("char", config.model.vocab_size)
                tokenizer.build_vocab(train_texts)
                
                tokenizer_path = config.output_dir / "tokenizer.json"
                tokenizer.save(tokenizer_path)
                
                logger.info(f"Tokenizer saved to {tokenizer_path}")
                logger.info(f"Vocab size: {tokenizer.vocab_size}")
        
        elif args.action == "test":
            logger.info(f"Testing tokenizer on: '{args.test_text}'")
            
            tokenizer_path = config.output_dir / "tokenizer.json"
            if not tokenizer_path.exists():
                logger.error("Tokenizer not found. Build it first with: python main.py tokenizer")
                return
            
            tokenizer = get_tokenizer("char", config.model.vocab_size)
            tokenizer.load(tokenizer_path)
            
            tokens = tokenizer.encode(args.test_text)
            decoded = tokenizer.decode(tokens)
            
            logger.info(f"Tokens: {tokens}")
            logger.info(f"Decoded: {decoded}")
    
    elif args.command == "info":
        logger.info("\n=== Model Information ===")
        logger.info(f"Vocab size: {config.model.vocab_size}")
        logger.info(f"Hidden size: {config.model.hidden_size}")
        logger.info(f"Num layers: {config.model.num_layers}")
        logger.info(f"Num heads: {config.model.num_heads}")
        logger.info(f"Max sequence length: {config.model.max_seq_length}")
        
        model = LanguageModel(
            vocab_size=config.model.vocab_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            ffn_hidden_size=config.model.ffn_hidden_size,
            max_seq_length=config.model.max_seq_length
        )
        
        total_params = count_parameters(model)
        logger.info(f"Total parameters: {total_params:,}")
        
        logger.info("\n=== Training Configuration ===")
        logger.info(f"Batch size: {config.training.batch_size}")
        logger.info(f"Learning rate: {config.training.learning_rate}")
        logger.info(f"Num epochs: {config.training.num_epochs}")
        logger.info(f"Device: {config.device}")
        
        logger.info("\n=== Data Configuration ===")
        logger.info(f"Dataset: {config.data.dataset_name}")
        logger.info(f"Validation split: {config.data.validation_split}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
