"""Evaluation and inference script"""
import torch
import logging
from pathlib import Path

from .config import config
from .model import LanguageModel
from .tokenizer import get_tokenizer
from .utils import load_checkpoint, generate_text, compute_perplexity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator:
    """Text generation interface"""
    
    def __init__(self, checkpoint_path: Path, cfg):
        self.config = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        tokenizer_path = cfg.output_dir / "tokenizer.json"
        self.tokenizer = get_tokenizer("char", cfg.model.vocab_size)
        
        if tokenizer_path.exists():
            self.tokenizer.load(tokenizer_path)
        else:
            logger.warning("Tokenizer not found")
            return
        
        # Create model
        self.model = LanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            ffn_hidden_size=cfg.model.ffn_hidden_size,
            max_seq_length=cfg.model.max_seq_length,
            dropout=0.0,  # No dropout during inference
            attention_dropout=0.0
        ).to(self.device)
        
        # Load checkpoint
        if checkpoint_path.exists():
            load_checkpoint(self.model, None, checkpoint_path, str(self.device))
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found")
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.95):
        """Generate text from prompt"""
        
        # Encode prompt
        prompt_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids.to(self.device),
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode
        text = self.tokenizer.decode(generated[0].cpu().tolist())
        
        return text


def evaluate_perplexity(model: LanguageModel, val_loader, criterion, device: str = "cpu"):
    """Evaluate perplexity on validation set"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples
    perplexity = compute_perplexity(avg_loss)
    
    logger.info(f"Validation loss: {avg_loss:.4f}")
    logger.info(f"Perplexity: {perplexity:.2f}")
    
    return avg_loss, perplexity


def interactive_generation():
    """Interactive text generation loop"""
    
    checkpoint_path = config.checkpoint_dir / "best_model.pt"
    generator = Generator(checkpoint_path, config)
    
    logger.info("Starting interactive text generation (type 'quit' to exit)")
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            
            if prompt.lower() == "quit":
                break
            
            print("\nGenerating...")
            text = generator.generate(
                prompt,
                max_length=200,
                temperature=0.8,
                top_k=50,
                top_p=0.95
            )
            
            print(f"\nGenerated text:\n{text}\n")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
