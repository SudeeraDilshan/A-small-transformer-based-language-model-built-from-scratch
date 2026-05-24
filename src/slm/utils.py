"""Utility functions"""
import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, 
                   path: Path):
    """Save model checkpoint"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(model: nn.Module, optimizer, path: Path, device: str = "cpu"):
    """Load model checkpoint"""
    if not path.exists():
        logger.warning(f"Checkpoint {path} not found")
        return 0, float('inf')
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    logger.info(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    
    return epoch, loss


def count_parameters(model: nn.Module):
    """Count total parameters"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WarmupScheduler:
    """Learning rate scheduler with warmup"""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warmup_steps
        else:
            # Linear decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = max(0, 1 - progress)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale


def generate_text(model: nn.Module, prompt_ids: torch.Tensor, tokenizer,
                 max_length: int = 100, temperature: float = 1.0, 
                 top_k: int = 50, top_p: float = 0.95, device: str = "cpu"):
    """Generate text from model"""
    model.eval()
    
    with torch.no_grad():
        generated = model.generate(
            prompt_ids.to(device),
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode
    text = tokenizer.decode(generated[0].cpu().tolist())
    
    return text


def get_device():
    """Get device (cuda if available, else cpu)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_attention_mask(seq_length: int, device: str = "cpu"):
    """Create causal attention mask"""
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device) * float('-inf'), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def compute_perplexity(loss: float):
    """Compute perplexity from loss"""
    return 2.71828 ** loss
