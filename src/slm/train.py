"""Training script for language model"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm

from .config import Config, config
from .model import LanguageModel
from .data import load_dataset, get_data_loaders
from .tokenizer import get_tokenizer
from .utils import AverageMeter, save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for language model"""
    
    def __init__(self, cfg: Config):
        self.config = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.config.create_dirs()
        
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        self.set_seed(cfg.seed)
        
        # Create or load tokenizer
        self.tokenizer = self._get_tokenizer()
        
        # Create model
        self.model = LanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            ffn_hidden_size=cfg.model.ffn_hidden_size,
            max_seq_length=cfg.model.max_seq_length,
            dropout=cfg.model.dropout,
            attention_dropout=cfg.model.attention_dropout,
            layer_norm_eps=cfg.model.layer_norm_eps
        ).to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Mixed precision
        self.scaler = None
        if cfg.training.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Logging
        self.writer = SummaryWriter(log_dir=str(cfg.log_dir))
        self.global_step = 0
        
    def set_seed(self, seed: int):
        """Set random seeds"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _get_tokenizer(self):
        """Get or create tokenizer"""
        tokenizer_path = self.config.output_dir / "tokenizer.json"
        
        if tokenizer_path.exists():
            logger.info("Loading tokenizer from checkpoint...")
            tokenizer = get_tokenizer("char", self.config.model.vocab_size)
            tokenizer.load(tokenizer_path)
            return tokenizer
        
        # Load data and build tokenizer
        logger.info(f"Loading {self.config.data.dataset_name} dataset...")
        train_texts, _ = load_dataset(
            self.config.data.dataset_name,
            self.config.data.cache_dir,
            self.config.data.max_samples
        )
        
        if train_texts is None:
            logger.warning("Failed to load dataset. Using dummy data.")
            train_texts = ["Hello world. This is a test."]
        
        logger.info("Building tokenizer...")
        tokenizer = get_tokenizer("char", self.config.model.vocab_size)
        tokenizer.build_vocab(train_texts)
        
        # Save tokenizer
        tokenizer.save(tokenizer_path)
        
        return tokenizer
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        loss_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, _ = self.model(input_ids)
                    loss = self.criterion(logits.view(-1, self.model.vocab_size), 
                                        target_ids.view(-1))
                
                # Backward pass
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   self.config.training.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits, _ = self.model(input_ids)
                loss = self.criterion(logits.view(-1, self.model.vocab_size), 
                                    target_ids.view(-1))
                
                loss.backward()
                if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   self.config.training.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            loss_meter.update(loss.item())
            pbar.set_postfix({"loss": loss_meter.avg})
            
            # Logging
            self.global_step += 1
            if self.global_step % 100 == 0:
                self.writer.add_scalar("train/loss", loss_meter.avg, self.global_step)
        
        return loss_meter.avg
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        loss_meter = AverageMeter()
        
        pbar = tqdm(val_loader, desc="Validating")
        
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, _ = self.model(input_ids)
            loss = self.criterion(logits.view(-1, self.model.vocab_size), 
                                target_ids.view(-1))
            
            loss_meter.update(loss.item())
            pbar.set_postfix({"loss": loss_meter.avg})
        
        return loss_meter.avg
    
    def train(self):
        """Train model"""
        logger.info("Loading dataset...")
        train_texts, val_texts = load_dataset(
            self.config.data.dataset_name,
            self.config.data.cache_dir,
            self.config.data.max_samples
        )
        
        if train_texts is None:
            logger.error("Failed to load dataset")
            return
        
        logger.info(f"Train samples: {len(train_texts)}")
        
        train_loader, val_loader = get_data_loaders(
            train_texts, self.tokenizer, self.config, split="train"
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            logger.info(f"Val loss: {val_loss:.4f}")
            
            # Logging
            self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, 
                              self.config.checkpoint_dir / "best_model.pt")
                logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                save_checkpoint(self.model, self.optimizer, epoch, val_loss,
                              self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
        
        self.writer.close()
        logger.info("Training completed!")


def main():
    # Load config
    cfg = config
    
    # Create trainer
    trainer = Trainer(cfg)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
