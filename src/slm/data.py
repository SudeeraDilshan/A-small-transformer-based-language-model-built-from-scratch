"""Data loading and preprocessing utilities"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    
    def __init__(self, texts: List[str], tokenizer, max_seq_length: int = 256,
                 stride: int = None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride if stride is not None else max_seq_length
        
        # Tokenize all texts
        self.token_ids = []
        for text in texts:
            tokens = tokenizer.encode(text)
            self.token_ids.extend(tokens)
        
        logger.info(f"Total tokens: {len(self.token_ids)}")
    
    def __len__(self):
        """Number of sequences"""
        return max(0, (len(self.token_ids) - self.max_seq_length) // self.stride)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input and target"""
        start = idx * self.stride
        end = start + self.max_seq_length + 1
        
        if end > len(self.token_ids):
            # Pad if necessary
            tokens = self.token_ids[start:] + [0] * (end - len(self.token_ids))
        else:
            tokens = self.token_ids[start:end]
        
        tokens = torch.tensor(tokens[:self.max_seq_length + 1], dtype=torch.long)
        
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        return input_ids, target_ids


class MemoryDataset(Dataset):
    """In-memory dataset with sequences"""
    
    def __init__(self, sequences: List[torch.Tensor]):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input and target"""
        seq = self.sequences[idx]
        input_ids = seq[:-1]
        target_ids = seq[1:]
        return input_ids, target_ids


def create_text_dataset(texts: List[str], tokenizer, max_seq_length: int = 256,
                       validation_split: float = 0.1):
    """Create train/validation datasets from texts"""
    
    dataset = TextDataset(texts, tokenizer, max_seq_length)
    
    # Split into train/val
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for batching sequences"""
    inputs, targets = zip(*batch)
    
    # Pad sequences to same length in batch
    max_len = max(seq.shape[0] for seq in inputs)
    
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(inputs, targets):
        if inp.shape[0] < max_len:
            pad_len = max_len - inp.shape[0]
            inp = torch.nn.functional.pad(inp, (0, pad_len), value=0)
            tgt = torch.nn.functional.pad(tgt, (0, pad_len), value=-100)  # -100 for loss ignore
        
        padded_inputs.append(inp)
        padded_targets.append(tgt)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)


def get_data_loaders(texts: List[str], tokenizer, config, split: str = "train"):
    """Create data loaders"""
    
    if split == "train":
        dataset = TextDataset(texts, tokenizer, config.model.max_seq_length)
        # Split into train/val
        val_size = int(len(dataset) * config.data.validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.training.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.training.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    else:
        dataset = TextDataset(texts, tokenizer, config.model.max_seq_length)
        loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.training.num_workers,
            pin_memory=True
        )
        return loader


def load_tinystories_dataset(data_dir: Path = Path("./data"), 
                            max_samples: Optional[int] = None):
    """Load TinyStories dataset from Hugging Face"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets library not installed. Please run: pip install datasets")
        return None, None
    
    logger.info("Loading TinyStories dataset...")
    
    try:
        dataset = load_dataset("tiny_stories", cache_dir=str(data_dir))
        train_texts = dataset["train"]["text"]
        validation_texts = dataset["validation"]["text"]
        
        if max_samples:
            train_texts = train_texts[:max_samples]
            validation_texts = validation_texts[:min(len(validation_texts), max_samples // 10)]
        
        return train_texts, validation_texts
    
    except Exception as e:
        logger.error(f"Failed to load TinyStories: {e}")
        return None, None


def load_wikitext_dataset(data_dir: Path = Path("./data"),
                         max_samples: Optional[int] = None):
    """Load WikiText dataset"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets library not installed. Please run: pip install datasets")
        return None, None
    
    logger.info("Loading WikiText dataset...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2", cache_dir=str(data_dir))
        train_texts = [t for t in dataset["train"]["text"] if t.strip()]
        validation_texts = [t for t in dataset["validation"]["text"] if t.strip()]
        
        if max_samples:
            train_texts = train_texts[:max_samples]
            validation_texts = validation_texts[:min(len(validation_texts), max_samples // 10)]
        
        return train_texts, validation_texts
    
    except Exception as e:
        logger.error(f"Failed to load WikiText: {e}")
        return None, None


def load_dataset(dataset_name: str = "tiny_stories", data_dir: Path = Path("./data"),
                max_samples: Optional[int] = None):
    """Load dataset by name"""
    
    dataset_name = dataset_name.lower()
    
    if dataset_name == "tiny_stories":
        return load_tinystories_dataset(data_dir, max_samples)
    elif dataset_name == "wikitext":
        return load_wikitext_dataset(data_dir, max_samples)
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        return None, None
