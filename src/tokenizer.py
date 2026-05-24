"""Tokenizer implementation"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import torch


class CharacterTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.built = False
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        chars = set()
        for text in texts:
            chars.update(set(text))
        
        chars = sorted(list(chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        self.built = True
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not self.built:
            raise RuntimeError("Tokenizer vocabulary not built. Call build_vocab first.")
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        return ''.join([self.idx_to_char[idx] for idx in token_ids if idx in self.idx_to_char])
    
    def save(self, path: Path):
        """Save tokenizer to file"""
        data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: Path):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        self.vocab_size = data['vocab_size']
        self.built = True


class BytePairEncodingTokenizer:
    """Byte Pair Encoding (BPE) tokenizer - simplified version"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_tokenizer = None
        self.bpe = {}
        self.bpe_rank = {}
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.built = False
        
    def build_vocab(self, texts: List[str], num_merges: int = None):
        """Build BPE vocabulary"""
        if num_merges is None:
            num_merges = self.vocab_size - 256
        
        # Initialize with character tokens
        vocab = defaultdict(int)
        for text in texts:
            words = text.split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
        
        # Perform BPE merges
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.bpe_rank[best_pair] = i
        
        # Build token to index mappings
        tokens = sorted(set(' '.join(vocab.keys()).split()))
        self.token_to_idx = {token: idx for idx, token in enumerate(tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.built = True
        
    @staticmethod
    def _get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Get frequency of adjacent pairs"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    @staticmethod
    def _merge_vocab(pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge most frequent pair in vocabulary"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not self.built:
            raise RuntimeError("Tokenizer vocabulary not built. Call build_vocab first.")
        
        words = text.split()
        tokens = []
        for word in words:
            word_tokens = list(word) + ['</w>']
            encoded = self._encode_word(word_tokens)
            tokens.extend(encoded)
        return tokens
    
    def _encode_word(self, word_tokens: List[str]) -> List[int]:
        """Encode individual word"""
        # Apply BPE merges
        word = ' '.join(word_tokens)
        while len(word_tokens) > 1:
            stats = self._get_stats({word: 1})
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            if best_pair not in self.bpe_rank:
                break
            word_tokens = self._merge_word(best_pair, word_tokens)
        
        # Convert to indices
        return [self.token_to_idx.get(token, 0) for token in word_tokens]
    
    @staticmethod
    def _merge_word(pair: Tuple[str, str], word_tokens: List[str]) -> List[str]:
        """Merge pair in word tokens"""
        new_word = []
        i = 0
        while i < len(word_tokens):
            if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == pair:
                new_word.append(word_tokens[i] + word_tokens[i + 1])
                i += 2
            else:
                new_word.append(word_tokens[i])
                i += 1
        return new_word
    
    def save(self, path: Path):
        """Save tokenizer"""
        data = {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'bpe_rank': {str(k): v for k, v in self.bpe_rank.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: Path):
        """Load tokenizer"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.token_to_idx = data['token_to_idx']
        self.idx_to_token = {int(k): v for k, v in data['idx_to_token'].items()}
        self.vocab_size = data['vocab_size']
        self.built = True


def get_tokenizer(tokenizer_type: str = "char", vocab_size: int = 10000):
    """Factory function to get tokenizer"""
    if tokenizer_type == "char":
        return CharacterTokenizer(vocab_size)
    elif tokenizer_type == "bpe":
        return BytePairEncodingTokenizer(vocab_size)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
