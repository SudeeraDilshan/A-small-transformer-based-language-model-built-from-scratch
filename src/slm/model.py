"""Transformer-based language model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryPositionEmbedding(nn.Module):
    """Rotary position embedding (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Pre-compute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Pre-compute rotation matrices
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for rotations"""
        seq_len = x.shape[-2]
        return self.cos_cached[..., :seq_len, :], self.sin_cached[..., :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with rotary position embeddings"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1, 
                 attention_dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)
        
        self.rope = RotaryPositionEmbedding(self.head_dim)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use cache if provided (for generation)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)
        
        # Apply rotary position embedding
        cos, sin = self.rope(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask (causal mask for language modeling)
        if attn_mask is not None:
            scores = scores + attn_mask
        else:
            # Create causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        return attn_output, (k, v)


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    
    def __init__(self, hidden_size: int, ffn_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.fc2 = nn.Linear(hidden_size, ffn_hidden_size)
        self.fc3 = nn.Linear(ffn_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU = (Linear + SiLU) * Linear
        gate = F.silu(self.fc1(x)) * self.fc2(x)
        return self.dropout(self.fc3(gate))


class TransformerBlock(nn.Module):
    """Single transformer block"""
    
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int, 
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout, attention_dropout)
        
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_hidden_size, dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm transformer block
        x_norm = self.ln1(x)
        attn_output, kv_cache = self.attention(x_norm, attn_mask, past_kv)
        x = x + attn_output
        
        x_norm = self.ln2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        
        return x, kv_cache


class LanguageModel(nn.Module):
    """Small transformer-based language model"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 4,
                 num_heads: int = 8, ffn_hidden_size: int = 1024, max_seq_length: int = 256,
                 dropout: float = 0.1, attention_dropout: float = 0.1, 
                 layer_norm_eps: float = 1e-6):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_hidden_size, 
                           dropout, attention_dropout, layer_norm_eps)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_final = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Weight sharing between input and output embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, past_kv_cache=None) -> Tuple[torch.Tensor, list]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token and position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)
        
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Pass through transformer blocks
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_cache[i] if past_kv_cache is not None else None
            x, kv = layer(x, past_kv=past_kv)
            new_kv_caches.append(kv)
        
        # Output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        return logits, new_kv_caches
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95) -> torch.Tensor:
        """Generate text autoregressively"""
        for _ in range(max_new_tokens):
            # Get logits
            logits, _ = self(input_ids[:, -self.max_seq_length:])
            
            # Get last token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[:, indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
