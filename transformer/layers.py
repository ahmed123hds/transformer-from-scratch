"""
layers.py -- Transformer Layers and Positional Encoding

Implements:
1. Positional Encoding (using sin/cos functions)
2. Encoder Layer (Self-Attention + FFN + Norm + Residual)
3. Decoder Layer (Masked Self-Attention + Cross-Attention + FFN + Norm + Residual)
"""

import math
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .ffn       import FeedForward


class PositionalEncoding(nn.Module):
    """
    Positional Encoding injects information about the relative or absolute
    position of the tokens in the sequence.
    
    PE_sub(pos, 2i) = sin(pos / 10000^{2i/d_{model}})
    PE_sub(pos, 2i+1) = cos(pos / 10000^{2i/d_{model}})
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create constant positional encoding matrix 'pe' with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Position vector (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Frequencies vector (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Compute sines and cosines
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (1, max_len, d_model) for broadcasting over batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as a buffer so it doesn't get updated by the optimizer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to token embeddings.
        Args:
            x : Token embeddings (batch_size, seq_len, d_model)
        Returns:
            Positionally encoded token embeddings
        """
        seq_len = x.size(1)
        # x is a Variable, self.pe is a tensor. Add dynamically.
        return x + self.pe[:, :seq_len, :]


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note: "Attention Is All You Need" places LayerNorm AFTER the residual connection (post-ln).
    Modern architectures often use pre-ln, but for historical fidelity, we'll
    use post-ln as originally described: LayerNorm(x + Sublayer(x)).
    Actually, let's implement post-LN to exactly match the original 2017 paper wording,
    where x is normalized after sub-layer block.
    """

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        # x + sublayer(x) with dropout, then normalization
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """
    An Encoder layer consists of two sublayers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = FeedForward(d_model, d_ff, dropout)
        
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.size      = d_model

    def forward(self, x, mask=None):
        """
        x    : (batch_size, seq_len, d_model)
        mask : (batch_size, 1, seq_len, seq_len) optionally
        """
        # 1. Multi-Head Self-Attention
        # self.self_attn returns (output, weights). We only want output.
        x = self.sublayer1(x, lambda _x: self.self_attn(_x, _x, _x, mask)[0])
        
        # 2. Position-wise Feed-Forward Network
        x = self.sublayer2(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    """
    A Decoder layer consists of three sublayers:
    1. Masked Multi-Head Self-Attention (prevents looking at future tokens)
    2. Multi-Head Cross-Attention (attends to encoder outputs)
    3. Position-wise Feed-Forward Network
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
        self.size      = d_model

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        x        : Target sequence (batch_size, seq_len, d_model)
        memory   : Encoder output (batch_size, seq_len, d_model)
        src_mask : Optional mask for encoder output (e.g. padding mask)
        tgt_mask : Mask for target sequence (subsequent mask + padding mask)
        """
        # 1. Masked Multi-Head Self-Attention
        x = self.sublayer1(x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask)[0])
        
        # 2. Multi-Head Cross-Attention (over Encoder Memory)
        # Query comes from decoder (x), Key and Value come from encoder (memory)
        x = self.sublayer2(x, lambda _x: self.cross_attn(_x, memory, memory, src_mask)[0])
        
        # 3. Position-wise Feed-Forward Network
        x = self.sublayer3(x, self.ffn)
        return x
