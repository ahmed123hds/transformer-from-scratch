"""
attention.py -- Multi-Head Attention

Implements the core attention mechanism from "Attention Is All You Need".
1. Scaled Dot-Product Attention
2. Multi-Head Attention (which wraps the scaled dot-product attention)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) block.
    
    This computes:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    
    Attention is the scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of each head

        # Linear projections for Query, Key, and Value
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Final linear projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Compute Scaled Dot-Product Attention.
        q, k, v: (batch_size, num_heads, seq_len, d_k)
        mask: (batch_size, 1, seq_len, seq_len)
        """
        # (B, H, L_q, d_k) @ (B, H, d_k, L_k) -> (B, H, L_q, L_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Mask contains 0s for tokens to attend to, and 1s for tokens to ignore.
            # Convert 1s to -infinity so they vanish after softmax.
            scores = scores.masked_fill(mask == 1, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (B, H, L_q, L_k) @ (B, H, L_k, d_k) -> (B, H, L_q, d_k)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query : (batch_size, seq_len_q, d_model)
            key   : (batch_size, seq_len_k, d_model)
            value : (batch_size, seq_len_k, d_model)
            mask  : (batch_size, 1, seq_len_q, seq_len_k) optionally broadcastable
        Returns:
            output       : (batch_size, seq_len_q, d_model)
            attn_weights : (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 1. Linear projections and reshape into multiple heads:
        # (B, L, d_model) -> (B, L, H, d_k) -> (B, H, L, d_k)
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        x, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 3. Concatenate all heads
        # (B, H, L, d_k) -> (B, L, H, d_k) -> (B, L, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Final linear projection
        output = self.w_o(x)

        return output, attn_weights
