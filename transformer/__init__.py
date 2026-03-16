"""
Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
Implemented from scratch in PyTorch.

Modules:
    attention -- Multi-Head Attention and Scaled Dot-Product Attention
    ffn       -- Position-wise Feed-Forward Networks
    layers    -- Encoder/Decoder layers and Positional Encoding
    model     -- Full Encoder-Decoder Transformer Architecture
"""

from .model import Transformer, TransformerConfig
