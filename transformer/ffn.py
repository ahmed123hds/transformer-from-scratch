"""
ffn.py -- Position-wise Feed-Forward Networks

Implements the feed-forward sublayer inside the Transformer architecture.
"""

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    This computes:
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    
    Where:
        W_1 has shape (d_model, d_ff)
        W_2 has shape (d_ff, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Two linear transformations with a ReLU activation in between
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Original paper uses ReLU: FFN(x) = max(0, xW1 + b1) W2 + b2
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x : (batch_size, seq_len, d_model)
        Returns:
            out : (batch_size, seq_len, d_model)
        """
        # Linear + Activation + Dropout
        x = self.dropout(self.relu(self.linear1(x)))
        
        # Linear projection back to d_model
        x = self.linear2(x)
        
        return x
