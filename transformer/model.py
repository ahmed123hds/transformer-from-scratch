"""
model.py -- Full Transformer Architecture

Implements the Encoder, Decoder, and the full Transformer model.
"""

from dataclasses import dataclass
import copy
import torch
import torch.nn as nn
from typing import Optional

from .layers import EncoderLayer, DecoderLayer, PositionalEncoding


@dataclass
class TransformerConfig:
    """
    Hyperparameters for the Transformer model.
    """
    src_vocab_size : int
    tgt_vocab_size : int
    d_model        : int   = 512
    num_encoder    : int   = 6
    num_decoder    : int   = 6
    num_heads      : int   = 8
    d_ff           : int   = 2048
    dropout        : float = 0.1
    max_seq_len    : int   = 5000
    pad_token_id   : int   = 0


def clone_module(module, N):
    """
    Produce N identical layers to form the encoder or decoder stacks.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    """
    Input embeddings. Instead of a bare nn.Embedding, the paper scales
    embeddings by sqrt(d_model) before adding positional encodings.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embed   = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # We multiply by sqrt(d_model) so the variance of the embeddings 
        # is roughly similar to the variance of positional encodings.
        return self.embed(x) * (self.d_model ** 0.5)


class Encoder(nn.Module):
    """
    The core encoder stack for the Transformer.
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        layer       = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.layers = clone_module(layer, num_layers)
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    The core decoder stack for the Transformer.
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        layer       = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.layers = clone_module(layer, num_layers)
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        Pass the output sequentially through all decoding layers.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    The full Transformer architecture.
    "Attention Is All You Need" (Vaswani et al, 2017)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.src_embed = Embeddings(config.src_vocab_size, config.d_model)
        self.tgt_embed = Embeddings(config.tgt_vocab_size, config.d_model)

        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        self.dropout      = nn.Dropout(config.dropout)
        
        # In the original paper, encoder and decoder use the same configurations
        self.encoder = Encoder(
            num_layers = config.num_encoder,
            d_model    = config.d_model,
            num_heads  = config.num_heads,
            d_ff       = config.d_ff,
            dropout    = config.dropout
        )
        
        self.decoder = Decoder(
            num_layers = config.num_decoder,
            d_model    = config.d_model,
            num_heads  = config.num_heads,
            d_ff       = config.d_ff,
            dropout    = config.dropout
        )
        
        self.generator = nn.Linear(config.d_model, config.tgt_vocab_size)

        # Initialize parameters with Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_seq, src_mask=None):
        """
        Embed source tokens, add positional encoding, and pass entirely through Encoder.
        """
        x = self.pos_encoding(self.src_embed(src_seq))
        x = self.dropout(x)
        return self.encoder(x, mask=src_mask)

    def decode(self, tgt_seq, memory, src_mask=None, tgt_mask=None):
        """
        Embed target tokens, add positional encoding, and pass entirely through Decoder,
        attending to the encoded memory.
        """
        x = self.pos_encoding(self.tgt_embed(tgt_seq))
        x = self.dropout(x)
        return self.decoder(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)

    def forward(self, src_seq, tgt_seq, src_mask=None, tgt_mask=None, targets=None):
        """
        Args:
            src_seq : Expected shape (batch_size, src_seq_len)
            tgt_seq : Expected shape (batch_size, tgt_seq_len)
            src_mask: Masks out padding tokens in the source. Shape (batch_size, 1, 1, src_seq_len)
            tgt_mask: Masks out padding + future tokens in target. Shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
            targets : Optional target tokens for cross entropy loss computation.

        Returns:
            logits  : Logits of shape (batch, target_len, target_vocab)
            loss    : Computed CrossEntropyLoss if targets are provided, else None
        """
        
        # 1. Forward through encoder
        memory = self.encode(src_seq, src_mask)
        
        # 2. Forward through decoder
        out = self.decode(tgt_seq, memory, src_mask, tgt_mask)
        
        # 3. Final Linear projection to target vocabulary
        logits = self.generator(out)
        
        loss = None
        if targets is not None:
            # Flatten predictions and targets to calculate the loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1), 
                ignore_index=self.config.pad_token_id
            )

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

