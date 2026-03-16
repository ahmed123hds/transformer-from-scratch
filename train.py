"""
train.py -- Transformer Copy Task

Trains the completed Transformer encoder-decoder model on a synthetic "copy" task.
The copy task is simple: given a random sequence of numbers (e.g., [3, 7, 2, ...]),
the model learns to copy it to the output.

This proves that all the dimensions, embeddings, multi-head attention blocks,
cross-attention, masking, and gradients work properly end-to-end.
"""

import time
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam

from transformer.model import Transformer, TransformerConfig

def make_std_mask(tgt, pad_id):
    """
    Creates a mask to hide padding and future words.
    Combines a subsequent mask (causal, triangular) and a padding mask.
    """
    batch_size, seq_len = tgt.size()
    
    # 1. Hide padding elements in the target sequence
    # Shape: (batch_size, 1, 1, seq_len)
    tgt_pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
    
    # 2. Hide future elements in the target sequence (subsequent mask)
    # Shape: (1, 1, seq_len, seq_len)  
    tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
    
    # Combine both constraints
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    return tgt_mask

def data_generator(batch_size, num_batches, seq_len, vocab_size, device):
    """
    Generate synthetic data for a copy task.
    """
    for _ in range(num_batches):
        # Generate random tokens between 1 and vocab_size (0 is reserved for padding)
        data = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
        
        # In a generic setup, the source mask prevents attending to padded values
        # We don't have padding here, so it is just a mask of all 1s
        src_mask = torch.ones((batch_size, 1, seq_len, seq_len), device=device).bool()
        
        # Teacher forcing inputs: remove the last element for the decoder input
        tgt_in = data[:, :-1]
        
        # Target truths: remove the first element (shift by 1)
        tgt_out = data[:, 1:]
        
        # Build causal mask for target input
        tgt_mask = make_std_mask(tgt_in, pad_id=0)
        
        yield data, tgt_in, src_mask, tgt_mask, tgt_out

def train_epoch(model, optimizer, data_gen, eval_interval=100):
    """
    Single train loop.
    """
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for i, (src, tgt_in, src_mask, tgt_mask, tgt_out) in enumerate(data_gen):
        optimizer.zero_grad()
        
        # Forward pass returning cross-entropy loss directly
        logits, loss = model(
            src_seq  = src,
            tgt_seq  = tgt_in,
            src_mask = src_mask,
            tgt_mask = tgt_mask,
            targets  = tgt_out
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (i + 1) % eval_interval == 0:
            elapsed = time.time() - start_time
            print(f"Batch {i + 1:4d} | Loss: {total_loss / eval_interval:.4f} | elapsed: {elapsed:.2f}s")
            total_loss = 0
            start_time = time.time()

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Hyperparameters for the synthetic task
    vocab_size = args.vocab_size
    seq_len    = args.seq_len
    
    # The original paper used d_model=512, but we'll scale it down for this quick test
    config = TransformerConfig(
        src_vocab_size = vocab_size,
        tgt_vocab_size = vocab_size,
        d_model        = args.d_model,
        num_encoder    = args.num_layers,
        num_decoder    = args.num_layers,
        num_heads      = args.num_heads,
        d_ff           = args.d_model * 4,
        pad_token_id   = 0,
        dropout        = 0.1
    )
    
    model = Transformer(config).to(device)
    print(f"Initialized Transformer with {model.count_parameters():,} parameters.")
    
    # Optimizer (Adam with relatively high learning rate since task is trivial)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch} ---")
        generator = data_generator(args.batch_size, args.batches, seq_len, vocab_size, device)
        train_epoch(model, optimizer, generator, eval_interval=50)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int, default=3)
    p.add_argument("--batches",    type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=100)
    p.add_argument("--seq_len",    type=int, default=32)
    p.add_argument("--d_model",    type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads",  type=int, default=4)
    p.add_argument("--lr",         type=float, default=5e-4)
    
    args = p.parse_args()
    run_experiment(args)
