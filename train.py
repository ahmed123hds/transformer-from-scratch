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

def load_data(path, max_len=32):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    
    src_texts = []
    tgt_texts = []
    
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            src_texts.append(parts[1].strip() + "<EOS>") # German is source
            tgt_texts.append(parts[0].strip() + "<EOS>") # English is target
            
    # Simple character-level tokenizer
    chars = sorted(list(set("".join(src_texts + tgt_texts))))
    vocab_size = len(chars) + 2 # + pad, + unk
    pad_id = 0
    unk_id = 1
    
    char2id = {c: i+2 for i, c in enumerate(chars)}
    
    def encode(text):
        return [char2id.get(c, unk_id) for c in text][:max_len]
        
    src_data = [encode(t) for t in src_texts]
    tgt_data = [encode(t) for t in tgt_texts]
    
    return src_data, tgt_data, vocab_size, pad_id

def data_generator(src_data, tgt_data, batch_size, max_len, pad_id, device):
    """
    Generate batches.
    """
    num_batches = len(src_data) // batch_size
    for i in range(num_batches):
        batch_src = src_data[i*batch_size : (i+1)*batch_size]
        batch_tgt = tgt_data[i*batch_size : (i+1)*batch_size]
        
        # Pad sequences
        src = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        tgt = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            src[b, :len(batch_src[b])] = torch.tensor(batch_src[b])
            tgt[b, :len(batch_tgt[b])] = torch.tensor(batch_tgt[b])
            
        src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
        
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        
        tgt_mask = make_std_mask(tgt_in, pad_id)
        
        yield src, tgt_in, src_mask, tgt_mask, tgt_out

def train_epoch(model, optimizer, data_gen, eval_interval=10):
    """
    Single train loop.
    """
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for i, (src, tgt_in, src_mask, tgt_mask, tgt_out) in enumerate(data_gen):
        optimizer.zero_grad()
        
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
    
    src_data, tgt_data, vocab_size, pad_id = load_data("data/deu_subset.txt", args.seq_len)
    print(f"Loaded {len(src_data)} translation pairs. Vocab size: {vocab_size}")

    config = TransformerConfig(
        src_vocab_size = vocab_size,
        tgt_vocab_size = vocab_size,
        d_model        = args.d_model,
        num_encoder    = args.num_layers,
        num_decoder    = args.num_layers,
        num_heads      = args.num_heads,
        d_ff           = args.d_model * 4,
        pad_token_id   = pad_id,
        dropout        = 0.1
    )
    
    model = Transformer(config).to(device)
    print(f"Initialized Transformer with {model.count_parameters():,} parameters.")
    
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch} ---")
        generator = data_generator(src_data, tgt_data, args.batch_size, args.seq_len, pad_id, device)
        train_epoch(model, optimizer, generator, eval_interval=10)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len",    type=int, default=32)
    p.add_argument("--d_model",    type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads",  type=int, default=4)
    p.add_argument("--lr",         type=float, default=5e-4)
    
    args = p.parse_args()
    run_experiment(args)
