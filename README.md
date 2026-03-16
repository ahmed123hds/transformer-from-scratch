# Transformer-from-scratch 🤖

A clean, mathematics-first, heavily-commented implementation of the original Transformer architecture from ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762).

This repository is built to demonstrate a deep understanding of the math and engineering behind modern deep learning. Every component—from Scaled Dot-Product Attention to Positional Encodings—is implemented from scratch in PyTorch.

## Features
- **Encoder-Decoder Architecture**: Fully implements the original seq2seq blueprint, not just a decoder-only model block.
- **Multi-Head Attention**: Exact PyTorch implementation of the `softmax(QK^T / sqrt(d))V` mechanism.
- **Positional Encodings**: Manual injection of sine and cosine sinusoidal frequencies into the token embeddings.
- **Masking Mechanism**: Demonstrates proper causality (future masking) and padding masking on the decoder side.
- **Synthetic Copy Task**: Includes `train.py` that successfully trains the model to copy sequences, proving end-to-end functionality of gradients and attention mechanisms.

## Project Structure
```text
transformer/
├── attention.py # Multi-Head Attention and Scaled Dot-Product
├── ffn.py       # Position-wise Feed-Forward Networks
├── layers.py    # Encoder/Decoder Layers & Positional Encoding
├── model.py     # Full Transformer gluing it all together
├── __init__.py  
train.py         # End-to-end training loop on a synthetic task
```

## Running the Code

### Train the model
You can train the full Encoder-Decoder model on a synthetic sequence copying task to verify the attention weights and loss convergence:

```bash
python train.py --epochs 3
```

By default it uses a very small hidden dimension (`d_model=64`) so that it converges instantly and proves that Cross-Attention and Masked Self-Attention are working flawlessly to route information from the Encoder to the Decoder.

## Mathematical Mapping

1. **Attention**: The core formula implemented in `transformer/attention.py`:
   $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
2. **Multi-Head**: Instead of performing a single attention function, queries, keys and values are linearly projected $h$ times. We perform attention in parallel, concatenate them, and project again.
3. **Feed Forward**: A two-layer linear transformation with ReLU activation.
   $$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$
4. **Positional Encoding**: Uses sinusoidal functions to give the model a sense of sequence order since there are no recurrences or convolutions:
   $$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}}) $$
   $$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}}) $$
   
This exact mapping ensures that the implementation is 100% true to the original 2017 paper.
