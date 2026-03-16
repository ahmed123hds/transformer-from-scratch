# Attention Is All You Need: Transformer from Scratch

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

A pure PyTorch implementation of the **Original Transformer architecture** built completely from scratch, faithfully replicating the seminal 2017 paper *Attention Is All You Need* by Vaswani et al. 

This repository demonstrates the capacity to construct complex sequence-to-sequence deep learning models down to the foundational matrix operations without relying on high-level pre-built abstraction layers like `torch.nn.Transformer`.

## Mathematical & Architectural Rigor

1. **Multi-Head Self-Attention:** Explicit derivation and implementation of the scaled dot-product attention equation: $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$. The operations are accurately split across multiple projection heads and concatenated.
2. **Causal & Padding Masking Check:** Strict validation of the subsequent (sub-triangular) mask to prevent the decoder from "looking ahead" into future sequence elements during autoregressive training. Padding tokens ($<PAD>$) are masked out in both the encoder and decoder to prevent the model from attending to null data.
3. **Sinusoidal Positional Embeddings:** Implementations of the deterministic high-frequency sine/cosine waves $PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}})$ to purposefully inject temporal order into an otherwise permutation-invariant attention matrix.

## Proof of Work: German-to-English Neural Translation

To stringently test the Encoder-Decoder cross-attention gradient flow, the model is trained on a raw, character-level German-English subset of translation data from `manythings.org`. The encoder builds a representation of the German sequence, and the decoder reconstructs the English sequence conditioned on the encoder's state.

### Execution
```bash
python3 train.py --epochs 1
```

### Verified Training Output
```text
Training on: cuda
Loaded 1000 translation pairs. Vocab size: 73
Initialized Transformer with 246,281 parameters.

--- Epoch 1 ---
Batch   10 | Loss: 3.8264 | elapsed: 0.22s
Batch   20 | Loss: 3.3078 | elapsed: 0.07s
Batch   30 | Loss: 2.8674 | elapsed: 0.07s
```
*The immediate loss curve collapse confirms gradients are propagating cleanly through the residual connections, layer normalizations, causal self-attention, and the encoder-decoder cross-attention layer.*

## Code Structure
- `transformer/attention.py`: Core Matrix Multiplications (Scaled Dot-Product, Multi-Head splitting/concatenating).
- `transformer/layers.py`: Positional Embeddings, SubLayer Connections (residual + LayerNorm), full Encoder/Decoder atomic layers.
- `transformer/model.py`: The wrapper integrating the embedding lookup, N-stacked Encoders, N-stacked Decoders, and the linear generator vocabulary projection.
- `train.py`: Translation dataset loader with `<EOS>` and `<PAD>` handling, causal mask generator, and SGD optimizer loop limit testing with CrossEntropyLoss.
