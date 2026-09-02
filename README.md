# Transformer from Scratch — Attention Is All You Need

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

A PyTorch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), built module by module and trained end-to-end — both on a toy task and on a real translation dataset — in a single notebook.

## What's in here

- **Full encoder-decoder Transformer**: embeddings, sinusoidal positional encoding, scaled dot-product attention, multi-head attention, position-wise feed-forward layers, residual connections + layer norm, encoder stack, decoder stack (masked self-attention + cross-attention), and the final linear + softmax output layer.
- **The paper's actual training recipe**: label smoothing loss, Adam with `betas=(0.9, 0.98)` and `eps=1e-9`, and the warmup-then-decay learning rate schedule.
- **Two training runs**:
  - A quick sanity check on a toy copy task (model has to reproduce its input) — runs on CPU in under two minutes.
  - Real translation training on **Multi30k** (English → German, ~29K sentence pairs), with a BPE tokenizer, GPU-sized batching, and a BLEU score at the end.

Everything lives in `transformer_attention_is_all_you_need.ipynb`.

## Architecture

| Component | Paper section |
|---|---|
| Input embeddings (scaled by √d_model) | 3.4 |
| Sinusoidal positional encoding | 3.5 |
| Scaled dot-product attention | 3.2.1 |
| Multi-head attention | 3.2.2 |
| Position-wise feed-forward network | 3.3 |
| Residual connections + layer norm | 3.1 |
| Encoder / decoder stacks (N layers) | 3.1 |
| Masking (padding + look-ahead) | 3.2.3 |
| Output linear + softmax | 3.4 |

Base model hyperparameters (Table 3 in the paper): `N=6`, `d_model=512`, `d_ff=2048`, `h=8`, `dropout=0.1` — a **~65M parameter model** at the real training run's ~8K shared BPE vocab.

## Training

**1. Copy task (quick check).** A smaller version of the architecture (`N=2`, `d_model=128`, `h=4`) trained on random sequence-copying, just to confirm masks/loss/gradients/schedule all work. Runs on CPU in under two minutes.
- Loss: **2.00 → 0.41** over 25 epochs
- Greedy-decode accuracy on 5 random test sequences: **96% average token match**

**2. Multi30k (real translation).** Downloads the actual [Multi30k](https://github.com/multi30k/dataset) English-German dataset, trains a BPE tokenizer (shared vocab, matching the paper's approach), and trains the full base-config model. The notebook auto-detects a GPU (`torch.cuda.is_available()`) and uses the base config on it — on CPU it falls back to a smaller config so the cell still runs, just as a quick check rather than a real training run. Ends with example translations and a corpus BLEU score.

On a **T4 GPU (Colab)**, the base config trains at a reasonable pace — increase `EPOCHS` in that cell for better translation quality; the default is a modest run so it finishes quickly.

## Requirements

```
torch
matplotlib
tokenizers
sacrebleu
```

```bash
pip install torch matplotlib tokenizers sacrebleu
```
(`tokenizers` and `sacrebleu` are also installed from inside the notebook via `!pip install`, so this matters mainly for running outside Colab.)

## Running it

Open `transformer_attention_is_all_you_need.ipynb` in Google Colab (select a **T4 GPU** runtime: `Runtime → Change runtime type → T4 GPU`) and run all cells top to bottom. It also runs on CPU-only environments (Jupyter, VS Code, etc.) — the real-training section just uses a smaller config there.

## Going further

- Train longer / bump `EPOCHS` and `N_EVAL` in the Multi30k section for better quality and a more reliable BLEU score.
- Swap Multi30k for WMT for a larger, harder dataset (same code, just point the data-loading cell at the new corpus).
- Increase `vocab_size` in the BPE trainer (paper uses ~37K on the much larger WMT corpus).

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

## License

MIT — see [LICENSE](LICENSE) for the full text issued by BASU PATIL
