# Method — Asymmetric Token Pruning (ATP-Lite)

This document describes the token-pruning method reproduced and extended in this project. The starting point is DynamicViT (Rao et al., NeurIPS 2021); the variant implemented here is **ATP-Lite (Asymmetric Token Pruning)**. The implementation lives in [`src/sota_hybrid_vit.py`](../src/sota_hybrid_vit.py); measured outcomes are in [Results](04_results.md).

## DynamicViT recap

DynamicViT accelerates a Vision Transformer by progressively discarding uninformative patch tokens as depth increases. At selected layers it predicts a per-token keep/drop decision, then physically removes dropped tokens so that the sequence length — and therefore the cost of every subsequent self-attention and MLP layer — shrinks. The original method learns the decision with a lightweight prediction module and trains it end-to-end with a Gumbel-Softmax relaxation plus auxiliary objectives to keep training stable.

ATP-Lite keeps the core DynamicViT idea (progressive, hard token reduction at a few layers) but makes two deliberate departures: it derives token importance from an attention signal already present in the backbone rather than a separate learned scorer, and it changes the attention algebra at pruned layers so the kept tokens still read the full original context. The remainder of this document covers those choices.

## Importance scoring via CLS-token attention

ATP-Lite does **not** train a separate scorer. Token importance is read directly from the backbone's own attention at blocks **3, 6, and 9**. The CLS token aggregates global context across the sequence, so its attention distribution over patches is a usable saliency signal without extra parameters dedicated to scoring.

To expose the attention matrix, the attention module at each pruned block is monkey-patched at construction (`patched_attn_forward` in `src/sota_hybrid_vit.py`). The patch disables fused/flash attention (`fused_attn = False`) so the explicit softmax weights are materialized, then stores the head-averaged matrix:

```python
# shape [B, num_heads, N, N] -> [B, N, N]
self.saved_attn_weights = attn.mean(dim=1)
```

Inside the pruner, the CLS row/column is dropped and the per-token importance is the column-sum of the patch-to-patch attention:

```python
patch_attn = attn_weights[:, 1:, 1:]   # [B, N, N], drop CLS
importance = patch_attn.sum(dim=1)     # [B, N]
```

This is a single attention signal, not a trained objective. As noted in the limitations, the resulting importance maps are foreground-biased but noisy.

## Hard top-k drop

Given a keep ratio for the layer, the number of survivors is `num_keep = ceil(N * keep_ratio)` (with a floor of 1). The top-`num_keep` tokens by importance are selected with a hard `torch.topk`; the rest are discarded and the sequence dimension physically shrinks. There is no soft mask and no straight-through estimator — the same hard selection is used in training and inference.

## Compounding keep ratios

The keep ratios are **0.75 / 0.50 / 0.40** at layers 3 / 6 / 9, and they **compound**: each stage prunes only what survived the previous stage. Starting from 576 input patch tokens, the survivors are 432, then 216, then 87 — roughly 15% of the original sequence by layer 9 (about 85% dropped).

### Token flow

| Layer | Keep ratio | Tokens | % of original |
|-------|-----------|--------|---------------|
| Input | — | 576 | 100% |
| After layer 3 | 0.75 | 432 | 75% |
| After layer 6 | 0.50 | 216 | 37.5% |
| After layer 9 | 0.40 | 87 | ≈15% |

![progressive](figures/progressive_01_automobile.png)

## Asymmetric attention

The defining feature of ATP-Lite is the attention algebra at a pruned layer, implemented in `AsymmetricTokenPruning`. The query is sliced to the kept top-k tokens, but the **key and value stay full-length over all N tokens**:

```python
q  = self.q(q_tokens)          # [B, H, K, d]  — only the kept top-k
kv = self.kv(patch_tokens)     # built from all N tokens
k, v = kv[0], kv[1]            # [B, H, N, d]
attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, K, N]
attn = attn.softmax(dim=-1)
x = (attn @ v)                 # [B, H, K, d]
out_tokens = q_tokens + x      # residual
```

The motivation is a **full-context read with no routing overhead**. A pruned token still attends over every original token (the K×N attention), so background information that was about to be dropped is still absorbed into the survivors via the residual `q_tokens + x`. Because only the kept queries are computed, the attention is K×N rather than N×N, which is where the FLOP saving comes from. The trade-off is explicit: keeping full-length K/V costs memory rather than saving it. This is why peak VRAM is **higher** for the pruned model (see [Results](04_results.md)), in contrast to a pure DynamicViT-style drop, which would reduce memory.

## Progressive curriculum

Pruning is ramped in over training rather than applied at full strength from the start:

| Phase | Epochs | Behaviour |
|-------|--------|-----------|
| Warmup | 1–2 | No drop (keep ratios effectively 1.0) |
| Ramp | 3–6 | Linear schedule from 0 toward the target keep ratios |
| Full | 7+ | Full pruning at 0.75 / 0.50 / 0.40 |

The warmup lets the backbone and pruner modules settle before any tokens are removed; the linear ramp then introduces the drop gradually so the model adapts to progressively shorter sequences instead of facing the full reduction abruptly.

## Design history: AToM → ATP

An earlier variant, **AToM** (token absorption/merging), merged dropped tokens into survivors instead of slicing the query. AToM produced **zero wall-clock speedup**: the scatter/gather operations needed to absorb merged tokens were memory-bandwidth bound and ate the compute saved. ATP-Lite replaced it. By slicing the query to the kept tokens and reading full-length K/V directly (K×N attention, no scatter/gather routing), ATP-Lite turns the reduced sequence into an actual throughput gain at server batch sizes, while still letting survivors read the full context. Comments and identifiers referencing "AToM" remain in `src/sota_hybrid_vit.py` as artifacts of that history.

## Forward pass

```text
patch_embed + pos_embed -> [CLS | patch tokens]
for each block i in backbone:
    x = block([CLS | patch tokens])          # standard ViT block
    split -> CLS, patch_tokens
    if i in {3, 6, 9}:
        attn        = block.attn.saved_attn_weights      # [B, N+1, N+1]
        importance  = attn[:, 1:, 1:].sum(dim=1)         # drop CLS, [B, N]
        keep_idx    = topk(importance, ceil(N * keep_ratio[i]))
        q_tokens    = gather(patch_tokens, keep_idx)      # [B, K, D]
        # asymmetric attention: Q = K kept tokens, K/V = all N tokens
        out         = q_tokens + AsymmetricAttention(q_tokens, patch_tokens)
        patch_tokens = out                               # sequence shrinks to K
norm -> head -> logits
```

For the training recipe (KD, cached teacher logits, augmentation) see [Training](03_training.md); for measured accuracy, FLOPs, throughput, and VRAM see [Results](04_results.md).
