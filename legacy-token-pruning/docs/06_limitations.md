# Limitations & Honest Caveats

This page enumerates the known limitations of the ATP-Lite reproduction. Each is stated plainly so that the trade-offs are clear before any number is read in isolation. Throughput numbers are conditional on batch size, and the pruned model uses *more* VRAM, not less — see the relevant sections below. For the full numbers in context, see [Results](04_results.md).

## Summary

| # | Limitation | One-line takeaway |
|---|------------|-------------------|
| 1 | Batch-size-dependent speedup | 1.35–1.40× at batch 32/128, but 0.62× (slower) at batch 1 |
| 2 | Higher VRAM | +71.6% at bs=128 from full-length K/V |
| 3 | +8.1% parameters | Extra pruner modules |
| 4 | Noisy importance maps | Single attention signal, not a trained scorer |
| 5 | No independent same-arch dense baseline | Ablation used as the reference instead |
| 6 | CIFAR-10 upscaled to 384 | An easy, somewhat artificial setting |

## Limitations

### 1. Speedup is batch-size dependent (slower at batch size 1)

The wall-clock speedup only materializes at server batch sizes. At batch size 1 the pruned model is **slower** than dense, because the pruner overhead — top-k selection, gather, and disabling fused/flash-attention at the 3 pruned layers to expose attention weights — outweighs the compute saved.

| Batch size | Dense (img/s) | Pruned (img/s) | Speedup |
|-----------:|--------------:|---------------:|:--------|
| 1 | 145.4 | 90.7 | 0.62× (pruned slower) |
| 32 | 307.8 | 430.7 | 1.40× |
| 128 | 328.9 | 445.6 | 1.35× |

![throughput](figures/fig_throughput.png)

The implication is concrete: this method helps batched/offline inference and server-side serving, but it is a net regression for strict single-image, latency-critical paths.

### 2. VRAM increases, not decreases

The asymmetric attention design keeps **full-length key/value (all N tokens)** at every pruned layer so the kept queries still get a full-context read. That choice trades memory for accuracy, so the pruned model uses *more* peak VRAM than dense at every batch size. A pure DynamicViT-style token-drop would instead save memory.

| Batch size | Dense (MB) | Pruned (MB) | Change |
|-----------:|-----------:|------------:|:-------|
| 1 | 469 | 719 | higher |
| 32 | 836 | 1188 | higher |
| 128 | 1972 | 3383 | +71.6% |

![vram](figures/fig_vram.png)

Anyone deploying this should budget for higher memory, not the memory savings one might expect from a "pruning" method.

### 3. +8.1% parameters

The pruner modules add parameters on top of the backbone. Parameter count rises from **21,815,434** (dense) to **23,586,058** (pruned), a **+8.1%** increase. The student is still small (~23.6M), but pruning here is not a parameter-reduction technique — it reduces FLOPs and (at batch) latency, not model size.

### 4. Token importance maps are noisy

Importance is read from a **single CLS-token attention signal** at blocks 3, 6, and 9 rather than from a trained scorer. The resulting maps are foreground-biased but noisy. This keeps the method simple (no separate learned scorer to train) at the cost of less reliable per-token rankings, especially on upscaled CIFAR-10. The qualitative token maps in [Token Visualizations](05_visualizations.md) show this behaviour directly.

### 5. No independently-trained same-architecture dense baseline at 384px

There is no separately trained, same-architecture dense model at 384px to compare against. The **99.00%** "pruning OFF" reference is the **same trained weights with pruning disabled** — an ablation, not an independent run. This is a deliberate choice: toggling pruning on the identical weights is the cleanest isolation of pruning's accuracy cost.

| Configuration | Test accuracy |
|---------------|--------------:|
| Pruning OFF (keep=1.0, same weights) | 99.00% |
| Pruning ON (0.75/0.5/0.4) | 98.92% |
| Accuracy cost of pruning | 0.08 pt |

The number it produces (a 0.08 pt cost) is therefore an isolation of the pruning effect, not a head-to-head against a second, independently optimized model.

### 6. CIFAR-10 upscaled to 384 is an easy setting

CIFAR-10 (32×32) is upscaled to **384×384** via bicubic interpolation to match the ViT positional embeddings. This is an easy, somewhat artificial setting: the effective information content per image is low relative to the input resolution, which makes high accuracy easier to reach and may flatter the pruning trade-off. Results here should not be read as evidence for harder, natively high-resolution datasets.

## What we'd do next

We would add an independently trained same-architecture dense baseline at 384px and evaluate on a natively high-resolution dataset (e.g. ImageNet) to remove the upscaling artifact and confirm the accuracy/speed trade-off generalizes. We would also replace the single CLS-attention signal with a trained token scorer and explore a memory-saving K/V variant so the method does not regress VRAM or single-image latency.
