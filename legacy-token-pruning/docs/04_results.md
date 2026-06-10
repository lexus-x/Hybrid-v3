# Results

All numbers on this page are measured on a single NVIDIA A100 80GB and read from `results/benchmark_results.json`. The same `vit_small_patch16_384` backbone is used throughout: dense speed/FLOPs use a clean dense model with no pruner modules, and the pruned configuration uses the trained ATP-Lite checkpoint with keep ratios 0.75 / 0.50 / 0.40 at layers 3 / 6 / 9.

See also: [Method](02_method.md) for how ATP-Lite works, and [Reproduction](07_reproduction.md) for the exact commands.

## Accuracy

Accuracy is measured on the same trained weights with pruning toggled, which isolates the accuracy cost of pruning from any difference in training.

| Configuration | CIFAR-10 test accuracy |
|---|---|
| Pruning OFF (keep = 1.0, dense behaviour) | 99.00% |
| Pruning ON (keep 0.75 / 0.50 / 0.40) | 98.92% |
| **Accuracy cost of pruning** | **0.08 pt** |

For reference, the dense models on the same dataset:

| Reference model | CIFAR-10 test accuracy |
|---|---|
| Teacher (`vit_large_patch16_384`) | 99.34% |
| Dense reference (`vit_base_patch16_224`) | 99.16% |

![accuracy](figures/fig_accuracy.png)

Toggling pruning on the trained student costs 0.08 pt (~0.1 pt). With knowledge distillation, the small pruned student stays within ~0.4 pt of the much larger dense models.

## Token budget

Token pruning is progressive and compounds: each pruned layer keeps the top-scoring fraction of the tokens that survived the previous layer. The input is 576 patch tokens (plus 1 CLS token).

| Stage | Keep ratio | Tokens | % of original |
|---|---|---|---|
| Input | — | 576 | 100% |
| After layer 3 | 0.75 | 432 | 75% |
| After layer 6 | 0.50 | 216 | 37.5% |
| After layer 9 | 0.40 | 87 | ≈15% |

![token budget](figures/fig_token_budget.png)

By the final pruned layer, 87 of the original 576 tokens remain (≈15% of the original; ≈85% dropped).

## Compute and speed

The pruner modules add parameters and reduce per-image FLOPs. Wall-clock speed depends on batch size: pruning is slower at batch size 1 and faster at server batch sizes (32, 128).

| Metric | Dense | Pruned | Change |
|---|---|---|---|
| Parameters | 21,815,434 | 23,586,058 | +8.1% |
| GFLOPs (batch 1) | 12.45 | 9.32 | −25.1% |

### Throughput (images/s)

| Batch size | Dense | Pruned | Speedup |
|---|---|---|---|
| 1 | 145.4 | 90.7 | 0.62× (pruned slower) |
| 32 | 307.8 | 430.7 | 1.40× |
| 128 | 328.9 | 445.6 | 1.35× |

### Latency (ms/img)

| Batch size | Dense | Pruned |
|---|---|---|
| 1 | 6.877 | 11.030 |
| 32 | 3.249 | 2.322 |
| 128 | 3.040 | 2.244 |

### Peak VRAM (MB)

| Batch size | Dense | Pruned | Change |
|---|---|---|---|
| 1 | 469 | 719 | — |
| 32 | 836 | 1188 | — |
| 128 | 1972 | 3383 | +71.6% |

![flops](figures/fig_flops.png)

![throughput](figures/fig_throughput.png)

![vram](figures/fig_vram.png)

## Honest reading

- **Accuracy is preserved.** Toggling pruning on the same trained weights costs 0.08 pt (99.00% → 98.92%), and the pruned student stays within ~0.4 pt of the larger dense models.
- **FLOPs drop ~25%.** Per-image compute falls from 12.45 to 9.32 GFLOPs (−25.1%) by removing ~85% of late-stage tokens.
- **Speedup is batch-size dependent.** Throughput is 1.40× at batch 32 and 1.35× at batch 128, but **0.62× (slower) at batch size 1** — at small batches the pruner overhead (top-k, gather, and disabling fused/flash-attention at the 3 pruned layers) outweighs the compute saved.
- **VRAM is higher, not lower.** Peak memory increases (+71.6% at batch 128): the asymmetric design keeps full-length key/value for a full-context read, trading memory for accuracy. A pure DynamicViT-style token-drop would instead save memory.
- **Parameters increase +8.1%** from the pruner modules.
