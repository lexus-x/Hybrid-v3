# Dense vs Pruned ViT-Small/384 — CIFAR-10 Benchmark

Backbone: `vit_small_patch16_384` @ 384px | prune layers [3, 6, 9] | keep ratios [0.75, 0.5, 0.4]
Checkpoint: `/home/user/Desktop/vla_projects_archive/tools/tc-pruner/class_project/outputs/sota_pruned_small_atp_final/best.pt`

## Accuracy (same trained weights, pruning toggled)

| Mode | Test accuracy |
|---|---|
| Pruning OFF (keep=1.0, dense behaviour) | 99.00% |
| Pruning ON (keep 0.75,0.5,0.4) | 98.92% |

**Accuracy cost of pruning: 0.08 pt**

## Token budget

Patches per image (dense): **576** (+1 CLS). Keep ratios compound across stages.

| Pruned at layer | keep ratio | avg kept tokens | % of original |
|---|---|---|---|
| 3 | 0.75 | 432 | 75% |
| 6 | 0.5 | 216 | 38% |
| 9 | 0.4 | 87 | 15% |

Final blocks operate on **87 / 576 tokens (15%)** — ~85% of tokens dropped.

## Compute & speed

| Metric | Dense | Pruned | Change |
|---|---|---|---|
| Parameters | 21,815,434 | 23,586,058 | +8.1% (pruner modules) |
| GFLOPs (bs=1) | 12.45 | 9.32 | -25.1% |
| Throughput bs=1 (img/s) | 145.4 | 90.7 | 0.62x (pruned slower) |
| Latency bs=1 (ms/img) | 6.877 | 11.030 | 60.4% slower |
| Peak VRAM bs=1 (MB) | 469 | 719 | 53.3% more |
| Throughput bs=32 (img/s) | 307.8 | 430.7 | 1.40x |
| Latency bs=32 (ms/img) | 3.249 | 2.322 | 28.5% faster |
| Peak VRAM bs=32 (MB) | 836 | 1188 | 42.2% more |
| Throughput bs=128 (img/s) | 328.9 | 445.6 | 1.35x |
| Latency bs=128 (ms/img) | 3.040 | 2.244 | 26.2% faster |
| Peak VRAM bs=128 (MB) | 1972 | 3383 | 71.6% more |

## Honest reading

- **Accuracy is preserved**: pruning to 15% of final-stage tokens costs only ~0.1 pt.
- **Compute drops 25%** (FLOPs) and **throughput improves 1.35-1.40x** at server batch sizes (32, 128).
- **At batch size 1 the pruned model is slower** (0.62x): pruner overhead (top-k, gather, disabled flash-attn at 3 layers) dominates when there is little compute to save.
- **VRAM is higher** for the pruned model: the asymmetric attention keeps full K/V (all N tokens) at the pruned layers, trading memory for a full-context read.
