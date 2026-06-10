# Training Setup

This document describes the data pipeline, teacher/student models, knowledge
distillation, the offline-caching optimization that made KD tractable, the full
hyperparameter recipe, and the progressive pruning curriculum used to produce the
headline result. All measured numbers reported here are also covered in
[Results](04_results.md); the method itself is described in
[Method](02_method.md).

## Dataset

We train and evaluate on **CIFAR-10**: 50,000 training images and 10,000 test
images across 10 classes.

CIFAR-10 images are natively 32x32. The ViT backbones use 384x384 positional
embeddings, so every image is upscaled from **32x32 to 384x384 with bicubic
interpolation**. This matches the patch grid the pretrained positional embeddings
were trained for, allowing the ImageNet-pretrained weights to transfer without
resizing the embedding table.

## Models

The setup uses a teacher-student pair. The large teacher provides soft targets;
the small student is the model we actually prune and ship.

| Role | Backbone | Notes |
|------|----------|-------|
| Teacher | `vit_large_patch16_384` (ImageNet-21k -> 1k) | Dense, **99.34%** CIFAR-10 test accuracy (50 epochs) |
| Student | `vit_small_patch16_384` (~22M params, ImageNet-21k pretrained) | Trained pruned with KD |
| Dense reference | `vit_base_patch16_224` | **99.16%** |

## Knowledge Distillation

The student is trained with **knowledge distillation** against the cached teacher
outputs. The distillation loss is **KL-divergence on softened logits**, combined
with the hard-label cross-entropy:

- **distill_alpha = 0.8** — 80% weight on teacher soft targets, 20% on hard labels.
- **temperature T = 2.0** — applied to both teacher and student logits before the
  softmax used in the KL term.

This keeps the small pruned student close to much larger dense models while
training on the same CIFAR-10 data.

## Offline Teacher-Logit Caching

Running the large teacher inside the training loop dominated wall-clock time. The
key optimization removes the teacher from the loop entirely:

1. The teacher is run **once** over the 50,000 clean training images. Its outputs
   are saved as a `[50000, 10]` tensor on disk.
2. During student training, a custom **`KDMixup`** exploits the **linearity of
   MixUp**: when two images are blended at ratio `lambda`, the cached clean logits
   for those two images are blended at the *same* ratio to form the soft target.
   No teacher forward pass is needed for the mixed inputs.

This cut KD training time from **~35 hours to ~2.5 hours (~14x faster)**.

```python
# Conceptual: blend cached clean logits at the same ratio as the image MixUp
mixed_image  = lam * image_a      + (1 - lam) * image_b
soft_target  = lam * cached_a     + (1 - lam) * cached_b   # no teacher forward pass
```

## Hyperparameters

The run that produced the headline result lives in
`outputs/sota_pruned_small_atp_final/`.

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 0.05 |
| Scheduler | LinearLR warmup -> CosineAnnealingLR (eta_min 1e-6) |
| MixUp alpha | 0.8 |
| CutMix alpha | 1.0 |
| Label smoothing | 0.1 |
| Batch size | 128 |
| Epochs | 100 |
| Precision | AMP (mixed precision) |
| Resolution | 32x32 upscaled to 384x384 (bicubic) |
| Distillation alpha | 0.8 |
| Distillation temperature | 2.0 |

## Pruning Curriculum

Pruning is not applied from the first step. A **progressive curriculum** lets the
student adapt to a shrinking token budget gradually:

- **Warmup — epochs 1-2:** no token drop (the model trains dense).
- **Ramp — epochs 3-6:** keep ratios interpolate linearly from 1.0 toward their
  targets.
- **Full pruning — epoch 7 onward:** target keep ratios of **0.75 / 0.50 / 0.40**
  at layers 3 / 6 / 9, which compound across stages.

This staged schedule avoids destabilizing the pretrained representation before the
model has settled on which tokens to keep.

## Training Curve

![training curve](figures/fig_training_curve.png)

Training reaches a best test accuracy of **98.94% at epoch 90**. For the full
benchmark — accuracy with pruning toggled on/off, FLOPs, throughput, and VRAM,
including the honest batch-size-dependent speed results — see
[Results](04_results.md).
