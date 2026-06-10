# Final Presentation — Dynamic Token Pruning for Vision Transformers

**Course:** AIB3004 | **Date:** 2026-06-04 | **Limit:** 15 min, 10 slides
**Reproduced paper:** DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification (NeurIPS 2021)
**Motivating research project:** TC-Pruner (task-conditioned token pruning for a vision-language-action system)

> All numbers below are measured on CIFAR-10 with a `vit_small_patch16_384` student on an A100 80GB.
> Figures: `outputs/token_viz/`. Efficiency data: `outputs/benchmark_table.md`, `outputs/benchmark_results.json`.

---

## Slide 1 — Problem & Motivation

- Vision Transformers split an image into **patch tokens**; self-attention cost grows with the number of tokens (≈ O(N²)).
- A 384×384 image at patch-16 = **576 tokens** per image — most are background and contribute little.
- My research project **TC-Pruner** hits the same token-efficiency wall in a harder vision-language-action setting.
- **Question:** can we drop most tokens inside the network and keep accuracy?

*Speaker note: this is the bridge from my real project to a clean, course-safe reproduction.*

---

## Slide 2 — Selected Paper

- **DynamicViT** (Rao et al., NeurIPS 2021).
- Top-tier venue, public implementation, directly about token pruning.
- **Core idea:** not all tokens matter — learn token importance and **progressively prune** the unimportant ones across depth.
- Reported result on ImageNet: ~30–35% FLOPs cut with <0.5% accuracy drop.

---

## Slide 3 — DynamicViT Method (intuition)

- Insert lightweight **prediction modules** at a few layers that score each token's importance.
- Keep the **top-k** tokens, discard the rest (a binary keep/drop decision).
- Apply this at **several depths** so the token count shrinks stage by stage.
- Foreground patches (the object) should survive; background patches get dropped.

---

## Slide 4 — Simplified Reproduction Design

- I reproduce the **core principle**, not the full ImageNet pipeline.
- Importance score = **CLS-token attention** at layers **3, 6, 9** (the CLS token already aggregates global context — no separate learned scorer needed).
- **Top-K hard drop** with a **progressive curriculum**: warmup (epochs 1–2, no drop) → ramp (epochs 3–6) → full pruning.
- Variant used: **ATP-Lite (Asymmetric Token Pruning)** — query is sliced to Top-K, key/value stay full-length for a full-context read.
- **Stripped out** from TC-Pruner: language conditioning, action decoding, robotics state — kept only token scoring + selection.

---

## Slide 5 — Relation to TC-Pruner (my project)

- TC-Pruner prunes tokens **conditioned on the task/instruction** in a VLA model.
- This class project isolates the **task-agnostic** version: prune by visual importance alone.
- Same skeleton — importance scoring, differentiable/hard selection, global-context preservation — at a fraction of the complexity.
- Lesson transferred back: **where** and **how aggressively** to prune, and the batch-size caveat (Slide 9).

---

## Slide 6 — Dataset & Training Setup

- **Dataset:** CIFAR-10 (50k train / 10k test), upscaled 32→384 (bicubic) to match ViT positional embeddings.
- **Student:** `vit_small_patch16_384` (22 M params, ImageNet-21k pretrained). **Teacher:** `vit_large_patch16_384` (99.34%).
- **Knowledge Distillation** with **offline cached teacher logits** + a custom `KDMixup` exploiting MixUp linearity → ~14× faster KD (35 h → 2.5 h).
- Optimizer AdamW, lr 5e-5, cosine schedule, MixUp/CutMix, label smoothing 0.1, 100 epochs, AMP, batch 128.
- **Keep ratios `0.75 / 0.5 / 0.4`** at layers 3 / 6 / 9 (they **compound**).

---

## Slide 7 — Quantitative Results

**Accuracy (same weights, pruning toggled) — the clean ablation:**

| Mode | Test accuracy |
|---|---|
| Pruning OFF (keep=1.0) | **99.00%** |
| Pruning ON (0.75/0.5/0.4) | **98.92%** |
| **Cost of pruning** | **−0.08 pt** |

**Token budget (compounding):** 576 → 432 → 216 → **87 tokens (≈15% kept, ≈85% dropped)**

**Compute & speed (A100):**

| Metric | Dense | Pruned | Change |
|---|---|---|---|
| GFLOPs | 12.45 | 9.32 | **−25%** |
| Throughput @ bs=128 | 328.9 | 445.6 img/s | **1.35×** |
| Throughput @ bs=32 | 307.8 | 430.7 img/s | **1.40×** |

→ **Core DynamicViT claim reproduced: ~25% less compute, ~0.1 pt accuracy loss.**

---

## Slide 8 — Qualitative Token Visualizations

- Show `outputs/token_viz/progressive_01_automobile.png` — **576 → 432 → 216 → 87** tokens across layers 3/6/9; surviving tokens concentrate on the car body.
- Show `outputs/token_viz/summary_grid.png` — original vs final kept tokens across 6 classes (horse, automobile, truck, frog, ship, bird).
- Kept tokens are **foreground-biased but noisy** — expected, since CIFAR-10 is upscaled and importance is a single attention signal, not a trained scorer.

*Speaker note: be honest that the maps are not perfectly clean — that's a real observation, not a failure.*

---

## Slide 9 — Limitations (honest)

- **Speedup is batch-size-dependent:** ~1.4× at batch 32–128, but **0.62× (slower) at batch size 1** — pruner overhead (Top-K, gather, disabled flash-attn at 3 layers) dominates when little compute is saved.
- **VRAM goes up, not down** (+72% at bs=128): ATP-Lite keeps full-length K/V for the full-context read. A pure token-drop (DynamicViT-style) would save memory instead.
- **+8% parameters** from the pruner modules.
- CIFAR-10 upscaled to 384 is an easy, somewhat artificial setting; importance maps are noisy.
- No same-architecture *independently-trained* dense baseline at 384 — the 99.00% reference is the same weights with pruning disabled (an ablation).

---

## Slide 10 — Takeaways

- **Reproduced the core idea of DynamicViT:** progressive token pruning removes ~85% of late-stage tokens and ~25% of FLOPs while losing only ~0.1 pt accuracy on CIFAR-10.
- **Knowledge distillation** lets a small pruned student stay within ~0.4 pt of much larger dense models.
- **Efficiency claims must be measured, not assumed:** wall-clock benefit depends on batch size, and this asymmetric variant trades memory for accuracy.
- **Back to TC-Pruner:** the same scoring-and-selecting skeleton extends to task-conditioned pruning in a VLA system — this project is the clean, defensible kernel of that idea.
