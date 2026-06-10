# Overview & Motivation

This document frames the class project: why Vision Transformers (ViTs) are expensive to run, the paper we reproduce (DynamicViT, NeurIPS 2021), how this work connects to the TC-Pruner research project, and a precise statement of what we do and do not reproduce.

## Why ViTs Are Expensive

A ViT splits an image into fixed-size patches and treats each patch as a token. At 384×384 input resolution with 16×16 patches, a single image becomes **576** patch tokens (plus one CLS token). Every transformer block then runs self-attention over that full sequence.

The cost matters for two reasons:

- **Attention scales as O(N²)** in the number of tokens N. Doubling the sequence length roughly quadruples the attention compute, so the token count is the dominant lever on per-image cost.
- **Background redundancy.** Many patches carry little task-relevant signal. For classification, the foreground object drives the prediction while large background regions contribute little. Spending full attention compute on every background patch at every layer is wasteful.

This is the opening that token pruning exploits: if most tokens are redundant after the first few layers, the model can carry fewer tokens through the deeper, more expensive blocks.

## The Selected Paper: DynamicViT (NeurIPS 2021)

**DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification**, Rao et al., NeurIPS 2021.

The core idea is straightforward:

1. A ViT processes image patches as tokens.
2. Not all tokens are equally useful.
3. A scoring module predicts per-token importance.
4. Low-importance tokens are pruned progressively across layers.
5. Accuracy drops only slightly while token count and compute decrease.

Pruning is **progressive**: rather than dropping tokens once, the model removes a fraction at several depths, so the surviving set shrinks stage by stage and the deepest blocks operate on a small token budget.

## Connection to the TC-Pruner Research Project

The motivating research project is **TC-Pruner**: task-conditioned token pruning for a vision-language-action (VLA) system. In that setting, a language instruction conditions which visual tokens are worth keeping, and the pruned representation feeds an action policy.

That full system is not a clean class-project reproduction — it carries JAX/PyTorch bridging, Meta-World robotics dependencies, a custom action head, and VLA-specific logic. Instead, this class project **isolates the task-agnostic core** of the idea: progressive token pruning in a ViT, studied in a small, controlled image-classification setting. The class story is:

> TC-Pruner studies task-conditioned token pruning in a VLA system. For the course project, the core token-pruning principle from a top-tier ViT paper is reproduced on a toy image-classification task, which then extends toward task-conditioned pruning in the VLA setting.

This keeps the scope honest and defensible while preserving the line of reasoning that connects the class work to the research project.

## What We Reproduce / What We Do NOT Reproduce

### Reproduce

- **Token importance scoring** — per-token importance to decide what to keep.
- **Top-k progressive pruning** — keep the top-scoring tokens and drop the rest, repeated across multiple layers so the budget compounds.
- **Accuracy-vs-compute tradeoff** — measure how accuracy moves as token count and FLOPs drop.

### Do NOT Reproduce

- Full ImageNet-scale experiments.
- The robotics / VLA stack (Meta-World training, the Octo backbone, language conditioning, action decoding).
- Exact original large-scale benchmark numbers.

## Setting and Headline Outcome

The reproduction runs on **CIFAR-10** (50,000 train / 10,000 test, 10 classes), with images upscaled to 384×384 to match the ViT positional embeddings. Token budget compounds across three pruning layers:

| Stage | Keep ratio | Tokens | Share of input |
| --- | --- | --- | --- |
| Input | — | 576 | 100% |
| After layer 3 | 0.75 | 432 | 75% |
| After layer 6 | 0.50 | 216 | 37.5% |
| After layer 9 | 0.40 | 87 | ≈15% |

![token budget](figures/fig_token_budget.png)

By the final pruning layer, roughly 85% of tokens have been dropped. The accuracy cost of pruning, measured on the same trained weights with pruning toggled, is **0.08 pt** (99.00% with pruning off vs 98.92% with pruning on), against a **−25.1%** reduction in GFLOPs at batch 1.

### A note on speed and memory (read honestly)

The compute saving does not translate into an unconditional speedup, and the design does not save memory:

- **Throughput is batch-size dependent.** Pruned is faster at server batch sizes — **1.40×** at batch 32 and **1.35×** at batch 128 — but **0.62× (slower)** at batch size 1, where pruner overhead outweighs the compute saved.
- **VRAM is higher for the pruned model**, not lower (e.g. **+71.6%** at batch 128), because the asymmetric design keeps full-length key/value tensors for a full-context read.

These tradeoffs are detailed in [Results](04_results.md) and the [Limitations](06_limitations.md) discussion. For the method itself, see [Method](02_method.md).
