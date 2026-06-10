# AIB3004 Project Plan

## 1. Brutal Summary

Do **not** present the full `OctoTC` / robotics stack as the paper reproduction.

That is not a clean class project reproduction. It is a research codebase with:

- JAX + PyTorch bridging
- Meta-World robotics dependencies
- custom action head / VLA logic
- results that are not strong enough to sell as a finished reproduction

Use this repo as the **project context** and reproduce only the **core token-pruning idea** in a small, controlled setting.

## 2. Recommended Topic and Paper

**Recommended topic:** Vision Transformers (ViT)

**Recommended paper:** DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification  
Venue: NeurIPS 2021

Why this choice:

- top-tier venue
- public implementation exists
- directly connected to this repo's core idea: token pruning
- realistic to explain and reproduce on a toy dataset

## 3. Honest Connection To This Repo

Your class story should be:

> My research project is `TC-Pruner`, which studies task-conditioned token pruning in a vision-language-action system. For the course project, I reproduce the **core token pruning principle** from a top-tier ViT paper in a toy image classification setup, then explain how that idea extends into my own project.

This is honest and defensible.

## 4. What To Reproduce

Reproduce only these core ideas from DynamicViT:

1. A ViT processes image patches as tokens.
2. Not all tokens are equally useful.
3. A learned scoring module can predict token importance.
4. Low-importance tokens can be pruned progressively.
5. Accuracy drops only slightly while token count / compute decreases.

## 5. What NOT To Reproduce

Do not try to reproduce:

- full ImageNet-scale experiments
- full robotics / Meta-World training
- Octo backbone behavior
- the whole `OctoTC` system
- exact original large-scale benchmark numbers

## 6. Concrete Implementation Scope

### 6.1 Toy Setup

Use a small image classification task:

- CIFAR-10 is the safest default
- CIFAR-100 is acceptable if CIFAR-10 is too easy
- resize images to match ViT input if needed

### 6.2 Baseline

Train or fine-tune a small ViT classifier:

- dense tokens
- no pruning

### 6.3 Reproduction Variant

Implement a simplified token-pruning module:

- per-token importance scoring
- top-k token selection
- progressive pruning across layers or stages
- fixed token budget

### 6.4 Repo Connection

Reuse this repo only at the idea/module level:

- `amet/models/plugins/pruner.py`

Keep only the reusable concepts:

- token scoring
- differentiable top-k / hard selection
- global token preservation + selective focus

Strip out:

- language conditioning
- EASA
- robotics-specific state inputs
- action decoding

## 7. Minimum Experiments

### 7.1 Main Comparison

- Dense ViT baseline
- Pruned ViT variant

Report:

- test accuracy
- number of kept tokens
- inference time per batch or relative speedup

### 7.2 Optional Ablations

- prune ratio: 25%, 50%, 75%
- one-stage vs two-stage pruning
- fixed token budget vs adaptive budget

### 7.3 Visualizations

Show 2-4 images with:

- original patch grid
- kept tokens highlighted
- dropped tokens dimmed

## 8. Exact Claim You Can Defend

Safe claim:

> I reproduced the core idea of dynamic token pruning for Vision Transformers on a toy dataset, and I connected it to my own TC-Pruner project, which applies a related pruning principle in a more complex vision-language-action setting.

Unsafe claim:

> I reproduced my full robotics project as the paper implementation.

Do not say that.

## 9. Preliminary Presentation Plan

**Date:** April 16, 2026  
**Limit:** 8 minutes, 5 slides

### Slide 1. Problem and Motivation

- ViTs are expensive because every image patch becomes a token
- attention cost grows with token count
- my project also faces this issue in a VLA setting

### Slide 2. Selected Paper

- DynamicViT, NeurIPS 2021
- key idea: prune unimportant tokens dynamically

### Slide 3. High-Level Intuition

- foreground patches matter more than background patches
- learn token importance and keep only useful tokens
- connect this intuition to `TC-Pruner`

### Slide 4. Implementation Plan

- dataset: CIFAR-10
- baseline: dense ViT
- reproduction: simplified dynamic token pruning
- metrics: accuracy, kept tokens, latency

### Slide 5. Expected Outcome and Relevance To My Project

- show that token pruning can reduce compute with limited accuracy loss
- explain that my project extends this idea toward task-conditioned pruning in VLA

## 10. Final Presentation Plan

**Date:** June 4, 2026  
**Limit:** 15 minutes, 10 slides

Use this structure:

1. problem
2. paper background
3. DynamicViT method
4. simplified reproduction design
5. relation to `TC-Pruner`
6. dataset and training setup
7. quantitative results
8. qualitative token visualizations
9. limitations
10. takeaways

## 11. Suggested Folder Plan

Keep all class-only work here:

- `class_project/scripts/`
- `class_project/src/`
- `class_project/slides/`
- `class_project/outputs/`

Do not mix it into the active robotics scripts and model files.

## 12. One-Sentence Positioning

> This course project reproduces the core idea of DynamicViT on a toy vision task and uses my TC-Pruner research project as the motivating application domain.
