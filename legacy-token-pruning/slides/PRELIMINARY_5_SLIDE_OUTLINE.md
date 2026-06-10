# Preliminary 5-Slide Outline

## Slide 1. Problem and Motivation

- Vision Transformers convert images into many patch tokens.
- More tokens mean higher attention cost.
- My main project, `TC-Pruner`, faces the same token-efficiency problem in a harder VLA setting.

## Slide 2. Selected Paper

- Paper: `DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification`
- Venue: NeurIPS 2021
- Core idea: learn which tokens matter and prune the rest

## Slide 3. Main Intuition

- Not every image patch is equally useful.
- Background tokens are often redundant.
- If the model learns token importance, it can keep informative patches and drop the rest.

## Slide 4. Reproduction Plan

- Dataset: CIFAR-10
- Baseline: dense ViT classifier
- Reproduction: simplified DynamicViT-style token pruning
- Evaluation: accuracy, kept-token ratio, inference time

## Slide 5. Relation To My Project

- `TC-Pruner` applies a related token-pruning idea in a vision-language-action system.
- The class project isolates the core pruning principle in a much smaller setting.
- This makes the course reproduction explainable and computationally realistic.
sffsdfsdfsdfsdfsdfsfsdfsdffdssfsdfsdfsdfsdfsd