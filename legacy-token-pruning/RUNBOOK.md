# Class Project Runbook

This is the shortest sane path to a toy-but-sincere class demo.

## 1. Pretrain SimCLR Backbone

Run the contrastive learning pretraining step to learn better initial representations:

```bash
python class_project/scripts/pretrain_simclr.py \
  --output-dir class_project/outputs/simclr_run \
  --epochs 100 \
  --batch-size 512
```

## 2. Run the experiment sweep

Run the main experiments, initialized with the pretrained weights from step 1:

```bash
python class_project/scripts/run_experiment_grid.py \
  --output-root class_project/outputs/experiments \
  --pretrained-checkpoint class_project/outputs/simclr_run/best.pt \
  --epochs 10 \
  --batch-size 128 \
  --keep-ratios 0.75,0.5,0.25
```

Expected variants:

- `dense`
- `pruned_k75`
- `pruned_k50`
- `pruned_k25`

## 3. Generate plots

```bash
python class_project/scripts/plot_experiment_results.py \
  --experiment-root class_project/outputs/experiments
```

Expected outputs under `class_project/outputs/experiments/plots/`:

- `accuracy_curves.png`
- `accuracy_bars.png`
- `speed_bars.png`
- `kept_tokens_bars.png`
- `summary_table.md`

## 4. Export token images

```bash
python class_project/scripts/visualize_tokens.py \
  --checkpoint class_project/outputs/experiments/pruned_k50/best.pt \
  --output-dir class_project/outputs/experiments/pruned_k50/token_viz \
  --num-images 8
```

## 5. Export a GIF

```bash
python class_project/scripts/make_token_animation.py \
  --checkpoint class_project/outputs/experiments/pruned_k50/best.pt \
  --output-dir class_project/outputs/experiments/pruned_k50/animations \
  --sample-index 0
```

## 6. Fast smoke test

If you want to verify the pipeline before a real run:

```bash
python class_project/scripts/run_experiment_grid.py \
  --output-root class_project/outputs/smoke \
  --epochs 1 \
  --max-train-batches 4 \
  --max-eval-batches 2 \
  --keep-ratios 0.5
```

Then:

```bash
python class_project/scripts/plot_experiment_results.py \
  --experiment-root class_project/outputs/smoke
```
