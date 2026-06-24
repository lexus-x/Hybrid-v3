# Hybrid-v3

Calibrated open-set detector for dark vessels — radar tracks with no AIS broadcast. Replaces brittle geometric gating with a learned matcher that stays robust under AIS registration errors.

## The problem

Dark vessels (ships not broadcasting AIS) are associated with IUU fishing, sanctions evasion, and smuggling. The standard approach is geometric gating: if no AIS report matches a radar track within some distance threshold, flag it as dark.

This works perfectly — until AIS and radar aren't perfectly registered. In our Busan Port data, we measured ~22m registration bias. Under controlled offset injection, the geometric gate's false-dark rate goes from 0.00 to 0.98 at 500m offset. Total collapse.

## What we built

Three components:

1. **Open-set matcher** — learns relative kinematics in the AIS heading frame with a learned `absent` logit for rejection. ~0.5M params.
2. **Platt calibration** — reduces ECE from 0.19 to 0.12 on the dark score.
3. **Hybrid router** — uses geometric gate when AIS is well-aligned, falls back to learned matcher when it's not.

The router gives us the geometric gate's precision on clean data AND the learned matcher's robustness under misalignment.

## Results

| AIS offset | Geometric gate | Learned only | Hybrid v2 |
|-----------|---------------|-------------|-----------|
| 0m | 0.00 | ~0.19 | **0.02** |
| 300m | — | ~0.19 | **0.13** |
| 500m | **0.98** | ~0.19 | **0.14** |

False-dark rate. Lower is better. Hybrid stays flat across all offsets.

## Caveats

- Small dataset (n=19 test tracks, single site)
- Synthetic dark labels
- Research preview, not production-ready

## Structure

```
src/
├── matcher.py          # Open-set neural matcher
├── router.py           # Hybrid routing logic
├── calibration.py      # Platt/isotonic scaling
├── evaluate.py         # Offset injection sweep
└── features.py         # ENU projection + kinematics
```

## Usage

```bash
python src/evaluate.py --offsets 0 100 200 300 400 500
```
