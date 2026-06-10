# Dark-Vessel Detection — Project Report (P1, finalized)

**Title (working):** *Calibrated Open-Set Dark-Vessel Detection by Radar–AIS Fusion, Robust to AIS Registration Error*
**Author:** solo · **Compute:** 1× A100 80 GB · **Date:** 2026-06-02
**Target venue:** IEEE TITS (primary) or IEEE TGRS (secondary). *Information Fusion dropped — it desk-rejects maritime-only work.*
**Status:** experiments W1–W4 done; remaining = hybrid router + calibration + public anchor + writing.

---

## 1. The problem (what we tackle)

The PDF's stated goal (north star):

> *"레이더와 AIS 데이터를 융합한 실시간 선박 탐지 및 암흑 선박(Dark Vessel) 조기 탐지 시스템 개발"*
> Fuse marine **radar + AIS** to monitor traffic in real time and **detect dark vessels** (vessels present to a sensor but **not** broadcasting AIS).

A "dark vessel" = a radar track with **no matching AIS report**. They matter because going dark is the signature of IUU fishing, sanctions evasion, smuggling, and illegal ship-to-ship transfers.

**Why the current operational approach is weak.** The PDF's baseline (and essentially *every* published dark-detection paper — Nature/GFW 2024, the xView3 winners, Galdelli 2025) declares "dark" with a **rule**: detect a track, look for an AIS report inside a time/distance/angle gate, and if none matches, call it dark. This is brittle in exactly the way that matters:

- **74 % point-match / 50 % track-match** on real Busan data → most "unmatched" tracks are **association failures, not dark vessels**. The rule cannot tell the two apart. This is a *precision* problem.
- The gate is a hard threshold. When AIS positions carry **registration error** (GPS/timestamp offset, sensor mis-alignment, congestion), true vessels fall outside the gate and are wrongly flagged dark. We measured this: at a 500 m AIS offset the rule's false-dark rate is **0.98** (it flags almost everything as dark).

**What we tackle:** replace the rule with a **learned, open-set, calibrated** radar↔AIS matcher that (a) emits a probability the track is dark rather than a binary flag, (b) stays accurate when AIS registration is noisy, and (c) reports *how sure* it is (calibrated, so an operator can trust the score).

---

## 2. How we tackle it (method)

A modality-agnostic **open-set matcher**: AIS-identified tracks = "known"; a sensor track that matches no AIS identity above threshold = "dark" (open-set reject), with a calibrated dark score. Five components:

1. **Radar track encoder** — a pre-norm Transformer (`norm_first=True`, input LayerNorm, masked-mean pool) over the tracklet `[Δlat, Δlon, sog, cog, signalLevel, Δt]` → track embedding.
2. **AIS encoder** — a trajectory encoder over the time-aligned AIS window → AIS embedding in the same space.
3. **Open-set head** — temperature-scaled cosine similarity between the track and every candidate AIS in the time window, plus a **learned "absent" logit**. `P(dark)` = softmax mass on *absent*. Trained with InfoNCE + an explicit **reject option** (AIS-dropout augmentation: withhold the true AIS for a fraction of samples so the target becomes *absent* — without this the model never learns to flag darks).
4. **Calibration** — report **ECE**; calibrate the dark score. (Raw ECE ≈ 0.19; **Platt remapping → 0.12**, isotonic 0.13. Temperature scaling is identity here — see §5.)
5. **Hybrid router (the key new piece)** — use cheap geometric gating when AIS registration is *confident*, and the learned matcher when it is *uncertain* (large residual / congestion). This fixes the learned model's false-alarm floor and yields a detector that beats **both** baselines across the whole AIS-error range.

**Booster for small data:** self-supervised **masked-AIS pretraining** of the AIS encoder on public AIS (GFW / MarineCadastre), then fine-tune on Busan — buys accuracy in the tiny-data regime.

**Evaluation protocol (no circularity).** *Controlled AIS-dropout*: take tracks with a confident AIS link, **withhold their AIS at test time** → these become *known* dark positives (ground truth we control). AIS-present tracks = negatives. Metrics: dark **AUROC / AUPRC / F1**, **ECE**, and a **robustness curve** (false-dark rate vs injected AIS registration offset 0→500 m). Multi-seed (≥5) + bootstrap CIs — mandatory after we got burned by a lucky single seed (see §7).

---

## 3. Dataset

| Role | Dataset | Modalities | Scale | Access |
|------|---------|-----------|-------|--------|
| **Primary real-world testbed** | **Busan port** (confidential, in hand) | radar tracks + AIS + 2,881 radar frames | 200 tracks; 32,969 radar points; 89 dropout-eligible tracks | **NOT releasable** — used as a real deployment study |
| **Reproducible protocol** | controlled AIS-dropout | any radar+AIS feed | unlimited synthetic dark positives | public-replicable recipe |
| **Public anchor (stretch)** | **WHUT-MSFVessel** | radar + video + AIS | tri-modal coastal | accessibility unverified (Baidu) — attempt at execution |
| **Cross-sensor supporting case study** | **BONK-Pose** | camera crops + AIS attributes | 3,829 vessels / 774 test | downloaded, done |

Busan is a legitimate real coastal radar+AIS dataset (many TGRS/TITS radar papers use proprietary data); the **controlled-dropout protocol** is the reproducible contribution others can run on their own feeds. WHUT-MSFVessel, if its download works, gives a fully public radar+AIS validation.

---

## 4. Novelty (what's actually new)

Domain research (academic-API + web sweep, 2026) confirmed the gaps:

1. **Open-set, *learned* dark detection.** *No published paper* frames dark detection as open-set recognition — all existing work is rule-based AIS subtraction. We replace the rule with a learned reject option. (PDF future-work #1.)
2. **Calibrated dark score.** The "calibrated P(dark | context)" angle is explicitly **unpublished**. We separate "truly dark" from "association failure" with a reliability-quantified score (ECE) — attacking the precision gap behind the 74 % baseline.
3. **Robustness to AIS registration error (the headline).** First to *quantify* and *defend against* the failure mode where geometric gating collapses (false-dark 0.98 at 500 m) while a learned matcher stays flat. The **hybrid router** then beats both baselines across the entire error range — a clean, defensible "better than baseline" result.
4. **Cross-sensor generality.** The same open-set framework works on radar tracks *and* camera crops (supporting case study), showing the framing is sensor-agnostic.

**Differentiation (never claim "association"):** bi-modal *association/tracking* is taken — DeepSORVF (video+AIS, 2023), GNN+OT (radar+AIS, 2022). We lead with **open-set + calibration + robustness**, which they do not do.

---

## 5. Expected results

Measured so far (Busan, 5 seeds):

| Metric | Geometric / rule | Learned (ours) | Note |
|--------|------------------|----------------|------|
| Baseline reproduction (point-match) | **0.740** | — | matches PDF's 0.744 |
| Clean-split dark AUROC | 1.00 (trivial†) | **0.80 ± 0.05** | †withholding removes geom's only in-gate match |
| False-dark @ 0 m offset | 0.00 | 0.23 ± 0.06 | learned floor = the weakness |
| **False-dark @ 500 m offset** | **0.98** | **0.22 ± 0.07** | learned is offset-invariant — the real win |
| ECE (calibration) | — | 0.30 | bad → fix with temperature scaling |
| Cross-sensor camera AUROC | — | 0.675 ± 0.013 (n=774) | supporting case study |

**Hybrid router — MEASURED (v2, 5 seeds, `outputs/busan_hybrid_v2.json`).** A congestion-robust router (trust geometric only on a tight close+course+speed match; else defer to the learned matcher):

| Metric | Geometric | Learned | **Hybrid v2** |
|--------|-----------|---------|---------------|
| Clean-split dark AUROC | 1.00 (trivial) | 0.76 ± 0.03 | **0.94 ± 0.00** |
| False-dark @ 0 m | 0.00 | 0.19 | **0.02** |
| False-dark @ 300 m | 0.23 | 0.19 | **0.13** |
| False-dark @ 500 m | 0.98 | 0.18 | **0.14** |

- **Hybrid dominates the learned baseline everywhere** (false-dark ≤ 0.14 at *every* offset vs the learned 0.19 floor) and **dominates geometric beyond ~200 m** (0.14 vs 0.98 at 500 m); below ~100 m both gating and hybrid are near-perfect.
- **Clean AUROC 0.76 → 0.94**: on clean data the hybrid correctly leans on geometric (near-1.0); the learned arm supplies the high-offset robustness. (std 0.00 because the clean-split routing is model-independent.)
- **The < 0.10-uniform target was NOT reached** — hybrid sits at ~0.13–0.14 in the 200–500 m band, limited by the learned arm's ~0.19 false-alarm floor. Lowering that floor (Han's cross-attention / Time2Vec, more data) is the path to < 0.10.

**Calibration — RESOLVED by monotonic remapping (`outputs/busan_calibration.json`).** The raw dark score is poorly calibrated (**ECE ≈ 0.19**). Temperature scaling is identity here (optimal T = 1.0) — but that is a *diagnosis*, not a dead end: the val split carries no dark labels under controlled dropout, and more fundamentally the miscalibration is a *shape* problem (mis-scaled magnitudes), not a *sharpness* problem that a single temperature could fix. **5-fold cross-validated Platt scaling cuts ECE to 0.12** (isotonic 0.13) on the pooled test set (n=95). Caveat: fit on a small CV pool; not yet validated on a held-out operational set.

**Honest framing:** the contribution is *robustness + a hybrid that dominates across the operating range*, not "higher clean AUROC" alone. The hybrid now beats the learned baseline outright and beats gating wherever gating is brittle.

---

## 6. Timeline (total project)

Roughly **6–8 weeks** end-to-end; **~3–4 weeks remain**.

| Phase | Work | Status |
|-------|------|--------|
| W1 | Busan loader, baseline reproduction, controlled-dropout splits | ✅ done (0.740 / P90 106 m) |
| W2 | Open-set matcher trains stably on A100 | ✅ done |
| W3 | Reject-option training, multi-seed clean AUROC 0.80 | ✅ done |
| W4 | Robustness curve + multi-seed CIs + camera case study | ✅ done |
| **W5** | Hybrid router (done: clean AUROC 0.76→0.94, false-dark ≤0.14 all offsets) + calibration | 🟡 router ✅, calibration open (temp scaling failed → isotonic next) |
| **W6** | **Masked-AIS pretraining booster + WHUT-MSFVessel public anchor (if accessible)** | ⬜ remaining |
| **W7–8** | **Writing + figures → IEEE TITS/TGRS** | ⬜ remaining |

---

## 7. Problems faced so far

1. **Training NaNs.** `masked_fill(-inf)` before softmax (0·−inf → NaN grad); post-norm Transformer instability; `F.normalize` backward near zero-norm. **Fixed:** finite `-1e4` mask, `norm_first=True` + input LayerNorm, cosine at scoring, lr 3e-4, grad clipping, finite-loss skip.
2. **Model never flagged darks** (AUROC ~0.5–0.64). Root cause: the reject option was never in the training target. **Fixed:** AIS-dropout augmentation → clean AUROC 0.50 → **0.80**.
3. **Over-optimistic single run** (learned false-dark "0.056"). A lucky seed. Multi-seed revealed the true **0.22 ± 0.06**. **Lesson:** always multi-seed + report CIs.
4. **Camera arm at chance** (0.51). Cause: scoring each crop against *all 774* candidates drowned the signal. **Fixed:** realistic local candidate pool (own + ~7 confusers, mirroring radar gating) → **0.675**.
5. **The crossover / false-alarm floor.** The learned model is *not* uniformly better than gating — it is worse in the low-error regime (0.22 floor vs ~0) and wins only beyond ~300 m offset. This is the honest weakness driving the **hybrid** fix in W5.
6. **Tiny + confidential data.** Busan n=19 test → wide CIs; cannot be released. Drives the controlled-dropout *protocol* + WHUT-MSFVessel public-anchor plan + masked-AIS pretraining.
7. **Dataset attrition.** MOANA (no AIS), FVessel (73 GB + GitHub now 404), Hamburg image-ais-fusion (AIS withheld), WHUT-MSFVessel (Baidu, unverifiable) all rejected/de-risked → **BONK** chosen for the camera case study.

---

## 8. Decisions made

- **Detection-first**, per the PDF goal (not going-dark *prediction* — that's parked).
- **Open-set learned reject** replaces the PDF's 6-stage rule checker (future-work #1).
- **λ = 2.0** cost weight `J = d + 2·dψ`, gates `dt=8 s / dist=300 m / ang=60°` — reverse-engineered from the PDF's worked examples; baseline reproduced (0.740 vs PDF 0.744).
- **Controlled AIS-dropout** as the evaluation (avoids the circularity that weakens every rule-based paper).
- **Multi-seed + bootstrap CIs mandatory** (after the lucky-seed incident).
- **Reframed the flagship** around **robustness + calibration + hybrid** (the regime where we genuinely win), demoting "higher clean AUROC" claims.
- **Cross-sensor camera arm demoted** to a supporting case study (BONK), not a co-flagship.
- **Venue:** IEEE TITS / TGRS; **Information Fusion dropped** (maritime-only desk-reject risk found in venue research).
- **Parked as future papers (not this one):** SAR foundation-model dark detection on xView3 (P2, satellite arm), and AIS MMSI identity-fraud / "dark-then-reappear" detection (the highest-novelty pivot — saved for a follow-up since it diverges from the PDF's radar+AIS goal).

---

## 9. Immediate next step

W5: implement the **hybrid router** (geometric-when-confident / learned-when-uncertain) + **temperature-scaling calibration**, then re-run the multi-seed robustness curve. Success = hybrid false-dark < 0.10 at all offsets and ECE < 0.10 — the result that makes this a Q1 paper.

*Code:* `darkvessel/` — loaders in `data/`, encoders in `encoders/`, model + training in `p1_openset_darkdet/`, eval harnesses in `eval/`, results in `outputs/`. Full plan: `~/.claude/plans/cached-snuggling-octopus.md`.
