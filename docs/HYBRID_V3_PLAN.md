# Hybrid v3: Relative Kinematics Matcher (Frenet Frame Alignment)

To achieve a 10x reduction in the false-alarm rate at high offsets (dropping the false-alarm rate from ~0.14 to < 0.02 under a 500m offset), we upgraded the matcher to use **translation-invariant relative kinematic features** (Frenet frame alignment) combined with training-time translation offset data augmentation.

---

## 1. Problem & Self-Debate

1. **The Translation Invariance Challenge:** The baseline models used separate radar and AIS encoders (dual-tower) and compared absolute ENU coordinates relative to the track mean. Under a 500m offset, the coordinates differ, and the model flags it as a mismatch, causing false alarms.

2. **Turn Artifacts:** If we project relative positions first and take derivatives, a static offset behaves like a relative velocity when the ship turns.

3. **The Solution (v3):**

   - Compute relative position offsets and their derivatives in the ENU frame **before** rotating them.
   - Project the relative position and velocity vectors into the AIS candidate's heading frame (along-track and cross-track).
   - Inject random translation offsets (up to 500m) to the true match candidate during training (Data Augmentation) to force the model to ignore distance magnitude and focus on relative velocity alignment.

---

## 2. Key Architecture Files

- **`relative_features.py`:** Calculates translation-invariant relative features (along-track, cross-track, derivative velocity components, SOG difference, COG difference).
- **`relative_model.py`:** Implements `JointRelativeMatcher`, a single-tower model that directly encodes the relative trajectory of each candidate into match logits.
- **`train_relative.py`:** Prepares training samples with 10x dataset expansion (static offset perturbations up to 500m) and trains the relative matcher.
- **`hybrid_v3_busan.py`:** The evaluation harness routing tight geometric matches to `0` (not dark) and deferring noisy matches to the calibrated `JointRelativeMatcher` dark probability.

---

## 3. How to Continue on Another GPU

To resume or run the training and evaluation on another GPU:

1. **Verify PyTorch GPU Availability:**

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Run Training and Evaluation:**

   Execute the hybrid v3 script. If the GPU has memory or queue contention, prefix it with `CUDA_VISIBLE_DEVICES=""` to run on CPU, which is highly efficient for this size:

   ```bash
   # Run on GPU (default)
   python -m eval.hybrid_v3_busan
   ```

   ```bash
   # Or run on CPU if GPU is congested
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" python -m eval.hybrid_v3_busan
   ```

3. **Check Outputs:**

   The metrics will be written to `outputs/busan_hybrid_v3.json` and a comparison chart will be saved as `outputs/fig_hybrid_v3.png`.
