"""Demo showing how to instantiate the JointRelativeMatcher model and run open-set dark vessel detection.

Usage:
    PYTHONPATH=src python examples/demo.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from p1_openset_darkdet.relative_model import JointRelativeMatcher

def run_demo():
    print("=== Hybrid-v3 Open-Set Matcher Demo ===")
    
    # 1. Instantiate the model
    # Features include: relative East offset, North offset, East velocity, North velocity, SOG difference, COG difference
    in_dim = 6 
    emb_dim = 128
    model = JointRelativeMatcher(in_dim=in_dim, emb=emb_dim)
    model.eval()
    print("✓ Model instantiated successfully.")
    print(f"  - Input dimensions: {in_dim}")
    print(f"  - Embedding size: {emb_dim}")
    print(f"  - Initial absent logit (open-set threshold): {model.absent.item():.4f}")

    # 2. Mock inputs for 3 candidates (K=3) over a trajectory of length 20 (L=20)
    K = 3
    L = 20
    
    # Let's create dummy relative features (K, L, 6)
    # - Candidate 0: Close match (small relative differences)
    # - Candidate 1: High offset match (medium relative differences)
    # - Candidate 2: Completely mismatched kinematic behavior
    cand_relative_feats = torch.randn(K, L, in_dim) * 0.1
    cand_relative_feats[2] += 5.0 # Mismatch candidate has large features
    
    # Target masks: True for active points along the track, False for padded
    cand_mask = torch.ones(K, L, dtype=torch.bool)
    
    # Candidate validity: True if candidate slot is populated
    cand_valid = torch.tensor([True, True, True], dtype=torch.bool)

    print("\nRunning inference...")
    with torch.no_grad():
        # Get raw logits (K + 1 elements: K candidates + 1 absent/reject option)
        logits = model.match_logits(cand_relative_feats, cand_mask, cand_valid)
        
        # Calculate probabilities using softmax
        probs = F.softmax(logits, dim=0)
        
        # Calculate P(dark) directly
        p_dark = model.dark_score(cand_relative_feats, cand_mask, cand_valid)

    # 3. Print results
    print("\n=== Matching Results ===")
    for k in range(K):
        print(f"Candidate {k}:")
        print(f"  - Match Logit: {logits[k].item():.4f}")
        print(f"  - Match Prob:  {probs[k].item():.4%}")
        
    print(f"\nOpen-Set 'Absent' (Dark Vessel) Option:")
    print(f"  - Absent Logit: {logits[-1].item():.4f}")
    print(f"  - P(Dark Vessel): {p_dark:.4%}")
    
    if p_dark > 0.5:
        print("\nDecision: [High Confidence] DARK VESSEL DETECTED (No AIS candidate matches the radar track).")
    else:
        best_candidate = torch.argmax(probs[:-1]).item()
        print(f"\nDecision: Associated with Candidate {best_candidate} (Radar track matches AIS broadcast).")

if __name__ == "__main__":
    run_demo()
