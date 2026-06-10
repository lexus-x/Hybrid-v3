"""Unit tests for the Hybrid Router logic in src/eval/hybrid_v3_busan.py.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from eval.hybrid_v3_busan import _confident_match
from data.pair_builder import GateCfg

def test_confident_match():
    # Setup mock AIS data
    ais = pd.DataFrame({
        "mmsi": [111, 222],
        "lon": [129.0, 129.01],
        "lat": [35.0, 35.01],
        "cog": [45.0, 90.0],
        "sog": [10.0, 15.0],
        "t": [100.0, 100.0]
    })
    
    cfg = GateCfg()
    
    # Mock radar track points
    rlon = np.array([129.0])
    rlat = np.array([35.0])
    rsog = np.array([10.0])
    rcog = np.array([45.0])
    rt = np.array([100.0])
    
    # Test case 1: Perfect match (candidate 111 is a confident match)
    assert _confident_match(ais, [111, 222], rlon, rlat, rsog, rcog, rt, {}, 35.0, cfg) == True
    
    # Test case 2: Candidate 222 (mismatched kinematics SOG/COG and distance)
    # Target only candidate 222
    assert _confident_match(ais, [222], rlon, rlat, rsog, rcog, rt, {}, 35.0, cfg) == False
    
    # Test case 3: Match candidate is not in candidate list
    assert _confident_match(ais, [333], rlon, rlat, rsog, rcog, rt, {}, 35.0, cfg) == False
    
    # Test case 4: Match exists but offsets push it out of tight range
    assert _confident_match(ais, [111], rlon, rlat, rsog, rcog, rt, {111: (500.0, 500.0)}, 35.0, cfg) == False
