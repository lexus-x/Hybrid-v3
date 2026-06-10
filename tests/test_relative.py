"""Unit tests for relative feature extraction in src/p1_openset_darkdet/relative_features.py.
"""
from __future__ import annotations

import numpy as np
import pytest
from p1_openset_darkdet.relative_features import compute_relative_features

def test_compute_relative_features_empty():
    rt = np.array([])
    rlon = np.array([])
    rlat = np.array([])
    rsog = np.array([])
    rcog = np.array([])
    
    at = np.array([])
    alon = np.array([])
    alat = np.array([])
    asog = np.array([])
    acog = np.array([])
    
    feats, mask = compute_relative_features(rt, rlon, rlat, rsog, rcog, at, alon, alat, asog, acog)
    assert feats.shape == (32, 6)
    assert not mask.any()

def test_compute_relative_features_valid():
    # 5 radar points
    rt = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    rlon = np.array([129.0, 129.001, 129.002, 129.003, 129.004])
    rlat = np.array([35.0, 35.001, 35.002, 35.003, 35.004])
    rsog = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    rcog = np.array([45.0, 45.0, 45.0, 45.0, 45.0])
    
    # 3 AIS points (need interpolation)
    at = np.array([0.0, 20.0, 40.0])
    alon = np.array([129.0, 129.002, 129.004])
    alat = np.array([35.0, 35.002, 35.004])
    asog = np.array([10.0, 10.0, 10.0])
    acog = np.array([45.0, 45.0, 45.0])
    
    feats, mask = compute_relative_features(rt, rlon, rlat, rsog, rcog, at, alon, alat, asog, acog)
    
    # Check shape
    assert feats.shape == (32, 6)
    assert mask.sum() == 5
    
    # Check that relative offsets (along/cross) are near 0 since the trajectories are perfectly aligned
    assert abs(feats[0, 0]) < 1e-3
    assert abs(feats[0, 1]) < 1e-3
    # Check SOG and COG difference scaling (should be 0 since they are aligned)
    assert feats[0, 4] == 0.0
    assert feats[0, 5] == 0.0
