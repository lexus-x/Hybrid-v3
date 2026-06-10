"""Unit tests for geodetic utilities in src/common/geo.py.
"""
from __future__ import annotations

import numpy as np
import pytest
from common.geo import haversine_m, angdiff_deg, enu_offset_m

def test_haversine_m():
    # 0 distance between same point
    assert haversine_m(120.0, 30.0, 120.0, 30.0) == 0.0
    
    # Distance between two distinct points (approximate values)
    # Busan site coordinates (approx 129.0, 35.0)
    d = haversine_m(129.0, 35.0, 129.01, 35.01)
    assert d > 0.0
    assert abs(d - 1420.0) < 50.0  # roughly 1.4 km

def test_angdiff_deg():
    # Identical angles
    assert angdiff_deg(45.0, 45.0) == 0.0
    # Modulo differences
    assert angdiff_deg(0.0, 360.0) == 0.0
    assert angdiff_deg(350.0, 10.0) == 20.0
    assert angdiff_deg(180.0, 0.0) == 180.0
    assert angdiff_deg(90.0, 270.0) == 180.0
    
    # Check with array inputs
    a = np.array([0.0, 350.0])
    b = np.array([360.0, 10.0])
    diff = angdiff_deg(a, b)
    np.testing.assert_array_almost_equal(diff, [0.0, 20.0])

def test_enu_offset_m():
    # 0 offset
    e, n = enu_offset_m(129.0, 35.0, 129.0, 35.0)
    assert e == 0.0
    assert n == 0.0
    
    # Positive East/North offsets
    e, n = enu_offset_m(129.0, 35.0, 129.01, 35.01)
    assert e > 0.0
    assert n > 0.0
