"""Smoke test of the scipy reference on the MATLAB example design (matlab/main_disorder.m).

    cd scipy_ref && python test_main.py

Expected output: Maximum KE ratio: ~0.148100 (MATLAB: 0.1481001164). Takes ~10 s.
"""
from main_disorder_v2 import main_disorder_v2


c2 = [-70, -100, -5, -35]
c3 = [1200, 2000, 1500, 3000]

result = main_disorder_v2(c2, c3)

print("Maximum KE ratio:", result)