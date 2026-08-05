"""One-off script: generates the golden regression fixture for tests/test_fractal.py
and tests/test_spectral.py, using the CURRENT (pre-refactor) implementation.

Run once, before any optimization work touches src/features/fractal.py or
src/features/spectral.py. The resulting .npz files are committed to the repo
so future test runs can diff against them with np.allclose, guaranteeing
that performance refactors do not change any numeric output.

Not part of the pipeline — delete after the refactor is verified, or keep
as documentation of how the golden fixtures were produced.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.features.fractal import compute_all_fractals
from src.features.spectral import compute_spectral_features

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "tests", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

# ---- Signals: fixed seeds, deterministic, cover the shapes actually used
# by the W700 pipeline (16 channels x 700 samples) plus a couple of edge
# lengths (500 and 2000, the sweep extremes).
rng = np.random.default_rng(42)

signals = {}

# fBm-like (cumsum of gaussian noise) -- typical EEG-ish autocorrelated signal
for n_samples in (500, 700, 2000):
    sig = np.cumsum(rng.standard_normal((16, n_samples)), axis=1)
    signals[f"fbm_{n_samples}"] = sig

# White noise, H should land near 0.5
signals["white_700"] = rng.standard_normal((16, 700))

# Strong linear trend + noise
t = np.arange(700)
trend = np.tile(t * 0.01, (16, 1)) + rng.standard_normal((16, 700)) * 0.1
signals["trend_700"] = trend

results = {}
for name, sig in signals.items():
    feats = compute_all_fractals(sig, fs=cfg.FS)
    results[name] = feats
    print(name, sig.shape, "->", feats.shape)

out_path = os.path.join(FIXTURES_DIR, "fractal_golden.npz")
np.savez(out_path, **results)
print("Saved:", out_path)

# ---- Spectral golden fixture
spectral_results = {}
for name, sig in signals.items():
    basal = sig[:, :250] if sig.shape[1] > 250 else None
    feats = compute_spectral_features(sig, fs=cfg.FS, basal_segment=basal)
    # Flatten dict of floats into a stable-order array for np.savez
    keys = sorted(feats.keys())
    spectral_results[name + "__keys"] = np.array(keys)
    spectral_results[name + "__values"] = np.array([feats[k] for k in keys], dtype=float)

out_path2 = os.path.join(FIXTURES_DIR, "spectral_golden.npz")
np.savez(out_path2, **spectral_results)
print("Saved:", out_path2)
