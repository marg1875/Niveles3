"""Tests for src/features/spectral.py.

Golden-fixture regression protects the Fase 1 refactor that consolidates the
two redundant scipy.signal.welch() calls (band power + entropy) into one.
"""
import os

import numpy as np
import pytest

import config as cfg
from src.features.spectral import (
    compute_alpha_beta_ratio,
    compute_band_powers,
    compute_spectral_entropy,
    compute_spectral_features,
)

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _white_noise_2d(n_channels=16, n_samples=700, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_samples))


def test_band_powers_keys_present():
    sig = _white_noise_2d()
    powers = compute_band_powers(np.mean(sig, axis=0), fs=cfg.FS)
    for band in ("delta", "theta", "alpha", "mu", "beta", "gamma"):
        assert f"{band}_power" in powers


def test_mu_power_equals_alpha_power_given_current_config():
    """mu and alpha bands are both (8, 12) Hz in config.py, so their power
    is defined to be identical by construction (spectral.py:27)."""
    sig = _white_noise_2d()
    powers = compute_band_powers(np.mean(sig, axis=0), fs=cfg.FS)
    assert powers["mu_power"] == powers["alpha_power"]


def test_spectral_entropy_bounds():
    sig = _white_noise_2d()
    entropy = compute_spectral_entropy(np.mean(sig, axis=0), fs=cfg.FS)
    assert entropy >= 0.0


def test_alpha_beta_ratio_positive_for_noise():
    sig = _white_noise_2d()
    powers = compute_band_powers(np.mean(sig, axis=0), fs=cfg.FS)
    ratio = compute_alpha_beta_ratio(powers)
    assert ratio > 0


def test_compute_spectral_features_erd_nan_without_basal():
    sig = _white_noise_2d()
    feats = compute_spectral_features(sig, fs=cfg.FS, basal_segment=None)
    assert np.isnan(feats["delta_erd"])


# ---------------------------------------------------------------------------
# Golden-fixture regression
# ---------------------------------------------------------------------------

def _regen_signal(name, seed=42):
    rng = np.random.default_rng(seed)
    signals = {}
    for n in (500, 700, 2000):
        signals[f"fbm_{n}"] = np.cumsum(rng.standard_normal((16, n)), axis=1)
    signals["white_700"] = rng.standard_normal((16, 700))
    t = np.arange(700)
    signals["trend_700"] = (np.tile(t * 0.01, (16, 1))
                             + rng.standard_normal((16, 700)) * 0.1)
    return signals[name]


@pytest.mark.skipif(not os.path.exists(os.path.join(FIXTURES_DIR, "spectral_golden.npz")),
                     reason="golden fixture not generated yet")
@pytest.mark.parametrize("name", ["fbm_500", "fbm_700", "fbm_2000", "white_700", "trend_700"])
def test_compute_spectral_features_matches_golden(name):
    golden = np.load(os.path.join(FIXTURES_DIR, "spectral_golden.npz"), allow_pickle=True)
    keys = golden[f"{name}__keys"]
    values = golden[f"{name}__values"]

    sig = _regen_signal(name)
    basal = sig[:, :250] if sig.shape[1] > 250 else None
    feats = compute_spectral_features(sig, fs=cfg.FS, basal_segment=basal)

    live_values = np.array([feats[k] for k in keys], dtype=float)
    np.testing.assert_allclose(live_values, values, rtol=1e-6, atol=1e-8, equal_nan=True,
                                err_msg=f"compute_spectral_features drifted for {name} "
                                        "vs tests/fixtures/spectral_golden.npz")
