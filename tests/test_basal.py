"""Tests for src/events/basal.py, focused on the segment_sec parameter
added to support genuine-rest-class extraction in the W700 pipeline
(scripts/extract_w700_full.py) — see src/events/basal.py's module docstring.
"""
import numpy as np
import pytest

import config as cfg
from src.events.basal import extract_basal_epochs


def _synthetic_eeg(n_channels=16, n_samples=5000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_samples))


def test_default_segment_length_matches_config():
    eeg = _synthetic_eeg(n_samples=5000)
    events = np.array([1000, 2000, 3000])  # leaves gaps > MIN_REST_GAP_SEC
    epochs = extract_basal_epochs(eeg, events, fs=cfg.FS)
    expected_len = round(cfg.BASAL_SEGMENT_DURATION_SEC * cfg.FS)
    assert epochs.shape[1:] == (16, expected_len)


def test_custom_segment_sec_w700():
    """W700 pipeline needs 700-sample (2.8s) epochs, not the 2.0s default."""
    eeg = _synthetic_eeg(n_samples=8000)
    events = np.array([2000, 5000])  # gap before/after long enough for 700-sample epochs
    W = 700
    epochs = extract_basal_epochs(eeg, events, fs=cfg.FS, segment_sec=W / cfg.FS)
    assert epochs.ndim == 3
    assert epochs.shape[1] == 16
    assert epochs.shape[2] == W


def test_at_least_one_epoch_always_returned():
    """extract_basal_epochs falls back to the start of the recording when no
    gap qualifies — callers (build_basal_reference) rely on never getting a
    truly empty result for a long-enough recording."""
    eeg = _synthetic_eeg(n_samples=2000)
    events = np.array([100, 200, 300, 400, 500, 600])  # events packed tight, no long gaps
    epochs = extract_basal_epochs(eeg, events, fs=cfg.FS, segment_sec=1.0)
    assert epochs.shape[0] >= 1


def test_no_events_uses_whole_recording_as_one_gap():
    eeg = _synthetic_eeg(n_samples=5000)
    epochs = extract_basal_epochs(eeg, np.array([]), fs=cfg.FS, segment_sec=2.8)
    assert epochs.shape[0] >= 1
    assert epochs.shape[2] == round(2.8 * cfg.FS)
