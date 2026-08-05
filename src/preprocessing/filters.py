"""
EEG signal filtering: DC removal, Notch (60Hz), Bandpass (0.5-45Hz).
"""
from functools import lru_cache

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

import config as cfg


@lru_cache(maxsize=None)
def _notch_coeffs(fs, f_notch, q):
    """Cached iirnotch design — fs/f_notch/q are constant across the whole
    pipeline (config.py), so there's no reason to redesign the filter on
    every call (once per file, across every extraction script)."""
    return iirnotch(f_notch, Q=q, fs=fs)


@lru_cache(maxsize=None)
def _bandpass_coeffs(fs, f_low, f_high, order):
    """Cached Butterworth bandpass design — see _notch_coeffs."""
    return butter(order, (f_low, f_high), btype="bandpass", fs=fs)


def remove_dc(signals: np.ndarray) -> np.ndarray:
    """Remove DC offset per channel (no variance scaling)."""
    return signals - np.mean(signals, axis=1, keepdims=True)


def z_score_normalize(signals: np.ndarray) -> np.ndarray:
    """Z-score normalization per channel (zero mean, unit variance).

    Applied ONCE before fractal feature extraction to make channels comparable.
    NOT applied for ERD amplitude analysis (which requires preserved amplitudes).
    """
    centered = signals - np.mean(signals, axis=1, keepdims=True)
    return centered / (np.std(centered, axis=1, keepdims=True) + 1e-10)


def notch_filter(signals: np.ndarray, fs: float = None, f_notch: float = None,
                 q: float = None) -> np.ndarray:
    """Apply 60 Hz notch filter to remove power line interference."""
    fs = fs or cfg.FS
    f_notch = f_notch or cfg.F_NOTCH
    q = q or cfg.NOTCH_Q

    b, a = _notch_coeffs(fs, f_notch, q)
    return filtfilt(b, a, signals, axis=1)


def bandpass_filter(signals: np.ndarray, fs: float = None,
                    f_low: float = None, f_high: float = None,
                    order: int = None) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    fs = fs or cfg.FS
    f_low = f_low or cfg.F_LOW
    f_high = f_high or cfg.F_HIGH
    order = order or cfg.BANDPASS_ORDER

    b, a = _bandpass_coeffs(fs, f_low, f_high, order)
    return filtfilt(b, a, signals, axis=1)


def preprocess_eeg(signals: np.ndarray, fs: float = None) -> np.ndarray:
    """Full preprocessing pipeline: DC removal → Notch → Bandpass.

    Returns cleaned EEG without Z-score (add z_score_normalize later if needed).
    """
    signals = np.nan_to_num(signals)
    signals = remove_dc(signals)
    signals = notch_filter(signals, fs=fs)
    signals = bandpass_filter(signals, fs=fs)
    return signals


def mu_beta_filter(signals: np.ndarray, fs: float = None) -> np.ndarray:
    """Extract mu/beta band (8-30 Hz) for ERD detection.

    Shares its cached filter design (_bandpass_coeffs) with
    src.events.detector.detect_erd_events, which needs the identical
    mu/beta Butterworth filter applied to a single channel — previously
    each independently called butter(4, cfg.MU_BAND, ...).
    """
    fs = fs or cfg.FS
    low, high = cfg.MU_BAND
    b, a = _bandpass_coeffs(fs, low, high, 4)
    return filtfilt(b, a, signals, axis=1)
