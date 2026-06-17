"""
EEG signal filtering: DC removal, Notch (60Hz), Bandpass (0.5-45Hz).
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

import config as cfg


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

    b, a = iirnotch(f_notch, Q=q, fs=fs)
    return filtfilt(b, a, signals, axis=1)


def bandpass_filter(signals: np.ndarray, fs: float = None,
                    f_low: float = None, f_high: float = None,
                    order: int = None) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    fs = fs or cfg.FS
    f_low = f_low or cfg.F_LOW
    f_high = f_high or cfg.F_HIGH
    order = order or cfg.BANDPASS_ORDER

    b, a = butter(order, [f_low, f_high], btype="bandpass", fs=fs)
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
    """Extract mu/beta band (8-30 Hz) for ERD detection."""
    fs = fs or cfg.FS
    band = cfg.MU_BAND
    b, a = butter(4, band, btype="bandpass", fs=fs)
    return filtfilt(b, a, signals, axis=1)
