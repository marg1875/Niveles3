import numpy as np
from scipy.signal import welch

import config as cfg


def _welch_psd(signal, fs):
    """Welch PSD with this module's fixed nperseg/noverlap convention.

    Factored out so compute_band_powers and compute_spectral_entropy can
    share a single Welch call per signal instead of each computing it
    independently (see compute_spectral_features).
    """
    nperseg = int(fs)
    noverlap = fs // 2
    return welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)


def compute_band_powers(signal, fs=None, psd_data=None):
    """Band powers normalized by total signal variance.

    Args:
        signal: 1D signal.
        fs: Sampling rate.
        psd_data: Optional precomputed (freqs, psd) tuple, e.g. from
            _welch_psd(signal, fs). When given, skips recomputing Welch.
    """
    if fs is None:
        fs = cfg.FS

    freqs, psd = psd_data if psd_data is not None else _welch_psd(signal, fs)

    total_var = float(np.var(signal))
    if total_var == 0:
        total_var = 1e-10

    powers = {}
    for band_name, (low, high) in cfg.SPECTRAL_BANDS.items():
        if band_name == "mu":
            continue
        mask = (freqs >= low) & (freqs <= high)
        band_power = float(np.sum(psd[mask]))
        powers[f"{band_name}_power"] = band_power / total_var

    powers["mu_power"] = powers["alpha_power"]
    return powers


def compute_erd_ratio(active_power_dict, basal_power_dict):
    erd = {}
    for band_name in cfg.SPECTRAL_BAND_NAMES:
        key = f"{band_name}_power"
        if basal_power_dict is None:
            erd[f"{band_name}_erd"] = float("nan")
        else:
            erd[f"{band_name}_erd"] = active_power_dict[key] / (basal_power_dict[key] + 1e-10)
    return erd


def compute_spectral_entropy(signal, fs=None, psd_data=None):
    """Shannon entropy of the normalized power spectral density.

    Args:
        signal: 1D signal.
        fs: Sampling rate.
        psd_data: Optional precomputed (freqs, psd) tuple — see compute_band_powers.
    """
    if fs is None:
        fs = cfg.FS

    freqs, psd = psd_data if psd_data is not None else _welch_psd(signal, fs)

    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0
    p = psd / psd_sum
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(entropy)


def compute_alpha_beta_ratio(band_powers_dict):
    return band_powers_dict["alpha_power"] / (band_powers_dict["beta_power"] + 1e-10)


def compute_mu_beta_ratio(band_powers_dict):
    # Was reading "alpha_power" — a dead duplicate of compute_alpha_beta_ratio.
    # Numerically identical today because cfg.SPECTRAL_BANDS defines mu and
    # alpha as the same (8, 12) Hz range (compute_band_powers aliases
    # mu_power = alpha_power), but this reads the field it's actually named
    # for, so a future change to the mu band's range is picked up correctly.
    return band_powers_dict["mu_power"] / (band_powers_dict["beta_power"] + 1e-10)


def compute_spectral_features(segment, fs=None, basal_segment=None, basal_band_powers=None):
    """Full spectral feature set (band powers, ERD ratios, entropy, ratios) for one segment.

    Args:
        segment: (n_channels, n_samples) or 1D spatially-averaged signal.
        fs: Sampling rate.
        basal_segment: Optional (n_channels, n_samples) or 1D basal reference
            signal. Its band powers are recomputed from scratch on every call
            — fine for a one-off call, but wasteful if the same basal segment
            is reused across many active segments from the same file (as in
            scripts/extract_w700_full.py). Prefer `basal_band_powers` for that.
        basal_band_powers: Optional pre-computed band-power dict (from
            compute_band_powers on the basal reference) for the ERD ratio.
            Takes precedence over `basal_segment` when both are given — lets
            callers compute the basal Welch/band-power once per file instead
            of once per segment.
    """
    if fs is None:
        fs = cfg.FS

    if segment.ndim == 2:
        segment = np.mean(segment, axis=0)

    psd_data = _welch_psd(segment, fs)
    band_powers = compute_band_powers(segment, fs=fs, psd_data=psd_data)
    spectral_entropy = compute_spectral_entropy(segment, fs=fs, psd_data=psd_data)
    alpha_beta_ratio = compute_alpha_beta_ratio(band_powers)
    mu_beta_ratio = compute_mu_beta_ratio(band_powers)

    features = {}
    features.update(band_powers)
    features["spectral_entropy"] = spectral_entropy
    features["alpha_beta_ratio"] = alpha_beta_ratio
    features["mu_beta_ratio"] = mu_beta_ratio

    if basal_band_powers is not None:
        erd = compute_erd_ratio(band_powers, basal_band_powers)
    elif basal_segment is not None:
        if basal_segment.ndim == 2:
            basal_segment = np.mean(basal_segment, axis=0)
        basal_band_powers = compute_band_powers(basal_segment, fs=fs)
        erd = compute_erd_ratio(band_powers, basal_band_powers)
    else:
        erd = compute_erd_ratio(band_powers, None)

    features.update(erd)
    return features
