import numpy as np
from scipy.signal import welch

import config as cfg


def compute_band_powers(signal, fs=None):
    if fs is None:
        fs = cfg.FS
    nperseg = int(fs)
    noverlap = fs // 2

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

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


def compute_spectral_entropy(signal, fs=None):
    if fs is None:
        fs = cfg.FS
    nperseg = int(fs)
    noverlap = fs // 2

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

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
    return band_powers_dict["alpha_power"] / (band_powers_dict["beta_power"] + 1e-10)


def compute_spectral_features(segment, fs=None, basal_segment=None):
    if fs is None:
        fs = cfg.FS

    if segment.ndim == 2:
        segment = np.mean(segment, axis=0)

    band_powers = compute_band_powers(segment, fs=fs)
    spectral_entropy = compute_spectral_entropy(segment, fs=fs)
    alpha_beta_ratio = compute_alpha_beta_ratio(band_powers)
    mu_beta_ratio = compute_mu_beta_ratio(band_powers)

    features = {}
    features.update(band_powers)
    features["spectral_entropy"] = spectral_entropy
    features["alpha_beta_ratio"] = alpha_beta_ratio
    features["mu_beta_ratio"] = mu_beta_ratio

    if basal_segment is not None:
        if basal_segment.ndim == 2:
            basal_segment = np.mean(basal_segment, axis=0)
        basal_band_powers = compute_band_powers(basal_segment, fs=fs)
        erd = compute_erd_ratio(band_powers, basal_band_powers)
    else:
        erd = compute_erd_ratio(band_powers, None)

    features.update(erd)
    return features
