"""
Adaptive fractal feature extraction for EEG.

Implements seven fractal methods with parameters that automatically adjust
to the epoch length (N samples). Each method returns a single value per
1D signal segment.

Methods:
1. R/S (Rescaled Range) — Hurst exponent via rescaled range analysis
2. Higuchi — Fractal dimension via curve-length across subdivisions
3. DFA (Detrended Fluctuation Analysis) — Long-range correlation exponent
4. Variogram — Hurst exponent from semivariance vs lag slope
5. HO (Hurst Original) — Classic R/S without detrending (Martinez-Peon 2024)
6. HRS (Hurst Rescaled Range partitions) — R/S with subseries pooling
7. HRS Vector — HRS across multiple partition values

All methods are normalized per channel. For multi-channel EEG, call
compute_all_fractals() with the full array.
"""
import numpy as np

import config as cfg


def compute_all_fractals(segment: np.ndarray, fs: float = None) -> np.ndarray:
    """Compute all 6 fractal features + average + HV for one segment.

    Args:
        segment: (n_channels, n_samples) EEG segment.
        fs: Sampling rate.

    Returns:
        (n_channels, 8) array: [RS, Higuchi, DFA, Variogram, Average, HO, HRS_p64, HV]
    """
    fs = fs or cfg.FS
    n_channels, n_samples = segment.shape

    results = np.zeros((n_channels, 8))
    for ch in range(n_channels):
        signal = segment[ch, :]
        rs = compute_rs(signal, fs=fs)
        hig = compute_higuchi(signal, fs=fs)
        dfa = compute_dfa(signal, fs=fs)
        var = compute_variogram(signal, fs=fs)
        ho = compute_ho(signal, fs=fs)
        hrs = compute_hrs(signal, fs=fs, p=cfg.BEST_HRS_PARTITION)

        values = [rs, hig, dfa, var]
        avg = np.nanmean(values)
        results[ch, :] = [rs, hig, dfa, var, avg, ho, hrs, var]

    return results


def compute_rs(signal: np.ndarray, fs: float = None) -> float:
    """R/S Hurst exponent with adaptive scales.

    Scale range adapts to signal length:
    - Minimum scale: max(32, fs * 0.128) samples
    - Maximum scale: N * 0.40 samples
    - Requires at least 5 valid scale points with ≥ 5 blocks each for log-log fit

    Args:
        signal: 1D float array of EEG data.
        fs: Sampling rate for scale calibration.

    Returns:
        Hurst exponent H ∈ [0,1] or NaN if insufficient data.
    """
    fs = fs or cfg.FS
    s = np.asarray(signal, dtype=float).flatten()
    N = len(s)

    if N < 40:
        return np.nan

    # Remove linear trend
    s = s - np.mean(s)
    t = np.arange(N)
    p = np.polyfit(t, s, 1)
    s_detrended = s - np.polyval(p, t)

    if np.std(s_detrended) < 1e-10:
        return np.nan

    s_norm = s_detrended

    scale_min = max(int(cfg.RS_SCALE_MIN_SEC * fs), 32)
    scale_max = int(N * cfg.RS_SCALE_MAX_RATIO)
    n_scales = 20
    scales = np.unique(
        np.round(np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)).astype(int)
    )
    scales = scales[scales > 0]

    rs_values = []
    valid_scales = []

    for n in scales:
        if n < 4 or n >= N:
            continue
        n_blocks = N // n
        if n_blocks < cfg.RS_MIN_BLOCKS_PER_SCALE:
            continue

        rs_sum = 0.0
        valid_blocks = 0
        for j in range(n_blocks):
            block = s_norm[j * n : (j + 1) * n]
            std_block = np.std(block)
            if std_block < 1e-10:
                continue
            y = np.cumsum(block - np.mean(block))
            r = np.max(y) - np.min(y)
            rs_sum += r / std_block
            valid_blocks += 1

        if valid_blocks > 0:
            rs_values.append(rs_sum / valid_blocks)
            valid_scales.append(n)

    if len(valid_scales) < cfg.RS_MIN_VALID_SCALES:
        return np.nan

    log_scales = np.log10(valid_scales)
    log_rs = np.log10(rs_values)
    p = np.polyfit(log_scales, log_rs, 1)
    return float(p[0])  # H = slope


def compute_higuchi(signal: np.ndarray, fs: float = None) -> float:
    """Higuchi fractal dimension with adaptive k_max.

    k_max adapts to signal length:
    - k_max = min(N * 0.25, 30), floor = 5
    - Requires at least 5 valid k values for log-log fit

    Returns:
        Fractal dimension D (higher = more complex).
        To convert to Hurst: H = 2 - D
    """
    s = np.asarray(signal, dtype=float).flatten()
    N = len(s)

    k_max = min(int(N * cfg.HIGUCHI_K_MAX_RATIO), cfg.HIGUCHI_K_MAX_CEIL)
    k_max = max(k_max, cfg.HIGUCHI_K_MAX_FLOOR)

    L_k = np.zeros(k_max)

    for k in range(1, k_max + 1):
        L_sum = 0.0
        valid_m = 0

        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) <= 2:
                continue

            length = np.sum(np.abs(np.diff(s[idx])))
            norm = (N - 1) / (np.floor((N - m) / k) * k)
            L_sum += length * norm
            valid_m += 1

        if valid_m > 0:
            L_k[k - 1] = L_sum / (k * valid_m)

    valid = L_k > 1e-10
    if np.sum(valid) < cfg.HIGUCHI_MIN_VALID_POINTS:
        return np.nan

    k_range = np.arange(1, k_max + 1)
    p = np.polyfit(np.log10(k_range[valid]), np.log10(L_k[valid]), 1)
    D = -p[0]
    return float(D)


def compute_dfa(signal: np.ndarray, fs: float = None) -> float:
    """DFA long-range correlation exponent with adaptive scale range.

    Scale range adapts to signal length:
    - Minimum scale: max(16, fs * 0.064) samples
    - Maximum scale: N * 0.33 samples (wider than Niveles2's 0.25)
    - Requires at least 5 valid scales for log-log fit

    Returns:
        Exponent α: α ≈ 0.5 for white noise, α > 0.5 for persistent correlations,
        α < 0.5 for anti-persistent correlations.
    """
    fs = fs or cfg.FS
    s = np.asarray(signal, dtype=float).flatten()

    if np.std(s) < 1e-8:
        return np.nan

    # Integrated profile
    y = np.cumsum(s - np.mean(s))
    N = len(y)

    scale_min = max(int(cfg.DFA_SCALE_MIN_SEC * fs), 16)
    scale_max = int(N * cfg.DFA_SCALE_MAX_RATIO)
    n_scales = 15
    scales = np.unique(
        np.floor(np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)).astype(int)
    )
    scales = scales[scales > 0]

    F_vals = []
    valid_scales = []

    for n in scales:
        n_windows = N // n
        if n_windows < cfg.DFA_MIN_WINDOWS_PER_SCALE:
            continue

        error_total = 0.0
        for j in range(n_windows):
            idx = np.arange(j * n, (j + 1) * n)
            window = y[idx]
            t = np.arange(n)
            p = np.polyfit(t, window, 1)
            trend = np.polyval(p, t)
            error_total += np.sum((window - trend) ** 2)

        F_n = np.sqrt(error_total / (n_windows * n))
        if F_n > 1e-10:
            F_vals.append(F_n)
            valid_scales.append(n)

    if len(valid_scales) < cfg.DFA_MIN_VALID_SCALES:
        return np.nan

    p = np.polyfit(np.log10(valid_scales), np.log10(F_vals), 1)
    return float(p[0])  # α = slope


def compute_variogram(signal: np.ndarray, fs: float = None) -> float:
    """Variogram-based Hurst exponent with extended adaptive lags.

    Lag range adapts to signal length:
    - Maximum lag: N * 0.20 (not fixed at 15)
    - Requires at least 10 lags and 5 valid points for log-log fit

    Returns:
        Hurst exponent H = slope / 2.
    """
    s = np.asarray(signal, dtype=float).flatten()
    N = len(s)

    max_lag = max(int(N * cfg.VARIOGRAM_MAX_LAG_RATIO), cfg.VARIOGRAM_MIN_LAGS)
    max_lag = min(max_lag, N // 5)

    lags = np.arange(1, max_lag)
    gamma = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        diffs = s[:N - lag] - s[lag:]
        gamma[i] = 0.5 * np.mean(diffs ** 2)

    valid = gamma > 1e-10
    if np.sum(valid) < cfg.VARIOGRAM_MIN_VALID_POINTS:
        return np.nan

    p = np.polyfit(np.log10(lags[valid]), np.log10(gamma[valid]), 1)
    return float(p[0] / 2.0)  # H = slope / 2


def compute_ho(signal: np.ndarray, fs: float = None) -> float:
    """HO = Hurst Original — classic R/S method (Martinez-Peon 2024).

    Computes rescaled range on the full signal without partition correction.
    Unlike compute_rs(), this does not detrend — only mean subtraction.

    Args:
        signal: 1D float array.
        fs: Sampling rate for scale calibration.

    Returns:
        Hurst exponent H or NaN on failure.
    """
    fs = fs or cfg.FS
    s = np.asarray(signal, dtype=float).flatten()
    N = len(s)

    if N < 16:
        return np.nan

    s = s - np.mean(s)

    if np.std(s) < 1e-10:
        return np.nan

    scale_min = max(int(cfg.DFA_SCALE_MIN_SEC * fs), 16)
    scale_max = int(N * 0.4)
    if scale_max <= scale_min:
        return np.nan

    n_scales = 15
    scales = np.unique(
        np.round(np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)).astype(int)
    )
    scales = scales[scales > 0]

    rs_values = []
    valid_scales = []

    for n in scales:
        if n < 4 or n >= N:
            continue
        n_blocks = N // n
        if n_blocks < 3:
            continue

        rs_sum = 0.0
        valid_blocks = 0
        for j in range(n_blocks):
            block = s[j * n : (j + 1) * n]
            std_block = np.std(block)
            if std_block < 1e-10:
                continue
            y = np.cumsum(block - np.mean(block))
            r = np.max(y) - np.min(y)
            rs_sum += r / std_block
            valid_blocks += 1

        if valid_blocks > 0:
            rs_values.append(rs_sum / valid_blocks)
            valid_scales.append(n)

    if len(valid_scales) < 5:
        return np.nan

    log_scales = np.log10(valid_scales)
    log_rs = np.log10(rs_values)
    p = np.polyfit(log_scales, log_rs, 1)
    return float(p[0])


def compute_hrs(signal: np.ndarray, fs: float = None, p: int = 64) -> float:
    """HRS = Hurst with Rescaled Range using partitions (Martinez-Peon 2024).

    Divides the signal into p contiguous subseries of length N//p.
    For each subseries, computes R/S at internal scales. Pools R/S
    statistics across all p subseries, then fits a single Hurst value.

    Falls back to compute_ho() if any subseries has too few points (<10).

    Args:
        signal: 1D float array.
        fs: Sampling rate.
        p: Number of partitions.

    Returns:
        Hurst exponent H or NaN on failure.
    """
    fs = fs or cfg.FS
    s = np.asarray(signal, dtype=float).flatten()
    N = len(s)

    if N < 16:
        return np.nan

    s = s - np.mean(s)

    if np.std(s) < 1e-10:
        return np.nan

    subseries_len = N // p
    if subseries_len < 10:
        return compute_ho(signal, fs=fs)

    scale_min = max(int(cfg.DFA_SCALE_MIN_SEC * fs), 16)
    scale_max = min(int(subseries_len * 0.4), 64)
    if scale_max <= scale_min:
        return compute_ho(signal, fs=fs)

    n_scales = 15
    scales_raw = np.unique(
        np.round(np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)).astype(int)
    )
    scales_raw = scales_raw[scales_raw > 0]

    pooled_rs = {}
    pooled_counts = {}

    for n in scales_raw:
        if n < 4 or n >= subseries_len:
            continue
        if subseries_len // n < 3:
            continue
        pooled_rs[n] = 0.0
        pooled_counts[n] = 0

    if not pooled_rs:
        return compute_ho(signal, fs=fs)

    for i in range(p):
        sub = s[i * subseries_len : (i + 1) * subseries_len]
        sub_N = len(sub)

        if np.std(sub) < 1e-10:
            continue

        for n in list(pooled_rs.keys()):
            n_blocks = sub_N // n
            if n_blocks < 3:
                continue
            rs_sum = 0.0
            valid_blocks = 0
            for j in range(n_blocks):
                block = sub[j * n : (j + 1) * n]
                std_block = np.std(block)
                if std_block < 1e-10:
                    continue
                y = np.cumsum(block - np.mean(block))
                r = np.max(y) - np.min(y)
                rs_sum += r / std_block
                valid_blocks += 1

            if valid_blocks > 0:
                pooled_rs[n] += rs_sum / valid_blocks
                pooled_counts[n] += 1

    valid_scales = []
    rs_values = []
    for n in sorted(pooled_rs.keys()):
        if pooled_counts[n] > 0:
            valid_scales.append(n)
            rs_values.append(pooled_rs[n] / pooled_counts[n])

    if len(valid_scales) < 5:
        return np.nan

    log_scales = np.log10(valid_scales)
    log_rs = np.log10(rs_values)
    k = np.polyfit(log_scales, log_rs, 1)
    return float(k[0])


def compute_hrs_vector(signal: np.ndarray, fs: float = None,
                       p_values: list = None) -> dict:
    """Compute HRS across multiple partition values (Martinez-Peon 2024).

    Args:
        signal: 1D float array.
        fs: Sampling rate.
        p_values: List of partition counts. Defaults to cfg.HRS_PARTITION_VALUES.

    Returns:
        Dict like {"HRS_p2": 0.72, "HRS_p4": 0.68, ..., "HRS_p64": 0.65}.
        Skips any p > N//10.
    """
    fs = fs or cfg.FS
    if p_values is None:
        p_values = list(cfg.HRS_PARTITION_VALUES)

    s = np.asarray(signal, dtype=float).flatten()
    N = len(s)

    result = {}
    for p_val in p_values:
        if p_val > N // 10:
            continue
        key = f"HRS_p{p_val}"
        result[key] = compute_hrs(signal, fs=fs, p=p_val)

    return result


def compute_fractal_vector(signal: np.ndarray, fs: float = None) -> np.ndarray:
    """Compute all 6 fractal features + average + HV for a single 1D signal.

    Convenience wrapper for single-channel use.

    Returns:
        (8,) array: [RS, Higuchi, DFA, Variogram, Average, HO, HRS_p64, HV]
    """
    fs = fs or cfg.FS
    rs = compute_rs(signal, fs=fs)
    hig = compute_higuchi(signal, fs=fs)
    dfa = compute_dfa(signal, fs=fs)
    var = compute_variogram(signal, fs=fs)
    ho = compute_ho(signal, fs=fs)
    hrs = compute_hrs(signal, fs=fs, p=cfg.BEST_HRS_PARTITION)
    values = [rs, hig, dfa, var]
    avg = np.nanmean(values)
    return np.array([rs, hig, dfa, var, avg, ho, hrs, var])
