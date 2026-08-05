"""
Fixed-window feature extraction pipeline for Niveles3.

Two strategies that avoid the adaptive-window stability validation:

Strategy A: Regular Segments
  For ALL files (async and sync):
  1. Load and preprocess EEG (same as extract_features.py)
  2. Divide the FULL signal into contiguous 2-second segments with 50% overlap
  3. Label each segment with the file's MVR level (10, 40, or 0 for rest)
  4. Compute fractal features (RS, Higuchi, DFA, Variogram, HO, HRS_p64, HV)
     and spectral features per segment
  5. Delta = Active features minus Basal global (average of first 5 segments)
  6. No ERD detection needed

Strategy B: Strict ERD
  For ASYNCHRONOUS files only:
  1. Same preprocessing
  2. Stricter ERD: threshold_factor=0.1 (was 0.2), min_event_distance=5.0s
     (was 3.0s), min_event_width=1.0s (was 0.5s)
  3. For each detected event, extract 3-second window centered on event
  4. Same feature computation as Strategy A (per-channel fractals, HRS, spectral)
  5. Basal computed from inter-event gaps (same as original pipeline)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
import numpy as np
from tqdm import tqdm

from src.io.loader import load_eeg, list_mat_files
from src.io.exporter import save_epoch_csv, build_feature_record
from src.preprocessing.filters import preprocess_eeg, z_score_normalize, mu_beta_filter
from src.preprocessing.ica import apply_ica
from src.events.detector import detect_events
from src.events.basal import extract_basal_epochs
from src.features.fractal import compute_all_fractals, compute_hrs_vector
from src.features.spectral import compute_spectral_features


def extract_regular_segments(eeg_zs, segment_duration_sec=2.0, overlap=0.5):
    """Divide signal into fixed segments with overlap.

    Args:
        eeg_zs: (n_channels, n_samples) z-scored EEG.
        segment_duration_sec: Duration of each segment in seconds.
        overlap: Fraction overlap between consecutive segments.

    Returns:
        segments: List of (n_channels, seg_samples) arrays.
        starts: List of sample indices for segment start.
    """
    seg_samples = int(segment_duration_sec * cfg.FS)
    step = int(seg_samples * (1 - overlap))
    n_samples = eeg_zs.shape[1]
    segments = []
    starts = []
    for s in range(0, n_samples - seg_samples + 1, step):
        segments.append(eeg_zs[:, s:s + seg_samples])
        starts.append(s)
    return segments, starts


def process_file_strategy_a(filepath, segment_duration_sec=2.0, overlap=0.5):
    """Process one file with the regular-segment strategy.

    Divides the full recording into fixed 2 s windows (50 % overlap),
    labels each with the file MVR level, and computes all features
    using a pseudo-basal from the first five segments.
    """
    print(f"  [FIXED-A] Processing: {os.path.basename(filepath)}")
    try:
        eeg, _, meta = load_eeg(filepath)
    except Exception as e:
        print(f"  [SKIP] Load error: {e}")
        return []

    mvr_level = meta["mvr_level"]
    if mvr_level not in (0, 10, 40):
        print(f"  [SKIP] MVR level {mvr_level}")
        return []

    n_channels, n_samples = eeg.shape

    eeg_clean = preprocess_eeg(eeg)
    if cfg.USE_ICA:
        eeg_clean = apply_ica(eeg_clean)
    eeg_zs = z_score_normalize(eeg_clean)

    segments, starts = extract_regular_segments(eeg_zs, segment_duration_sec, overlap)
    print(f"  Extracted {len(segments)} fixed segments")

    basal_segments = segments[:min(5, len(segments))]

    basal_global = np.full(8, np.nan)
    basal_per_ch = None
    basal_ref_segment = None

    if len(basal_segments) > 0:
        basal_features_all = np.zeros((len(basal_segments), n_channels, 8))
        for b, seg in enumerate(basal_segments):
            basal_features_all[b] = compute_all_fractals(seg)
        basal_global = np.nanmean(basal_features_all, axis=(0, 1))
        basal_per_ch = np.nanmean(basal_features_all[..., :8], axis=0)
        basal_ref_segment = np.nanmean(np.stack(basal_segments), axis=0)

    epochs = []
    for i, (seg, start) in enumerate(zip(segments, starts)):
        event_time = start / cfg.FS

        pc_features = compute_all_fractals(seg)

        spatial_mean = np.nanmean(seg, axis=0)
        hrs_vector = compute_hrs_vector(spatial_mean, fs=cfg.FS)

        stable = {}
        method_names = ["RS", "Higuchi", "DFA", "Variogram", "Average"]
        feat_mean = np.nanmean(pc_features, axis=0)
        for j, m in enumerate(method_names):
            stable[f"{m}_value"] = float(feat_mean[j]) if j < 5 else 0.0
            stable[f"{m}_window_sec"] = segment_duration_sec
            stable[f"{m}_cv"] = 0.0
            stable[f"{m}_stability"] = "fixed"

        active_martinez = {
            "HO": float(np.nanmean(pc_features[:, 5])),
            "HV": float(np.nanmean(pc_features[:, 7])),
        }
        active_martinez.update(hrs_vector)

        if not np.all(np.isnan(basal_global)):
            basal_martinez = {
                "HO": float(basal_global[5]),
                "HV": float(basal_global[7]),
            }
            if basal_ref_segment is not None:
                hrs_basal = compute_hrs_vector(
                    np.nanmean(basal_ref_segment, axis=0), fs=cfg.FS)
                basal_martinez.update(hrs_basal)
            else:
                for p_val in cfg.HRS_PARTITION_VALUES:
                    basal_martinez[f"HRS_p{p_val}"] = np.nan
        else:
            basal_martinez = {"HO": np.nan, "HV": np.nan}
            for p_val in cfg.HRS_PARTITION_VALUES:
                basal_martinez[f"HRS_p{p_val}"] = np.nan

        martinez_features = {"basal": basal_martinez, "active": active_martinez}

        spectral_features = compute_spectral_features(
            seg, fs=cfg.FS, basal_segment=basal_ref_segment)

        record = build_feature_record(
            patient=meta["paciente"],
            month=meta["month"],
            paradigm=meta["paradigm"],
            mvr_class=mvr_level,
            event_time_sec=event_time,
            basal_global=basal_global,
            stable_features=stable,
            original_filename=meta["filename"],
            per_channel_features={
                "basal_per_ch": basal_per_ch,
                "active_per_ch": pc_features[:, :8],
            },
            martinez_features=martinez_features,
            spectral_features=spectral_features,
        )
        epochs.append(record)

    print(f"  Generated {len(epochs)} fixed-window epochs")
    return epochs


def process_file_strategy_b(filepath, event_window_sec=3.0):
    """Process one ASYNCHRONOUS file with strict ERD detection.

    Uses stricter ERD parameters:
      - threshold_factor = 0.1  (default 0.2)
      - min_event_distance  = 5.0 s (default 3.0 s)
      - min_event_width     = 1.0 s (default 0.5 s)

    For each detected event a 3-second window centred on the event is
    extracted and the same feature set as Strategy A is computed.
    """
    print(f"  [FIXED-B] Processing: {os.path.basename(filepath)}")
    try:
        eeg, _, meta = load_eeg(filepath)
    except Exception as e:
        print(f"  [SKIP] Load error: {e}")
        return []

    if meta["paradigm"] != "ASYNCHRONOUS":
        print(f"  [SKIP] Not ASYNCHRONOUS (paradigm={meta['paradigm']})")
        return []

    mvr_level = meta["mvr_level"]
    if mvr_level not in (10, 40):
        print(f"  [SKIP] MVR level {mvr_level}")
        return []

    n_channels, n_samples = eeg.shape

    eeg_clean = preprocess_eeg(eeg)
    if cfg.USE_ICA:
        eeg_clean = apply_ica(eeg_clean)
    eeg_zs = z_score_normalize(eeg_clean)

    data_raw = {}
    try:
        from scipy.io import loadmat
        raw = loadmat(filepath, squeeze_me=False)
        for key in raw:
            if not key.startswith("__"):
                data_raw[key] = raw[key]
    except Exception:
        pass

    saved_threshold = cfg.ERD_THRESHOLD_FACTOR
    saved_distance = cfg.MIN_EVENT_DISTANCE_SEC
    saved_width = cfg.MIN_EVENT_WIDTH_SEC

    cfg.ERD_THRESHOLD_FACTOR = 0.1
    cfg.MIN_EVENT_DISTANCE_SEC = 5.0
    cfg.MIN_EVENT_WIDTH_SEC = 1.0

    try:
        events = detect_events(eeg_clean, data_raw, is_imagery=True)
    finally:
        cfg.ERD_THRESHOLD_FACTOR = saved_threshold
        cfg.MIN_EVENT_DISTANCE_SEC = saved_distance
        cfg.MIN_EVENT_WIDTH_SEC = saved_width

    if len(events) == 0:
        print(f"  [SKIP] No events detected with strict ERD")
        return []

    print(f"  Strict ERD events: {len(events)}")

    eeg_mu = mu_beta_filter(eeg_clean)
    basal_epochs = extract_basal_epochs(eeg_mu, events)
    basal_epochs_zs = extract_basal_epochs(eeg_zs, events)
    print(f"  Basal epochs: {len(basal_epochs)}")

    basal_global = np.full(8, np.nan)
    basal_features_all = None

    if basal_epochs.size > 0 and not np.isnan(basal_epochs).all():
        n_b, n_ch, n_s = basal_epochs.shape
        basal_features_all = np.zeros((n_b, n_ch, 8))
        for b in range(n_b):
            basal_features_all[b] = compute_all_fractals(basal_epochs[b])

        if n_b > 0 and not np.all(np.isnan(basal_features_all)):
            basal_global = np.nanmean(basal_features_all, axis=(0, 1))

    basal_ref_segment = None
    if basal_epochs_zs.size > 0 and not np.isnan(basal_epochs_zs).all():
        basal_ref_segment = np.nanmean(basal_epochs_zs, axis=0)

    basal_per_ch = None
    if basal_features_all is not None and basal_features_all.size > 0:
        basal_per_ch = np.nanmean(basal_features_all[:, :, :8], axis=0)

    active_samples = int(event_window_sec * cfg.FS)
    epochs = []

    for ev_center in events:
        start = ev_center - active_samples // 2
        end = start + active_samples
        if start < 0:
            start, end = 0, min(active_samples, n_samples)
        if end > n_samples:
            start, end = max(0, n_samples - active_samples), n_samples

        event_time = (start + end) // 2 / cfg.FS
        segment = eeg_zs[:, start:end]

        pc_features = compute_all_fractals(segment)

        spatial_mean = np.nanmean(segment, axis=0)
        hrs_vector = compute_hrs_vector(spatial_mean, fs=cfg.FS)

        stable = {}
        method_names = ["RS", "Higuchi", "DFA", "Variogram", "Average"]
        feat_mean = np.nanmean(pc_features, axis=0)
        for j, m in enumerate(method_names):
            stable[f"{m}_value"] = float(feat_mean[j]) if j < 5 else 0.0
            stable[f"{m}_window_sec"] = event_window_sec
            stable[f"{m}_cv"] = 0.0
            stable[f"{m}_stability"] = "fixed"

        active_martinez = {
            "HO": float(np.nanmean(pc_features[:, 5])),
            "HV": float(np.nanmean(pc_features[:, 7])),
        }
        active_martinez.update(hrs_vector)

        if not np.all(np.isnan(basal_global)):
            basal_martinez = {
                "HO": float(basal_global[5]),
                "HV": float(basal_global[7]),
            }
            if basal_ref_segment is not None:
                hrs_basal = compute_hrs_vector(
                    np.nanmean(basal_ref_segment, axis=0), fs=cfg.FS)
                basal_martinez.update(hrs_basal)
            else:
                for p_val in cfg.HRS_PARTITION_VALUES:
                    basal_martinez[f"HRS_p{p_val}"] = np.nan
        else:
            basal_martinez = {"HO": np.nan, "HV": np.nan}
            for p_val in cfg.HRS_PARTITION_VALUES:
                basal_martinez[f"HRS_p{p_val}"] = np.nan

        martinez_features = {"basal": basal_martinez, "active": active_martinez}

        spectral_features = compute_spectral_features(
            segment, fs=cfg.FS, basal_segment=basal_ref_segment)

        record = build_feature_record(
            patient=meta["paciente"],
            month=meta["month"],
            paradigm=meta["paradigm"],
            mvr_class=mvr_level,
            event_time_sec=event_time,
            basal_global=basal_global,
            stable_features=stable,
            original_filename=meta["filename"],
            per_channel_features={
                "basal_per_ch": basal_per_ch,
                "active_per_ch": pc_features[:, :8],
            },
            martinez_features=martinez_features,
            spectral_features=spectral_features,
        )
        epochs.append(record)

    print(f"  Generated {len(epochs)} strict-ERD epochs")
    return epochs


def main():
    print("=" * 55)
    print("NIVELES3 — FIXED WINDOW FEATURE EXTRACTION")
    print("=" * 55)
    print("Strategy A: Regular segments (2s, 50% overlap)")
    print("Strategy B: Strict ERD (threshold=0.1, dist=5s, width=1s, 3s window)")
    print(f"Data directory: {cfg.DATA_DIR}")
    print("=" * 55)

    t0 = time.time()

    files = list_mat_files()
    print(f"Found {len(files)} .mat files\n")

    print("=== STRATEGY A: Regular Segments ===")
    all_epochs_a = []
    for filepath in tqdm(files, desc="Fixed-window extraction A"):
        epochs = process_file_strategy_a(filepath, segment_duration_sec=2.0, overlap=0.0)
        all_epochs_a.extend(epochs)

    if all_epochs_a:
        out_a = os.path.join(cfg.FEATURES_DIR, "FixedA_Features.csv")
        save_epoch_csv(all_epochs_a, out_a)
        print(f"\nStrategy A CSV: {out_a} ({len(all_epochs_a)} epochs)")

    print("\n=== STRATEGY B: Strict ERD ===")
    all_epochs_b = []
    for filepath in tqdm(files, desc="Strict ERD extraction B"):
        epochs = process_file_strategy_b(filepath, event_window_sec=3.0)
        all_epochs_b.extend(epochs)

    if all_epochs_b:
        out_b = os.path.join(cfg.FEATURES_DIR, "FixedB_Features.csv")
        save_epoch_csv(all_epochs_b, out_b)
        print(f"\nStrategy B CSV: {out_b} ({len(all_epochs_b)} epochs)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Strategy A: {len(all_epochs_a)} epochs")
    print(f"Strategy B: {len(all_epochs_b)} epochs")


if __name__ == "__main__":
    main()
