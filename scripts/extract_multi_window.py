"""
Multi-window fixed-segment feature extraction for Niveles3.

Extracts features using 16 window sizes (500-2000 samples, step 100)
with 0% overlap. Processes each .mat file once, caching preprocessed
signals, then segments and computes features for all window sizes.

Output: output/features/Fixed_W{N}_Features.csv (one CSV per window size)
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from tqdm import tqdm
from src.io.loader import load_eeg, list_mat_files
from src.io.exporter import save_epoch_csv, build_feature_record
from src.preprocessing.filters import preprocess_eeg, z_score_normalize
from src.preprocessing.ica import apply_ica
from src.features.fractal import compute_all_fractals, compute_hrs_vector
from src.features.spectral import compute_spectral_features

WINDOW_SIZES = list(range(500, 2050, 100))  # 500, 600, ..., 2000 samples


def preprocess_file(filepath):
    """Load and preprocess one file. Returns (eeg_zs, meta) or raises on error."""
    eeg, _, meta = load_eeg(filepath)
    eeg_clean = preprocess_eeg(eeg)
    if cfg.USE_ICA:
        eeg_clean = apply_ica(eeg_clean)
    eeg_zs = z_score_normalize(eeg_clean)
    return eeg_zs, meta


def extract_features_for_window(eeg_zs, window_samples, meta):
    """Segment signal into contiguous windows and extract features.

    Args:
        eeg_zs: (n_channels, n_samples) z-scored EEG.
        window_samples: Number of samples per window.
        meta: Metadata dict from load_eeg.

    Returns:
        List of feature record dicts (one per segment).
    """
    from src.io.exporter import build_feature_record

    n_samples = eeg_zs.shape[1]
    if n_samples < window_samples:
        return []

    segments = []
    starts = []
    for s in range(0, n_samples - window_samples + 1, window_samples):
        segments.append(eeg_zs[:, s:s + window_samples])
        starts.append(s)

    if len(segments) == 0:
        return []

    n_basal = min(5, len(segments))
    basal_segments = segments[:n_basal]

    basal_global = np.full(8, np.nan)
    basal_per_ch = None
    basal_ref_segment = None

    if n_basal > 0:
        basal_features_all = np.zeros((n_basal, eeg_zs.shape[0], 8))
        for b in range(n_basal):
            basal_features_all[b] = compute_all_fractals(basal_segments[b])
        basal_global = np.nanmean(basal_features_all, axis=(0, 1))
        basal_per_ch = np.nanmean(basal_features_all[..., :8], axis=0)
        basal_ref_segment = np.nanmean(np.stack(basal_segments), axis=0)

    basal_martinez = _build_basal_martinez(basal_global, basal_ref_segment)
    mat_paradigm = meta.get("paradigm", "ASYNCHRONOUS")
    mvr_level = meta.get("mvr_level", 0)
    window_sec = window_samples / cfg.FS

    epochs = []
    for i, (seg, start) in enumerate(zip(segments, starts)):
        event_time = (start + window_samples // 2) / cfg.FS

        pc_features = compute_all_fractals(seg)

        spatial_mean = np.nanmean(seg, axis=0)
        hrs_vector = compute_hrs_vector(spatial_mean, fs=cfg.FS)

        stable = {}
        feat_mean = np.nanmean(pc_features, axis=0)
        for j, m in enumerate(["RS", "Higuchi", "DFA", "Variogram", "Average"]):
            stable[f"{m}_value"] = float(feat_mean[j])
            stable[f"{m}_window_sec"] = window_sec
            stable[f"{m}_cv"] = 0.0
            stable[f"{m}_stability"] = "fixed"

        active_martinez = {
            "HO": float(np.nanmean(pc_features[:, 5])),
            "HV": float(np.nanmean(pc_features[:, 7])),
        }
        active_martinez.update(hrs_vector)
        martinez_features = {"basal": basal_martinez, "active": active_martinez}

        spectral = compute_spectral_features(seg, fs=cfg.FS, basal_segment=basal_ref_segment)

        record = build_feature_record(
            patient=meta["paciente"],
            month=meta["month"],
            paradigm=mat_paradigm,
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
            spectral_features=spectral,
        )
        epochs.append(record)

    return epochs


def _build_basal_martinez(basal_global, basal_ref_segment):
    """Build basal Martinez features dict from global basal array."""
    if np.all(np.isnan(basal_global[:5])):
        result = {"HO": np.nan, "HV": np.nan}
        for p in cfg.HRS_PARTITION_VALUES:
            result[f"HRS_p{p}"] = np.nan
        return result

    result = {"HO": float(basal_global[5]), "HV": float(basal_global[7])}
    if basal_ref_segment is not None and not np.all(np.isnan(basal_ref_segment)):
        hrs_basal = compute_hrs_vector(np.nanmean(basal_ref_segment, axis=0), fs=cfg.FS)
        result.update(hrs_basal)
    else:
        for p in cfg.HRS_PARTITION_VALUES:
            result[f"HRS_p{p}"] = np.nan
    return result


def main():
    print("=" * 60)
    print("MULTI-WINDOW FIXED SEGMENT EXTRACTION")
    print("=" * 60)
    print(f"Window sizes: {len(WINDOW_SIZES)} ({WINDOW_SIZES[0]}-{WINDOW_SIZES[-1]} samples)")
    print(f"Samples at {cfg.FS} Hz: {WINDOW_SIZES[0]/cfg.FS:.1f}s - {WINDOW_SIZES[-1]/cfg.FS:.1f}s")
    print(f"Overlap: 0% (contiguous segments)")
    print(f"ICA: {'On' if cfg.USE_ICA else 'Off'}")
    print("=" * 60)

    t0 = time.time()
    files = list_mat_files()
    print(f"Found {len(files)} .mat files\n")

    all_epochs = {w: [] for w in WINDOW_SIZES}
    preprocess_time = 0.0
    feature_time = 0.0

    for fi, filepath in enumerate(tqdm(files, desc="Processing files"), 1):
        basename = os.path.basename(filepath)
        print(f"\n[{fi}/{len(files)}] {basename}")

        t_pre = time.time()
        try:
            eeg_zs, meta = preprocess_file(filepath)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        preprocess_time += time.time() - t_pre

        mvr_level = meta.get("mvr_level", -1)
        if mvr_level not in (0, 10, 40):
            print(f"  SKIP: MVR={mvr_level}")
            continue

        print(f"  {meta['paciente']} M{meta['month']} {meta['paradigm']} MVR={mvr_level}% "
              f"| signal: {eeg_zs.shape[1]} samples ({eeg_zs.shape[1]/cfg.FS:.1f}s)")

        for w in tqdm(WINDOW_SIZES, desc="  Window sizes", leave=False):
            t_feat = time.time()
            epochs = extract_features_for_window(eeg_zs, w, meta)
            feature_time += time.time() - t_feat
            if epochs:
                all_epochs[w].extend(epochs)

        print(f"  [{time.time()-t0:.0f}s elapsed] pre={preprocess_time:.0f}s feat={feature_time:.0f}s")

    print("\n" + "=" * 60)
    print("SAVING CSVs")

    for w in WINDOW_SIZES:
        epochs = all_epochs[w]
        if epochs:
            out_path = os.path.join(cfg.FEATURES_DIR, f"Fixed_W{w}_Features.csv")
            save_epoch_csv(epochs, out_path)
            print(f"  W{w:4d} ({w/cfg.FS:.1f}s): {len(epochs):5d} epochs → Fixed_W{w}_Features.csv")
        else:
            print(f"  W{w:4d}: 0 epochs (SKIPPED)")

    elapsed = time.time() - t0
    total_epochs = sum(len(v) for v in all_epochs.values())
    print(f"\nDONE in {elapsed/60:.1f} min | {total_epochs} total epochs | "
          f"pre={preprocess_time/60:.1f}m feat={feature_time/60:.1f}m")


if __name__ == "__main__":
    main()
