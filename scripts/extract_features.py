"""
Main feature extraction pipeline.

Processes all .mat EEG files:
1. Load EEG + metadata
2. Preprocess (DC → Notch → Bandpass → ICA → Z-score)
3. Detect events (ERD for imagery, markers for stimulus-driven)
4. Extract basal epochs
5. Compute basal global features
6. For each event: extract adaptive-window features with stability validation
7. Generate basal-class epochs (class 0%)
8. Export per-file CSV + global consolidated CSV
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
from src.features.stability import compute_stable_features, get_stability_summary
from src.features.spectral import compute_spectral_features


def process_file(filepath: str) -> list:
    """Process a single .mat file and return epoch feature dicts.

    Returns:
        List of dicts, one per epoch (including basal epochs).
        Returns empty list on failure.
    """
    print(f"Processing: {os.path.basename(filepath)}")

    # 1. Load
    try:
        eeg, _, meta = load_eeg(filepath)
    except Exception as e:
        print(f"  [SKIP] Load error: {e}")
        return []

    is_imagery = meta["paradigm"] == "ASYNCHRONOUS"
    mvr_level = meta["mvr_level"]
    if mvr_level not in (10, 40):
        print(f"  [SKIP] MVR level not 10 or 40: {mvr_level}")
        return []

    n_channels, n_samples = eeg.shape
    print(f"  Loaded: {n_channels}ch × {n_samples}samples ({n_samples/cfg.FS:.1f}s) "
          f"| Patient={meta['paciente']} Month={meta['month']} "
          f"Paradigm={meta['paradigm']} MVR={mvr_level}%")

    # 2. Preprocess
    eeg_clean = preprocess_eeg(eeg)
    if cfg.USE_ICA:
        eeg_clean = apply_ica(eeg_clean)

    # 3. Detect events
    data_raw = {}
    try:
        from scipy.io import loadmat
        raw = loadmat(filepath, squeeze_me=False)
        for key in raw:
            if not key.startswith("__"):
                data_raw[key] = raw[key]
    except Exception:
        pass

    events = detect_events(eeg_clean, data_raw, is_imagery)
    if len(events) == 0:
        print(f"  [SKIP] No events detected")
        return []

    print(f"  Events detected: {len(events)}")

    # 4. Z-score normalize for fractal feature extraction (preserves relative structure)
    eeg_zs = z_score_normalize(eeg_clean)

    # 5. Extract basal epochs and compute basal global
    eeg_mu = mu_beta_filter(eeg_clean)  # mu/beta band for basal analysis
    basal_epochs = extract_basal_epochs(eeg_mu, events)
    print(f"  Basal epochs: {len(basal_epochs)}")

    # Basal epochs from z-scored EEG for spectral reference
    basal_epochs_zs = extract_basal_epochs(eeg_zs, events)

    # Compute basal global on filtered EEG
    if basal_epochs.size > 0 and not np.isnan(basal_epochs).all():
        n_b, n_ch, n_s = basal_epochs.shape
        basal_features_all = np.zeros((n_b, n_ch, 8))
        for b in range(n_b):
            basal_features_all[b] = compute_all_fractals(basal_epochs[b])

        if n_b > 0 and not np.all(np.isnan(basal_features_all)):
            basal_global_per_channel = np.nanmean(basal_features_all, axis=0)
            basal_global = np.nanmean(basal_global_per_channel, axis=0)
            print(f"  Basal global: RS={basal_global[0]:.3f} Hig={basal_global[1]:.3f} "
                  f"DFA={basal_global[2]:.3f} Var={basal_global[3]:.3f} "
                  f"HO={basal_global[5]:.3f} HRS_p64={basal_global[6]:.3f} HV={basal_global[7]:.3f}")
        else:
            basal_global = np.full(8, np.nan)
    else:
        basal_global = np.full(8, np.nan)

    # Basal reference segment for spectral features (z-scored, spatial mean across epochs)
    if basal_epochs_zs.size > 0 and not np.isnan(basal_epochs_zs).all():
        basal_ref_segment = np.nanmean(basal_epochs_zs, axis=0)  # (n_ch, n_s)
    else:
        basal_ref_segment = None

    # 6. Process each event with stability validation
    epochs = []
    active_window_sec = cfg.STABILITY_WINDOWS_SEC[-1]  # 5s default
    active_samples = int(active_window_sec * cfg.FS)

    # Basal per-channel features (for the first approach: average across basal epochs)
    basal_per_ch = None
    if basal_epochs.size > 0 and not np.isnan(basal_epochs).all():
        # Average basal features across all basal epochs, per channel
        # basal_features_all shape: (n_b, n_ch, 8)
        basal_per_ch = np.nanmean(basal_features_all[:, :, :8], axis=0)  # (n_ch, 8)

    for i, ev_center in enumerate(events):
        start = ev_center - active_samples // 2
        end = start + active_samples

        if start < 0:
            start = 0
        if end > n_samples:
            end = n_samples

        actual_center = (start + end) // 2
        event_time = actual_center / cfg.FS

        # Stability validation (uses spatial average for speed)
        stable = compute_stable_features(eeg_zs, actual_center)

        # Per-channel features for the largest window
        win_samples = active_samples
        cs = actual_center - win_samples // 2
        ce = cs + win_samples
        if cs < 0:
            cs, ce = 0, min(win_samples, n_samples)
        if ce > n_samples:
            cs, ce = max(0, n_samples - win_samples), n_samples
        segment_pc = eeg_zs[:, cs:ce]
        pc_features = compute_all_fractals(segment_pc)  # (n_channels, 8)
        active_per_ch = pc_features[:, :8]  # (n_channels, 8)

        # Spectral features for active segment
        spectral_features = compute_spectral_features(segment_pc, fs=cfg.FS,
                                                      basal_segment=basal_ref_segment)

        # Martinez features: basal and active spatial means for HO, HV, HRS partitions
        spatial_mean_active = np.nanmean(segment_pc, axis=0)  # (n_samples,)
        hrs_active = compute_hrs_vector(spatial_mean_active, fs=cfg.FS)
        active_martinez = {
            "HO": float(np.nanmean(pc_features[:, 5])),
            "HV": float(np.nanmean(pc_features[:, 7])),
        }
        active_martinez.update(hrs_active)  # adds HRS_p2 ... HRS_p128

        if not np.all(np.isnan(basal_global)):
            basal_martinez = {
                "HO": float(basal_global[5]),
                "HV": float(basal_global[7]),
            }
            # HRS vector for basal (spatial mean of basal epochs)
            if basal_epochs.size > 0 and not np.isnan(basal_epochs).all():
                mean_basal_signal = np.nanmean(basal_epochs, axis=(0, 1))
                hrs_basal = compute_hrs_vector(mean_basal_signal, fs=cfg.FS)
                basal_martinez.update(hrs_basal)
            else:
                for p in cfg.HRS_PARTITION_VALUES:
                    basal_martinez[f"HRS_p{p}"] = np.nan
        else:
            basal_martinez = {"HO": np.nan, "HV": np.nan}
            for p in cfg.HRS_PARTITION_VALUES:
                basal_martinez[f"HRS_p{p}"] = np.nan
        martinez_features = {"basal": basal_martinez, "active": active_martinez}

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
                "active_per_ch": active_per_ch,
            },
            martinez_features=martinez_features,
            spectral_features=spectral_features,
        )
        epochs.append(record)

    # 7. Generate basal-class epochs
    if basal_epochs.size > 0 and cfg.WEKA_DEFAULT_SUBSET:
        for b in range(min(len(basal_epochs), 3)):
            basal_feat = np.nanmean(basal_features_all[b], axis=0)  # (8,) spatial avg

            basal_stable = {}
            for j, method in enumerate(["RS", "Higuchi", "DFA", "Variogram",
                                        "Average", "HO", "HRS_p64", "HV"]):
                basal_stable[f"{method}_value"] = basal_feat[j]
                basal_stable[f"{method}_window_sec"] = cfg.BASAL_SEGMENT_DURATION_SEC
                basal_stable[f"{method}_cv"] = 0.0
                basal_stable[f"{method}_stability"] = "basal"

            # Per-channel features for this basal segment
            bpc = basal_features_all[b, :, :8]  # (n_ch, 8)

            # Spectral features for this basal epoch
            if basal_epochs_zs.size > 0:
                basal_spectral_b = compute_spectral_features(
                    basal_epochs_zs[b], fs=cfg.FS, basal_segment=None)
            else:
                basal_spectral_b = {}

            # Martinez features for this basal epoch
            basal_mean_signal_b = np.nanmean(basal_epochs[b], axis=0)  # spatial mean of this basal epoch
            hrs_b = compute_hrs_vector(basal_mean_signal_b, fs=cfg.FS)
            basal_m = {
                "HO": float(basal_feat[5]),
                "HV": float(basal_feat[7]),
            }
            basal_m.update(hrs_b)  # adds HRS_p2 ... HRS_p128
            martinez_b = {"basal": basal_m, "active": basal_m}

            basal_record = build_feature_record(
                patient=meta["paciente"],
                month=meta["month"],
                paradigm=meta["paradigm"],
                mvr_class=0,
                event_time_sec=b * cfg.BASAL_SEGMENT_DURATION_SEC,
                basal_global=basal_global,
                stable_features=basal_stable,
                original_filename=meta["filename"],
                per_channel_features={
                    "basal_per_ch": bpc,
                    "active_per_ch": bpc,  # active = basal for class 0
                },
                martinez_features=martinez_b,
                spectral_features=basal_spectral_b,
            )
            epochs.append(basal_record)

    print(f"  Epochs generated: {len(epochs)} (active + basal)")
    return epochs


def main():
    print("=" * 55)
    print("NIVELES3 — FRACTAL FEATURE EXTRACTION")
    print("=" * 55)
    print(f"Data directory: {cfg.DATA_DIR}")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    print(f"Sampling rate: {cfg.FS} Hz")
    print(f"Montage: {len(cfg.CHANNEL_NAMES)} channels (Cz at index {cfg.IDX_CZ})")
    print(f"Epoch windows for stability: {cfg.STABILITY_WINDOWS_SEC}s")
    print("=" * 55)
    print()

    t0 = time.time()

    files = list_mat_files()
    print(f"Found {len(files)} .mat files\n")

    all_epochs = []
    processed = 0
    skipped = 0
    total_epochs = 0

    for i, filepath in enumerate(tqdm(files, desc="Processing files"), 1):
        print(f"  [{i}/{len(files)}] {os.path.basename(filepath)}")
        epochs = process_file(filepath)
        if epochs:
            all_epochs.extend(epochs)
            total_epochs += len(epochs)
            processed += 1
        else:
            skipped += 1

        # Save per-file CSV
        basename = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(cfg.FEATURES_DIR, f"{basename}_Features.csv")
        save_epoch_csv(epochs, out_path)

    # Save global consolidated CSV
    if all_epochs:
        global_path = os.path.join(cfg.FEATURES_DIR, "Global_All_Features.csv")
        save_epoch_csv(all_epochs, global_path)
        print(f"\nGlobal CSV saved: {global_path} ({len(all_epochs)} epochs)")

        import pandas as pd
        df = pd.DataFrame(all_epochs)
        print(f"\nClass distribution:")
        for cls in sorted(df["MVR_Class"].unique()):
            count = len(df[df["MVR_Class"] == cls])
            print(f"  MVR {cls}%: {count} epochs ({100*count/len(df):.1f}%)")

        # Stability summary
        print(f"\nStability summary:")
        for method in ["RS", "Higuchi", "DFA", "Variogram", "Average",
                       "HO", "HRS_p64", "HV"]:
            col = f"{method}_stability"
            if col in df.columns:
                counts = df[col].value_counts()
                stats = ", ".join(f"{k}={v}" for k, v in counts.items())
                print(f"  {method}: {stats}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s | {processed} files processed, {skipped} skipped | {total_epochs} epochs total")


if __name__ == "__main__":
    main()
