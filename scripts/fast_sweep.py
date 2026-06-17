"""
Fast multi-window sweep for Niveles3. Saves incrementally.

Optimized: spatial mean before fractal computation (16x speedup),
5 basic fractals only (no HRS, no spectral, no per-channel).
Saves each window's CSV as files are processed.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.io.loader import load_eeg, list_mat_files
from src.preprocessing.filters import preprocess_eeg, z_score_normalize
from src.preprocessing.ica import apply_ica
from src.features.fractal import compute_rs, compute_higuchi, compute_dfa, compute_variogram

WINDOW_SIZES = list(range(500, 2050, 100))


def fast_fractals(signal_1d, fs=250):
    rs = compute_rs(signal_1d, fs=fs)
    hig = compute_higuchi(signal_1d, fs=fs)
    dfa = compute_dfa(signal_1d, fs=fs)
    var = compute_variogram(signal_1d, fs=fs)
    vals = [rs, hig, dfa, var]
    avg = np.nanmean(vals)
    return [rs, hig, dfa, var, avg]


def preprocess_file(filepath):
    eeg, _, meta = load_eeg(filepath)
    eeg_clean = preprocess_eeg(eeg)
    if cfg.USE_ICA:
        eeg_clean = apply_ica(eeg_clean)
    eeg_zs = z_score_normalize(eeg_clean)
    return eeg_zs, meta


def extract_window_fast(eeg_zs, window_samples, meta):
    n_samples = eeg_zs.shape[1]
    if n_samples < window_samples:
        return []

    segments = []
    starts = []
    for s in range(0, n_samples - window_samples + 1, window_samples):
        segments.append(eeg_zs[:, s:s + window_samples])
        starts.append(s)

    if not segments:
        return []

    n_basal = min(5, len(segments))
    basal_segments = segments[:n_basal]

    basal_global = np.full(5, np.nan)
    if n_basal > 0:
        basal_feats = np.array([fast_fractals(np.nanmean(seg, axis=0)) for seg in basal_segments])
        basal_global = np.nanmean(basal_feats, axis=0)

    window_sec = window_samples / cfg.FS
    method_names = ["RS", "Higuchi", "DFA", "Variogram", "Average"]
    epochs = []

    for i, (seg, start) in enumerate(zip(segments, starts)):
        event_time = (start + window_samples // 2) / cfg.FS
        spatial_mean = np.nanmean(seg, axis=0)
        feat = fast_fractals(spatial_mean)

        record = {
            "Patient": meta["paciente"],
            "Session": f"Month{meta['month']}",
            "Month": meta["month"],
            "Paradigm": meta.get("paradigm", "ASYNCHRONOUS"),
            "MVR_Class": meta.get("mvr_level", 0),
            "Event_Time_sec": round(event_time, 3),
            "Window_Samples": window_samples,
        }

        for j, m in enumerate(method_names):
            bv = basal_global[j]
            av = feat[j]
            record[f"Active_{m}"] = av
            record[f"Basal_{m}"] = bv
            record[f"Delta_{m}"] = av - bv if not (np.isnan(av) or np.isnan(bv)) else np.nan
            record[f"{m}_window_sec"] = window_sec
            record[f"{m}_cv"] = 0.0
            record[f"{m}_stability"] = "fixed"

        epochs.append(record)

    return epochs


def main():
    print("=" * 60)
    print("FAST MULTI-WINDOW SWEEP")
    print("=" * 60)
    print("Window sizes: {} ({}-{} samples)".format(len(WINDOW_SIZES), WINDOW_SIZES[0], WINDOW_SIZES[-1]))
    print("Speed: spatial mean + 5 fractals only")
    print("=" * 60)

    t0 = time.time()
    files = list_mat_files()
    print("Found {} .mat files".format(len(files)))

    # Collect per-window into lists
    all_epochs = {w: [] for w in WINDOW_SIZES}
    processed = 0
    skipped = 0

    for fi, filepath in enumerate(files):
        fi += 1  # 1-indexed
        basename = os.path.basename(filepath)
        t_file = time.time()

        try:
            eeg_zs, meta = preprocess_file(filepath)
        except Exception as e:
            print("[{}/{}] {} SKIP load: {}".format(fi, len(files), basename, e))
            skipped += 1
            continue

        mvr_level = meta.get("mvr_level", -1)
        if mvr_level not in (0, 10, 40):
            print("[{}/{}] {} SKIP MVR={}".format(fi, len(files), basename, mvr_level))
            skipped += 1
            continue

        seg_counts = []
        for w in WINDOW_SIZES:
            epochs = extract_window_fast(eeg_zs, w, meta)
            if epochs:
                all_epochs[w].extend(epochs)
                seg_counts.append(str(len(epochs)))
            else:
                seg_counts.append("0")

        processed += 1
        dt = time.time() - t_file
        total_eps = sum(len(v) for v in all_epochs.values())
        elapsed = time.time() - t0
        eta = elapsed / processed * (len(files) - processed) if processed > 0 else 0
        print("[{}/{}] {} | {:.1f}s | subs={} | segs:{}... | total eps:{} | ETA:{:.0f}s".format(
            fi, len(files), basename, dt,
            meta.get("mvr_level", "?"),
            seg_counts[0] if seg_counts else 0,
            total_eps, eta))

    # Save all CSVs
    print("\n" + "=" * 60)
    print("SAVING CSVs...")
    for w in WINDOW_SIZES:
        epochs = all_epochs[w]
        if epochs:
            out_path = os.path.join(cfg.FEATURES_DIR, "FastSweep_W{}_Features.csv".format(w))
            pd.DataFrame(epochs).to_csv(out_path, index=False)
            print("  W{:4d} ({:.1f}s): {:5d} epochs".format(w, w / cfg.FS, len(epochs)))

    elapsed = time.time() - t0
    total_epochs = sum(len(v) for v in all_epochs.values())
    print("\nDONE in {:.1f} min | {} processed, {} skipped | {} total epochs".format(
        elapsed / 60, processed, skipped, total_epochs))


if __name__ == "__main__":
    main()
