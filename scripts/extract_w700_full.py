"""Full extraction for best window W700 (2.8s) with per-channel + HRS + spectral."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from joblib import Parallel, delayed
from src.io.loader import load_eeg, list_mat_files
from src.io.exporter import save_epoch_csv, build_feature_record
from src.preprocessing.filters import preprocess_eeg, z_score_normalize
from src.preprocessing.ica import apply_ica
from src.preprocessing.segmentation import segment_fixed_window
from src.events.detector import detect_erd_events
from src.events.basal import extract_basal_epochs
from src.features.fractal import compute_all_fractals, compute_hrs_vector
from src.features.spectral import compute_spectral_features
from src.features.basal_reference import build_basal_reference

W = 700  # samples = 2.8s at 250 Hz
OVERLAP = 0.0


def _build_record(seg, seg_class, event_time, meta, basal, window_sec):
    """Compute all fractal/spectral/Martinez features for one W700 segment
    and assemble a CSV row. Shared by both the "active" segments (contiguous,
    labeled with the file's MVR class) and the genuine "rest" segments
    (gap-detected, labeled class 0) below — identical feature computation,
    only the segment source and class label differ.
    """
    pc_features = compute_all_fractals(seg)
    spatial_mean = np.nanmean(seg, axis=0)
    hrs_vector = compute_hrs_vector(spatial_mean, fs=cfg.FS)

    stable = {}
    feat_mean = np.nanmean(pc_features, axis=0)
    for j, m in enumerate(["RS", "Higuchi", "DFA", "Variogram", "Average"]):
        stable["{}_value".format(m)] = float(feat_mean[j])
        stable["{}_window_sec".format(m)] = window_sec
        stable["{}_cv".format(m)] = 0.0
        stable["{}_stability".format(m)] = "fixed"

    active_martinez = {
        "HO": float(np.nanmean(pc_features[:, 5])),
        "HV": float(np.nanmean(pc_features[:, 7])),
    }
    active_martinez.update(hrs_vector)
    martinez_features = {"basal": basal.basal_martinez, "active": active_martinez}

    spectral = compute_spectral_features(
        seg, fs=cfg.FS, basal_band_powers=basal.basal_band_powers
    )

    return build_feature_record(
        patient=meta["paciente"], month=meta["month"],
        paradigm=meta.get("paradigm", "ASYNCHRONOUS"),
        mvr_class=seg_class, event_time_sec=event_time,
        basal_global=basal.basal_global, stable_features=stable,
        original_filename=meta["filename"],
        per_channel_features={"basal_per_ch": basal.basal_per_ch,
                               "active_per_ch": pc_features[:, :8]},
        martinez_features=martinez_features, spectral_features=spectral,
    )


def process_file(filepath):
    """Load, preprocess and extract all W700 epoch records for one .mat file.

    Returns a list of record dicts (possibly empty if the file is skipped).
    Factored out of the old flat per-file loop so it can run in a worker
    process via joblib.Parallel — each file's pipeline (load -> filter ->
    ICA -> segment -> fractal/spectral features) is independent of every
    other file, so this is embarrassingly parallel across cfg.N_JOBS workers.

    Produces two kinds of rows per file:
    - "Active": contiguous, non-overlapping W700 windows across the whole
      recording, each labeled with the file's MVR class (10% or 40%).
    - "Rest" (MVR_Class=0): genuine rest-period epochs found via gap-based
      detection (src.events.basal.extract_basal_epochs) in the gaps between
      detected MI/stimulus events — the SAME epochs used as the Δ (Active -
      Basal) reference, so there is one consistent definition of "basal" for
      this pipeline (see src/events/basal.py). Earlier revisions used "the
      first 5 contiguous W700 segments" as a Δ-only reference with no rest
      class; see INFORME_COMPLETO.md's methodology note for why that changed.
    """
    try:
        eeg, stims, meta = load_eeg(filepath)
    except Exception:
        return []

    mvr_level = meta.get("mvr_level", -1)
    if mvr_level not in (0, 10, 40):
        return []

    eeg_clean = preprocess_eeg(eeg)
    if cfg.USE_ICA:
        eeg_clean = apply_ica(eeg_clean)
    eeg_zs = z_score_normalize(eeg_clean)

    if eeg_zs.shape[1] < W:
        return []

    # Detect task events to find genuine rest gaps. Motor imagery files have
    # no discrete markers (ERD is a signal property, detected from eeg_clean
    # itself); stimulus-driven files' markers were already parsed once by
    # load_eeg (the `stims` return value) — reused here instead of
    # re-loading the raw .mat file a second time just for event detection.
    is_imagery = meta.get("paradigm", "ASYNCHRONOUS") == "ASYNCHRONOUS"
    if is_imagery:
        event_indices = detect_erd_events(eeg_clean, fs=cfg.FS)
    else:
        event_indices = stims if stims is not None else np.array([], dtype=int)

    window_sec = W / cfg.FS

    # Genuine rest epochs, W700-sized, in the same z-scored domain as the
    # active segments (extract_basal_epochs always returns >=1 epoch — it
    # falls back to the start of the recording when no qualifying gap is
    # found — so `basal` below is never built from zero epochs).
    basal_epochs_arr = extract_basal_epochs(eeg_zs, event_indices, fs=cfg.FS,
                                             segment_sec=window_sec)
    rest_segments = [basal_epochs_arr[b] for b in range(basal_epochs_arr.shape[0])]

    basal = build_basal_reference(rest_segments, n_basal=len(rest_segments), fs=cfg.FS)

    active_segments, active_starts = segment_fixed_window(eeg_zs, W, OVERLAP)
    if not active_segments:
        return []

    file_records = []

    for seg, start in zip(active_segments, active_starts):
        event_time = (start + W // 2) / cfg.FS
        file_records.append(_build_record(seg, mvr_level, event_time, meta, basal, window_sec))

    # Rest-class rows use a negative event_time_sec — these epochs come from
    # gap detection, not from the contiguous active-window grid, so a
    # positive "position in file" value would be misleading. Negative +
    # strictly decreasing keeps each row's key unique and immediately
    # recognizable as a rest epoch when inspecting the CSV.
    for b, seg in enumerate(rest_segments):
        event_time = -(b + 1) * window_sec
        file_records.append(_build_record(seg, 0, event_time, meta, basal, window_sec))

    return file_records


def main():
    print("=" * 60)
    print("FULL EXTRACTION W700 (2.8s, 0% overlap, per-channel+HRS+spectral)")
    print("=" * 60)

    t0 = time.time()
    files = list_mat_files()
    print("Found {} .mat files".format(len(files)))
    print("Parallel workers (cfg.N_JOBS): {}".format(cfg.N_JOBS))

    # Each file's full pipeline (load/filter/ICA/segment/features) is
    # independent of every other file — embarrassingly parallel across
    # cfg.N_JOBS worker processes instead of one sequential Python loop.
    results = Parallel(n_jobs=cfg.N_JOBS, verbose=5)(
        delayed(process_file)(filepath) for filepath in files
    )
    all_epochs = [record for file_records in results for record in file_records]

    elapsed = time.time() - t0
    print("\nExtracted {} epochs in {:.1f} min".format(len(all_epochs), elapsed / 60))

    if all_epochs:
        out_path = os.path.join(cfg.FEATURES_DIR, "Fixed_W700_3class_Features.csv")
        save_epoch_csv(all_epochs, out_path)
        print("Saved: {}".format(out_path))

        df = pd.DataFrame(all_epochs)
        print("Classes: {}".format(df["MVR_Class"].value_counts().sort_index().to_dict()))
        print("Sessions: {}".format(df["Session"].value_counts().sort_index().to_dict()))
        print("Patients: {}".format(df["Patient"].value_counts().sort_index().to_dict()))
        print("Columns: {}".format(len(df.columns)))


if __name__ == "__main__":
    # The __main__ guard is required for joblib's process-based backend on
    # Windows (spawn-based multiprocessing re-imports this module in each
    # worker; without the guard, extraction would recursively re-launch).
    main()
