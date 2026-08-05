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
from src.features.fractal import compute_all_fractals, compute_hrs_vector
from src.features.spectral import compute_spectral_features
from src.features.basal_reference import build_basal_reference

W = 700  # samples = 2.8s at 250 Hz
OVERLAP = 0.0
N_BASAL = 5


def process_file(filepath):
    """Load, preprocess and extract all W700 epoch records for one .mat file.

    Returns a list of record dicts (possibly empty if the file is skipped).
    Factored out of the old flat per-file loop so it can run in a worker
    process via joblib.Parallel — each file's pipeline (load -> filter ->
    ICA -> segment -> fractal/spectral features) is independent of every
    other file, so this is embarrassingly parallel across cfg.N_JOBS workers.
    """
    try:
        eeg, _, meta = load_eeg(filepath)
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

    segments, starts = segment_fixed_window(eeg_zs, W, OVERLAP)
    if not segments:
        return []

    basal = build_basal_reference(segments, n_basal=N_BASAL, fs=cfg.FS)
    window_sec = W / cfg.FS

    file_records = []
    for i, (seg, start) in enumerate(zip(segments, starts)):
        event_time = (start + W // 2) / cfg.FS

        # All segments (including the first N_BASAL used as the Δ reference)
        # carry the file's MVR class label. The basal segments are used only
        # to compute the Δ (Active - Basal) fractal features and the ERD
        # ratios — they are not relabeled as a separate "rest" class,
        # matching the published methodology (INFORME_COMPLETO.md) and the
        # convention used in the reference literature (Martinez-Peon 2024).
        # See src/events/basal.py for the full basal-convention note.
        seg_class = mvr_level

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

        record = build_feature_record(
            patient=meta["paciente"], month=meta["month"],
            paradigm=meta.get("paradigm", "ASYNCHRONOUS"),
            mvr_class=seg_class, event_time_sec=event_time,
            basal_global=basal.basal_global, stable_features=stable,
            original_filename=meta["filename"],
            per_channel_features={"basal_per_ch": basal.basal_per_ch,
                                   "active_per_ch": pc_features[:, :8]},
            martinez_features=martinez_features, spectral_features=spectral,
        )
        file_records.append(record)

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
