"""Full extraction for best window W700 (2.8s) with per-channel + HRS + spectral."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from tqdm import tqdm
from src.io.loader import load_eeg, list_mat_files
from src.io.exporter import save_epoch_csv, build_feature_record
from src.preprocessing.filters import preprocess_eeg, z_score_normalize
from src.preprocessing.ica import apply_ica
from src.features.fractal import compute_all_fractals, compute_hrs_vector
from src.features.spectral import compute_spectral_features

W = 700  # samples = 2.8s at 250 Hz
OVERLAP = 0.0

print("=" * 60)
print("FULL EXTRACTION W700 (2.8s, 0% overlap, per-channel+HRS+spectral)")
print("=" * 60)

t0 = time.time()
files = list_mat_files()
print("Found {} .mat files".format(len(files)))

all_epochs = []

for fi, filepath in enumerate(tqdm(files, desc="Files"), 1):
    try:
        eeg, _, meta = load_eeg(filepath)
    except Exception:
        continue

    mvr_level = meta.get("mvr_level", -1)
    if mvr_level not in (0, 10, 40):
        continue

    eeg_clean = preprocess_eeg(eeg)
    if cfg.USE_ICA:
        eeg_clean = apply_ica(eeg_clean)
    eeg_zs = z_score_normalize(eeg_clean)

    n_samples = eeg_zs.shape[1]
    if n_samples < W:
        continue

    segments = []
    starts = []
    for s in range(0, n_samples - W + 1, W):
        segments.append(eeg_zs[:, s:s + W])
        starts.append(s)

    if not segments:
        continue

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

    basal_martinez = {}
    if not np.all(np.isnan(basal_global)):
        basal_martinez = {"HO": float(basal_global[5]), "HV": float(basal_global[7])}
        if basal_ref_segment is not None:
            hrs_basal = compute_hrs_vector(np.nanmean(basal_ref_segment, axis=0), fs=cfg.FS)
            basal_martinez.update(hrs_basal)
    else:
        basal_martinez = {"HO": np.nan, "HV": np.nan}
        for p in cfg.HRS_PARTITION_VALUES:
            basal_martinez["HRS_p{}".format(p)] = np.nan

    window_sec = W / cfg.FS

    for i, (seg, start) in enumerate(zip(segments, starts)):
        event_time = (start + W // 2) / cfg.FS

        # First n_basal segments are class 0 (basal/rest)
        seg_class = 0 if i < n_basal else mvr_level

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
        martinez_features = {"basal": basal_martinez, "active": active_martinez}

        spectral = compute_spectral_features(seg, fs=cfg.FS, basal_segment=basal_ref_segment)

        record = build_feature_record(
            patient=meta["paciente"], month=meta["month"],
            paradigm=meta.get("paradigm", "ASYNCHRONOUS"),
            mvr_class=seg_class, event_time_sec=event_time,
            basal_global=basal_global, stable_features=stable,
            original_filename=meta["filename"],
            per_channel_features={"basal_per_ch": basal_per_ch, "active_per_ch": pc_features[:, :8]},
            martinez_features=martinez_features, spectral_features=spectral,
        )
        all_epochs.append(record)

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
