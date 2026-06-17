"""Quick test of multi-window extraction on 1 file with 3 window sizes."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
import numpy as np
from src.io.loader import load_eeg, list_mat_files
from src.preprocessing.filters import preprocess_eeg, z_score_normalize
from src.preprocessing.ica import apply_ica
from src.features.fractal import compute_all_fractals, compute_hrs_vector
from src.features.spectral import compute_spectral_features
from src.io.exporter import build_feature_record

files = list_mat_files()
test_file = None
for f in files:
    try:
        eeg, _, meta = load_eeg(f)
        if meta.get("mvr_level") == 10 and meta.get("paradigm") == "ASYNCHRONOUS":
            test_file = f
            break
    except:
        continue

print(f"Testing: {os.path.basename(test_file)}")
eeg, _, meta = load_eeg(test_file)
print(f"  Shape={eeg.shape}, Duration={eeg.shape[1]/cfg.FS:.1f}s")
print(f"  Patient={meta['paciente']} Month={meta['month']} MVR={meta['mvr_level']}")

t0 = time.time()
eeg_clean = preprocess_eeg(eeg)
if cfg.USE_ICA:
    eeg_clean = apply_ica(eeg_clean)
eeg_zs = z_score_normalize(eeg_clean)
print(f"  Preprocess: {time.time()-t0:.1f}s")

# Test 3 window sizes
test_windows = [500, 1000, 2000]
for w in test_windows:
    n_samples = eeg_zs.shape[1]
    segments = []
    for s in range(0, n_samples - w + 1, w):
        segments.append(eeg_zs[:, s:s + w])
    print(f"\n  W{w} ({w/cfg.FS:.1f}s): {len(segments)} segments")

    if len(segments) > 0:
        t_feat = time.time()
        # Basal
        n_basal = min(5, len(segments))
        basal_feats = np.array([compute_all_fractals(segments[b]) for b in range(n_basal)])
        basal_global = np.nanmean(basal_feats, axis=(0, 1))
        # One segment
        pc = compute_all_fractals(segments[0])
        hrs = compute_hrs_vector(np.nanmean(segments[0], axis=0), fs=cfg.FS)
        spec = compute_spectral_features(segments[0], fs=cfg.FS, basal_segment=np.nanmean(np.stack(segments[:n_basal]), axis=0))
        print(f"    Feature time per seg: {(time.time()-t_feat)/(len(segments)+1):.2f}s")
        print(f"    Basal: RS={basal_global[0]:.3f} Hig={basal_global[1]:.3f} DFA={basal_global[2]:.3f}")
        print(f"    PC-shape={pc.shape}, HRS-keys={list(hrs.keys())[:3]}..., Spec-keys={list(spec.keys())[:3]}...")

print(f"\nAll good! Total: {time.time()-t0:.1f}s")
