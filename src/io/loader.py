"""
Robust .mat EEG file loader with metadata extraction.

Handles:
- EEG_all vs B matrix (EEG_all is the full signal when present; B is full signal
  only when EEG_all is absent; B+EEG_all means B is a 10s preview — discarded)
- stims detection (80 or 320 markers, converts time vectors vs binary masks)
- Month extraction from normalized folder names (Month1, Month3, Month6)
- Paradigm detection (img keyword → ASYNCHRONOUS, else SYNCHRONOUS)
- MVR level detection (10 or 40, evaluating 40 before 10 to avoid false matches)
"""
import os
import re
import numpy as np
from scipy.io import loadmat
from typing import Optional, Dict, Any, Tuple, List

import config as cfg


def list_mat_files(data_dir: str = None) -> List[str]:
    """Find all .mat files recursively in data directory."""
    if data_dir is None:
        data_dir = cfg.DATA_DIR
    files = []
    for root, _, fnames in os.walk(data_dir):
        for fname in fnames:
            if fname.lower().endswith(".mat"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def _parse_month_from_folder(folder_name: str) -> int:
    """Extract month number from folder name.

    Normalized formats: Month1, Month3, Month6
    Also handles legacy: 1°MES, 3°MES, 6°MES, Mes1, etc.
    """
    normalized = folder_name.lower()
    # Direct match for normalized names
    for pattern, month in cfg.MONTH_FOLDER_PATTERNS.items():
        if pattern in normalized:
            return month
    # Fallback: extract first digit that matches 1,3,6
    digits = re.findall(r"\d+", normalized)
    for d in digits:
        n = int(d)
        if n in (1, 3, 6):
            return n
    return 0


def parse_metadata(filepath: str) -> Dict[str, Any]:
    """Extract patient ID, month, paradigm, and MVR level from file path.

    Path structure: data/Px.NNN/MonthN/pxNNN_LL[_img]_X.mat

    Returns dict with keys: paciente, month, paradigm, mvr_level
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    month_folder = os.path.basename(os.path.dirname(filepath))
    patient_folder = os.path.basename(os.path.dirname(os.path.dirname(filepath)))

    # --- Patient ID ---
    m = re.match(r"[Pp][xX]\.?(\d+)", basename)
    if m:
        paciente = f"Px.{m.group(1)}"
    else:
        m = re.search(r"(\d{3})", basename)
        paciente = f"Px.{m.group(1)}" if m else patient_folder

    # --- Month (from folder, never from filename) ---
    month = _parse_month_from_folder(month_folder)

    # --- Paradigm ---
    name_lower = basename.lower()
    is_imagery = any(kw in name_lower for kw in cfg.IMAGERY_KEYWORDS)

    # --- MVR level (check 40 BEFORE 10 to avoid false match on IDs like Px.0010) ---
    has_40 = bool(re.search(r"_40[^0-9]|_40$|40img|40%|_40img", name_lower))
    has_10 = bool(re.search(r"_10[^0-9]|_10$|10img|10%", name_lower)) and not has_40

    if has_40:
        mvr_level = 40
    elif has_10:
        mvr_level = 10
    else:
        mvr_level = 0

    return {
        "paciente": paciente,
        "month": month,
        "paradigm": "ASYNCHRONOUS" if is_imagery else "SYNCHRONOUS",
        "mvr_level": mvr_level,
        "filename": basename,
        "filepath": filepath,
    }


def load_eeg(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Load EEG signals and stimulus markers from a .mat file.

    Returns:
        eeg: (n_channels, n_samples) float64 array
        stims: (n_markers,) array or None
        meta: metadata dict from parse_metadata()
    """
    meta = parse_metadata(filepath)

    try:
        raw = loadmat(filepath, squeeze_me=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")

    eeg = _extract_eeg_matrix(raw, filepath)
    stims = _extract_stimulus_markers(raw, eeg)

    meta["n_channels"] = eeg.shape[0]
    meta["n_samples"] = eeg.shape[1]
    meta["duration_sec"] = eeg.shape[1] / cfg.FS

    return eeg, stims, meta


def _extract_eeg_matrix(raw: Dict[str, Any], filepath: str) -> np.ndarray:
    """Extract EEG data matrix from loaded .mat dict.

    Priority:
    1. EEG_all — if it exists and has enough samples (>= MIN_SIGNAL_DURATION)
    2. B — only if EEG_all is absent AND B has enough samples
       (when both exist, B is a 10s preview — discard)
    3. Generic fallback: any 2D numeric matrix with correct shape and size
    """
    # Priority: EEG_all first
    for field_name in ["EEG_all", "EEG", "eeg"]:
        if field_name in raw:
            val = np.array(raw[field_name], dtype=float)
            if val.ndim == 2 and val.size > cfg.MIN_SIGNAL_SAMPLES * cfg.N_CHANNELS:
                return _orient_eeg(val, filepath)

    # B as secondary — only when EEG_all is absent
    b_fields_present = [k for k in raw.keys() if k.startswith("B") and not k.startswith("__")]
    has_eeg_all = any("EEG" in k.upper() for k in raw.keys() if not k.startswith("__"))

    if not has_eeg_all and "B" in raw:
        val = np.array(raw["B"], dtype=float)
        if val.ndim == 2 and val.shape[1] > cfg.MIN_SIGNAL_SAMPLES:
            return _orient_eeg(val, filepath)

    # Generic search
    for field, val in raw.items():
        if field.startswith("__"):
            continue
        try:
            val = np.array(val, dtype=float)
        except (ValueError, TypeError):
            continue
        if val.ndim == 2 and cfg.N_CHANNELS in val.shape and val.size > 10000:
            return _orient_eeg(val, filepath)

    raise ValueError(f"No valid EEG matrix found in {filepath}. "
                     f"Available fields: {[k for k in raw.keys() if not k.startswith('__')]}")


def _orient_eeg(arr: np.ndarray, filepath: str) -> np.ndarray:
    """Ensure EEG is [channels × samples]."""
    if arr.shape[0] > arr.shape[1] and arr.shape[1] == cfg.N_CHANNELS:
        arr = arr.T

    if arr.shape[0] != cfg.N_CHANNELS:
        raise ValueError(
            f"Channel count mismatch in {filepath}: "
            f"got {arr.shape[0]}, expected {cfg.N_CHANNELS}. "
            f"Shape: {arr.shape}"
        )
    return np.nan_to_num(arr)


def _extract_stimulus_markers(raw: Dict[str, Any], eeg: np.ndarray) -> Optional[np.ndarray]:
    """Extract stimulus marker vector from loaded .mat dict.

    Handles three formats:
    1. Binary mask (values are 0/constant pattern) — detect rising edges
    2. Time vector in seconds — values within [0, duration], convert to samples
    3. Sample index vector — values already in sample indices, use directly
    """
    for field in cfg.MARKER_FIELDS:
        if field not in raw:
            continue
        try:
            stims_raw = np.array(raw[field], dtype=float).flatten()
        except (ValueError, TypeError):
            continue

        if len(stims_raw) == 0:
            continue

        n_samples = eeg.shape[1]
        unique_vals = np.unique(stims_raw)

        # Case 1: Binary mask — only 2 unique values (e.g. 0 and 10)
        #   - If len ≥ n_samples - 100: per-sample binary mask
        #   - If len < n_samples and looks like binary pattern: treat as binary
        if len(unique_vals) <= 3 and np.all(unique_vals >= 0):
            # Simple binary mask — find rising edges
            diff = np.diff(stims_raw)
            marks = np.where(diff > 0)[0] + 1

            # If the result count is reasonable and not all zero/near-zero
            if len(marks) > 0:
                # For short binary masks, indices are relative to mask resolution
                # Scale mask to full signal length
                if len(stims_raw) < n_samples:
                    scale = n_samples / len(stims_raw)
                    marks = np.round(marks * scale).astype(int)
                marks = marks[(marks > 0) & (marks < n_samples)]
                if len(marks) > 0:
                    return marks.astype(int)

        # Case 2: Short vector of time values or sample indices
        if len(stims_raw) < n_samples:
            # Check if values are monotonically increasing time stamps
            is_increasing = np.all(np.diff(stims_raw) >= 0)
            max_val = np.max(stims_raw)

            if is_increasing and max_val > 0:
                # If max is within [0, duration+5], treat as seconds
                if max_val <= (n_samples / cfg.FS) + 5:
                    marks = np.round(stims_raw * cfg.FS).astype(int)
                else:
                    # Sample indices
                    marks = np.round(stims_raw).astype(int)
            else:
                # Non-increasing short vector — try both interpretations
                if np.max(stims_raw) <= (n_samples / cfg.FS) + 5:
                    marks = np.round(stims_raw * cfg.FS).astype(int)
                else:
                    marks = np.round(stims_raw).astype(int)

            marks = marks[(marks > 0) & (marks < n_samples)]
            if len(marks) > 0:
                return marks.astype(int)

    return None


def scan_dataset(data_dir: str = None) -> List[Dict[str, Any]]:
    """Scan all .mat files and return metadata for each.

    Returns list of dicts with keys from parse_metadata() plus n_channels,
    n_samples, duration_sec, and has_stims (bool).
    """
    files = list_mat_files(data_dir)
    results = []

    for fpath in files:
        try:
            meta = parse_metadata(fpath)
            raw = loadmat(fpath, squeeze_me=False)

            # Quickly estimate signal info without full extraction
            n_ch = None
            n_sm = None
            for field in cfg.EEG_FIELDS:
                if field in raw:
                    val = np.array(raw[field], dtype=float)
                    if val.ndim == 2:
                        n_ch = val.shape[0] if val.shape[0] > val.shape[1] else val.shape[1]
                        n_sm = val.shape[1] if val.shape[0] > val.shape[1] else val.shape[0]
                        break

            has_stims = any(f in raw for f in cfg.MARKER_FIELDS)

            meta.update({
                "n_channels": n_ch,
                "n_samples": n_sm,
                "duration_sec": n_sm / cfg.FS if n_sm else 0,
                "has_stims": has_stims,
            })
            results.append(meta)

        except Exception as e:
            print(f"[WARNING] Could not scan {os.path.basename(fpath)}: {e}")
            continue

    return results
