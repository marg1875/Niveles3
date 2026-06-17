# Niveles3 — EEG Fractal Feature Extraction for ECV Patients

## Overview

Research pipeline for extracting **fractal dimension features** (RS, Higuchi, DFA, Variogram) from 16-channel EEG recordings of stroke (ECV) patients performing motor tasks at three Motor Voluntary Resistance (MVR) levels: 0%, 10%, 40%.

Two paradigms are analysed:
- **ASYNCHRONOUS** — motor imagery (ERD-based event detection)
- **SYNCHRONOUS** — stimulus-driven execution (marker-based event detection)

Data collected at Month 1, Month 3, and Month 6 of neurorehabilitation.

## Channel Montage

```
Fp1, Fp3, F3, Fz, F4    (frontal)
Cz                        (central-motor)
Pz, P5, P3                (parietal)
O1, Oz                    (occipital)
T8, P8, P6, P4            (temporo-parietal)
T7                        (temporal left)
```

## Quick Start

```bash
pip install -r requirements.txt

# Extract fractal features from all .mat files
python scripts/extract_features.py

# Prepare WEKA classification datasets
python scripts/prepare_weka.py
```

## Project Structure

```
Niveles3/
├── config.py              Central configuration (no magic numbers)
├── src/                   Reusable library modules
│   ├── io/                Loader (.mat) + Exporter (CSV/ARFF)
│   ├── preprocessing/     Filters, ICA
│   ├── events/            Event detection, basal epochs
│   ├── features/          Fractal algorithms + stability validation
│   ├── stats/             Wilcoxon, FDR, Spearman (future)
│   └── viz/               Plotting utilities (future)
├── scripts/               Runnable pipeline scripts
├── data/                  Input .mat EEG files (patient folders)
└── output/                All generated results
    ├── features/          Per-file and global feature CSVs
    └── weka/              .arff and .csv for WEKA
```

## Key Improvements over Niveles2

| Issue (Niveles2) | Fix (Niveles3) |
|---|---|
| Wrong motor channel (index 7 = Pz) | Correct index 5 = Cz |
| Fixed epoch window (3s all methods) | Adaptive parameters per method |
| Fixed fractal params (k_max=8, lag≤15) | k_max ∝ N//4, lags ∝ N//5 |
| No stability validation | Multi-window CV + stable/marginal/unstable flag |
| Double Z-score (R/S + preprocess) | Single Z-score in preprocess only |
| Encoding issues (1°MES → 1A MES) | Normalized folder names (Month1, Month3, Month6) |
| Monolithic MATLAB/Python duplication | Pure Python, modular src/ architecture |

## Feature Output Columns

Each epoch row includes:
- Metadata: Patient, Session, Month, Paradigm, MVR_Class, Event_Time
- Basal features: RS, Higuchi, DFA, Variogram, Average (global baseline)
- Active features: same 5 methods (during motor task)
- Delta features: Active - Basal (for classification)
- Stability metrics: window_sec, CV, stability_flag per method
