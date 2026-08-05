# scripts/legacy/

Scripts moved out of `scripts/` because nothing in the repo reads their
output and nothing they read is still generated. Kept for reference rather
than deleted — verified as of this audit (see git log around this commit).

- **`fast_intra.py`** — a stripped-down dev/test copy of `classify.py`'s
  intra-subject CV path. Hardcodes an absolute path to a specific machine
  (`sys.path.insert(0, r"C:\Users\...")`), which nothing else in the repo
  does. Its output (`results_intra_fast.csv`) is not referenced by
  `INFORME_COMPLETO.md` or `INFORME_FRACTAL.md`.
- **`extract_features_fixed.py`** — writes `FixedA_Features.csv` /
  `FixedB_Features.csv` (two ERD-detection strategies). No other script in
  the repo reads either file.

## Important: most other "candidate" scripts were NOT moved here

An earlier pass of this audit assumed most of `scripts/*.py` besides
`extract_w700_full.py` / `classify_w700_full.py` was safe to archive as
superseded. That assumption was wrong and was corrected before moving
anything — several scripts turn out to feed content that is still live:

| Script(s) | Still feeds |
|---|---|
| `extract_features.py`, `classify.py` | `Global_All_Features.csv` → `results_lopo.csv`, whose LOPO numbers (63.65% etc.) are still cited in `INFORME_COMPLETO.md` §6.1/§8.3 |
| `extract_multi_window.py`, `fast_sweep.py`, `classify_windows.py`, `classify_sweep_fast.py` | The 16-window-size sweep table in `INFORME_COMPLETO.md` §4 |
| `classify_fractal_extended.py`, `classify_fractal_individual.py`, `analyze_per_patient.py`, `hyperparameter_tuning.py`, `hyperparameter_tuning_fine.py`, `plot_svm_analysis.py`, `plot_tuned_results.py`, `generate_conclusions_pdf.py` | **`INFORME_FRACTAL.md`** — a separate, still-active document (SMOTE-based 3-class per-channel analysis, 85.46% headline) that this audit was not scoped to touch |

**Follow-up worth flagging separately:** `classify_fractal_extended.py` and
`classify_fractal_individual.py` both read `Fixed_W700_3class_Features.csv`
— the same feature file this session's audit regenerated (basal-labeling
fix + `HRS_p64` correction, see the methodology note at the top of
`INFORME_COMPLETO.md`). If `INFORME_FRACTAL.md`'s numbers were produced from
those scripts' output, they may now be stale relative to the regenerated
feature file. That document was out of scope for this session and was not
touched.

## What Fase 4 of the audit actually consolidated

Rather than archiving, the real duplication (the same ~25-line blocks
copy-pasted 4-6 times across extraction/classification scripts) was
extracted into shared functions, and the two "final" scripts were rewritten
to use them:

- `src/preprocessing/segmentation.py` — `segment_fixed_window()`
- `src/features/basal_reference.py` — `build_basal_reference()`
- `src/classification/validation.py` — `run_intra_subject_pool_cv()` (next to the existing `lopo_split()`)

Verified byte-identical output before/after this consolidation (see commit
history / session notes): `extract_w700_full.py`'s regenerated CSV had 0 max
absolute difference across all 412 numeric columns, and
`classify_w700_full.py`'s `classify_condition()` matched the previously
saved `results_W700_full_intra.csv` exactly on spot-checked conditions.
