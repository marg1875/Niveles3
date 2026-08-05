"""Generate all figures for INFORME_COMPLETO.md (W700 pipeline results).

Separate from scripts/plot_fractal_results.py (INFORME_FRACTAL.md's figures)
— reads results_W700_full_intra.csv, doesn't touch or regenerate anything
under the other pipeline's output.
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.viz.plots_w700 import (
    plot_accuracy_by_month,
    plot_fractal_vs_spectral,
    plot_rest_epoch_distribution,
    plot_confusion_matrix_3class_best,
)
import config as cfg

print("=" * 60)
print("GENERANDO FIGURAS PARA INFORME_COMPLETO.md (W700)")
print("=" * 60)

t0 = time.time()
output_dir = cfg.PLOTS_DIR

plots = [
    ("Accuracy by month", plot_accuracy_by_month),
    ("Fractal vs spectral", plot_fractal_vs_spectral),
    ("Rest epoch distribution", plot_rest_epoch_distribution),
    ("Confusion matrix 3-class (Month 6)", plot_confusion_matrix_3class_best),
]

for i, (name, func) in enumerate(plots, 1):
    print(f"[{i}/{len(plots)}] {name}...")
    func(output_dir)

elapsed = time.time() - t0
print(f"\nCompleto en {elapsed:.0f}s")
