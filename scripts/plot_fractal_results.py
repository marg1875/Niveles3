"""Generate all 8 figures for INFORME_FRACTAL.md - Martinez-Peon 2024 style."""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.viz.plots_fractal import (
    plot_accuracy_per_patient, plot_confusion_matrix_best,
    plot_per_channel_comparison, plot_channel_scaling,
    plot_spatial_vs_perchannel, plot_kappa_comparison,
    plot_feature_distribution, plot_evolution_months,
)
import config as cfg

print("=" * 60)
print("GENERANDO FIGURAS PARA INFORME_FRACTAL")
print("=" * 60)

t0 = time.time()
output_dir = cfg.PLOTS_DIR

# Remove old plots
old_pngs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
if old_pngs:
    for p in old_pngs:
        os.remove(os.path.join(output_dir, p))
    print(f"Eliminados {len(old_pngs)} PNGs antiguos")

plots = [
    ("Accuracy per patient", plot_accuracy_per_patient),
    ("Confusion matrix best", plot_confusion_matrix_best),
    ("Per-channel comparison", plot_per_channel_comparison),
    ("Channel scaling", plot_channel_scaling),
    ("Spatial vs Per-channel", plot_spatial_vs_perchannel),
    ("Kappa comparison", plot_kappa_comparison),
    ("Feature distribution", plot_feature_distribution),
    ("Evolution months", plot_evolution_months),
]

for i, (name, func) in enumerate(plots, 1):
    print(f"[{i}/{len(plots)}] {name}...")
    try:
        func(output_dir)
    except Exception as e:
        print(f"  [ERROR] {e}")

elapsed = time.time() - t0
pngs = sorted(f for f in os.listdir(output_dir) if f.endswith(".png"))

print(f"\n{'='*60}")
print(f"COMPLETO — {len(pngs)} PNGs en {elapsed:.0f}s")
for p in pngs:
    size_kb = os.path.getsize(os.path.join(output_dir, p)) / 1024
    print(f"  {p} ({size_kb:.0f} KB)")
