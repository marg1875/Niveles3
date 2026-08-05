"""Generate the heatmap alternative to per_channel_comparison.png.

Separate from scripts/plot_fractal_results.py on purpose: that script wipes
every PNG in output/plots/ and regenerates the original 8 figures. This one
only adds "per_channel_comparison_heatmap.png" alongside them, leaving
per_channel_comparison.png (and everything else) untouched.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.viz.plots_fractal import plot_per_channel_comparison_heatmap
import config as cfg

if __name__ == "__main__":
    plot_per_channel_comparison_heatmap(cfg.PLOTS_DIR)
