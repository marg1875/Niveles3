import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.viz.plots import (
    plot_ranking,
    plot_accuracy_vs_channels,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison,
    plot_recall_per_class,
)

def main():
    print("=" * 55)
    print("NIVELES3 — GENERATING CLASSIFICATION PLOTS")
    print("=" * 55)
    
    results_csv = os.path.join(cfg.CLASSIFICATION_DIR, "results_lopo.csv")
    if not os.path.isfile(results_csv):
        print(f"[ERROR] Results CSV not found: {results_csv}")
        print("Run classify.py first.")
        return
    
    roc_data_dir = cfg.ROC_DATA_DIR
    output_dir = cfg.PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results: {results_csv}")
    print(f"ROC data: {roc_data_dir}")
    print(f"Plots: {output_dir}")
    print()
    
    # Generate all plots
    plots = [
        ("Model ranking (top 20)", plot_ranking, [results_csv, output_dir]),
        ("Accuracy vs #electrodes", plot_accuracy_vs_channels, [results_csv, output_dir]),
        ("Confusion matrices", plot_confusion_matrices, [roc_data_dir, results_csv, output_dir]),
        ("ROC curves", plot_roc_curves, [roc_data_dir, results_csv, output_dir]),
        ("Model comparison", plot_model_comparison, [results_csv, output_dir]),
        ("Recall per class", plot_recall_per_class, [results_csv, output_dir]),
    ]
    
    for name, func, args in plots:
        try:
            print(f"[{plots.index((name,func,args))+1}/{len(plots)}] {name}...")
            func(*args)
            print(f"  [OK]")
        except Exception as e:
            print(f"  [WARN] {e}")
    
    generated = os.listdir(output_dir)
    print(f"\n{'='*55}")
    print(f"COMPLETE — {len([f for f in generated if f.endswith('.png')])} plots saved to {output_dir}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
