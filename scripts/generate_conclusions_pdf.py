"""Generate a single-page PDF with the study conclusions."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
})

TITLE = "Classification of Post-Stroke Motor Imagery Using\nPer-Channel Fractal EEG Features: Conclusions"

CONCLUSIONS = """
1. Per-electrode fractal features with a standard SVM achieve 85.46% three-class accuracy at
   Month 6 post-stroke (chronic phase). The combination of four complementary fractal methods
   (RS, Higuchi, DFA, Semivariogram) across 16 channels (64 features) with SMOTE and an
   RBF-kernel SVM (C=12, γ=0.02) outperformed all other configurations (+17.7 pts over baseline).

2. Accuracy shows a progressive temporal trajectory correlating with neurological recovery.
   SVM evolved from 75.0% (Month 1) to 84.3% (Month 3) to 85.5% (Month 6). This convergence
   was consistent across all metrics (κ: +0.625 → +0.782; MAE: 7.50 → 4.50).

3. Higuchi's fractal dimension is the single most discriminative feature. Evaluated alone
   (16 features), Higuchi reached 81.2% (Month 6) and 83.5% (Month 3), outperforming DFA
   (77.0%), Semivariogram (78.0%), and RS (68.0%) by 3-13 points. At Month 3, Higuchi alone
   fell within 0.8 points of the four-method combination, yet the combined set was necessary to
   maximize MVR 40% detection (TPR 89% vs 78%).

4. The three Hurst exponent methods (HO, HRS, HV) degrade performance when added to the
   four basic fractal methods. The seven-method set (112 features) achieved 82.4% vs 85.5%
   (-3.1 pts), with rest-class TPR collapsing from 39% to 22%. This contradicts findings in
   healthy subjects (Martinez-Peon et al., 2024) and suggests the Hurst exponent is insufficient
   to capture the multifractal dynamics of post-stroke EEG.

5. Feature quality outweighs feature quantity. The four-method set (64 features) consistently
   outperformed the seven-method set (112 features). Per-electrode features outperformed
   spatial-mean aggregation by 15-28 points, confirming that topographic information is critical.

6. Inter-patient variability remains the principal obstacle to clinical deployment.
   Per-patient accuracy at Month 6 ranged from 77.8% (P1) to 90.5% (P3), with a standard
   deviation of ±5.1 pts. Leave-one-patient-out (LOPO) inter-subject evaluation reached only
   63.65%, underscoring the need for patient-specific calibration.

KEY METRICS (SVM, 3-class, Month 6):   Accuracy 85.46%  |  Kappa +0.749  |  MAE 4.50
                                       TPR Rest 38.8%   |  TPR MVR10% 91.3%  |  TPR MVR40% 88.8%
"""

fig, ax = plt.subplots(figsize=(8.5, 11))
ax.axis("off")

ax.text(0.5, 0.96, TITLE, transform=ax.transAxes, ha="center", va="top",
        fontsize=16, fontweight="bold", color="#2166AC")

ax.text(0.5, 0.91, "Niveles3 — Fractal EEG Classification Pipeline", transform=ax.transAxes,
        ha="center", va="top", fontsize=11, fontstyle="italic", color="#666666")

ax.text(0.05, 0.85, CONCLUSIONS, transform=ax.transAxes, ha="left", va="top",
        fontsize=9.5, linespacing=1.55, family="sans-serif")

ax.text(0.5, 0.04, "C:\\Users\\Desktop-UHC57\\OneDrive\\Escritorio\\Niveles3", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=7, color="#999999")

out_path = os.path.join(cfg.OUTPUT_DIR, "Conclusions.pdf")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
