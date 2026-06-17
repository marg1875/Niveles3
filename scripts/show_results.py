import pandas as pd
import numpy as np

csv_path = r"C:\Users\Desktop-UHC57\OneDrive\Escritorio\Niveles3\output\classification\results_lopo.csv"
df = pd.read_csv(csv_path)
valid = df.dropna(subset=["accuracy_mean"])

print("=" * 70)
print("BEST PER CHANNEL CONFIGURATION (3-class)")
print("=" * 70)
v3 = valid[valid["Class_Type"] == "3class"]
idx = v3.groupby("Channel_Config")["accuracy_mean"].idxmax()
for _, r in v3.loc[idx].sort_values("accuracy_mean", ascending=False).iterrows():
    print(f"  {r['Channel_Config']:12s} ({int(r['N_Channels'])}ch)  "
          f"{r['Model']:18s}  {r['Feature_Subset']:16s}  "
          f"Acc={r['accuracy_mean']:.3f}  F1={r['f1_macro_mean']:.3f}  "
          f"R0={r['recall_cls0_mean']:.2f} R10={r['recall_cls10_mean']:.2f} "
          f"R40={r['recall_cls40_mean']:.2f}")

print()
print("=" * 70)
print("BEST PER MODEL (3-class, avg across configs)")
print("=" * 70)
bm = v3.groupby("Model")["accuracy_mean"].agg(["mean","std","max"]).sort_values("mean", ascending=False)
print(bm.to_string())

print()
print("=" * 70)
print("BEST PER FEATURE SUBSET (3-class, avg)")
print("=" * 70)
bf = v3.groupby("Feature_Subset")["accuracy_mean"].agg(["mean","std","max"]).sort_values("mean", ascending=False)
print(bf.to_string())

print()
print("=" * 70)
print("BEST 2-CLASS (10pct vs 40pct)")
print("=" * 70)
v2 = valid[valid["Class_Type"] == "2class"]
idx2 = v2.groupby("Channel_Config")["accuracy_mean"].idxmax()
for _, r in v2.loc[idx2].sort_values("accuracy_mean", ascending=False).iterrows():
    print(f"  {r['Channel_Config']:12s} ({int(r['N_Channels'])}ch)  "
          f"{r['Model']:18s}  {r['Feature_Subset']:16s}  "
          f"Acc={r['accuracy_mean']:.3f}  "
          f"R10={r.get('recall_cls10_mean',0):.2f}  "
          f"R40={r.get('recall_cls40_mean',0):.2f}")
