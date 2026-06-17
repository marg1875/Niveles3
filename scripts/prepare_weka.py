"""
Prepare WEKA-ready datasets from extracted features.

Converts Global_All_Features.csv into .arff and .csv files
with configurable feature subsets (DELTA, ACTIVO, ACTIVO_DELTA, TODOS).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.io.exporter import load_global_features, prepare_weka_dataset


def main():
    print("=" * 55)
    print("NIVELES3 — PREPARE WEKA DATASET")
    print("=" * 55)

    global_csv = os.path.join(cfg.FEATURES_DIR, "Global_All_Features.csv")
    if not os.path.isfile(global_csv):
        print(f"[ERROR] Global features not found: {global_csv}")
        print("Run extract_features.py first.")
        return

    print(f"Loading: {global_csv}")
    df = load_global_features(global_csv)
    print(f"Total epochs: {len(df)}")
    print(f"Classes: {sorted(df['MVR_Class'].unique())}")
    print()

    # Generate WEKA datasets for all subsets
    for subset_name in cfg.WEKA_FEATURE_SUBSETS:
        print(f"Preparing WEKA dataset: '{subset_name}'")
        prepare_weka_dataset(df, subset=subset_name)
        print(f"  [OK] Generated .csv and .arff files for {subset_name}")

    print(f"\nWEKA-ready files saved to: {cfg.WEKA_DIR}")
    print("\nNext steps:")
    print("  1. Open WEKA Explorer")
    print("  2. Preprocess -> Open file -> Select .arff from output/weka/")
    print("  3. Classify -> Choose classifier (J48, RandomForest, etc.)")
    print("  4. Cross-validation (10 folds)")


if __name__ == "__main__":
    main()
