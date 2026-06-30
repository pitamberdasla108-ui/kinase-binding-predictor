#!/usr/bin/env python3
"""
load_data.py — Load the KIBA kinase-inhibitor binding benchmark.
KIBA: ~118k interactions, 229 kinases x 2,111 inhibitors.
Each row = kinase sequence + inhibitor SMILES -> KIBA binding score.
Source: Tang et al. 2014; DeepDTA benchmark. Loaded via TDC.
"""
import os
import pandas as pd
from tdc.multi_pred import DTI

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("Loading KIBA via TDC (downloads on first run)...")
    data = DTI(name="KIBA")
    df = data.get_data()

    print("\n=== KIBA dataset ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Unique drugs (inhibitors):", df["Drug_ID"].nunique())
    print("Unique targets (kinases) :", df["Target_ID"].nunique())
    print("KIBA score range: %.2f to %.2f" % (df["Y"].min(), df["Y"].max()))
    print("\nFirst 3 rows:")
    with pd.option_context("display.max_colwidth", 60):
        print(df.head(3).to_string())

    out_csv = os.path.join(OUT_DIR, "kiba_raw.csv")
    df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

if __name__ == "__main__":
    main()
