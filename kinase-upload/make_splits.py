#!/usr/bin/env python3
"""make_splits.py — random / cold-drug / cold-target splits for honest eval."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED, TEST_FRAC = 42, 0.2

def random_split(df, rng):
    idx = np.arange(len(df)); rng.shuffle(idx)
    n_test = int(len(idx)*TEST_FRAC)
    return idx[n_test:], idx[:n_test]

def cold_split(df, col, rng):
    ent = df[col].unique(); rng.shuffle(ent)
    test_ent = set(ent[:int(len(ent)*TEST_FRAC)])
    m = df[col].isin(test_ent).values
    return np.where(~m)[0], np.where(m)[0]

def main():
    df = pd.read_csv(os.path.join(DATA,"kiba_raw.csv"))
    rng = np.random.default_rng(SEED)
    splits = {}
    tr,te = random_split(df,rng);            splits["random_train"],splits["random_test"]=tr,te
    tr,te = cold_split(df,"Drug_ID",rng);    splits["cold_drug_train"],splits["cold_drug_test"]=tr,te
    tr,te = cold_split(df,"Target_ID",rng);  splits["cold_target_train"],splits["cold_target_test"]=tr,te
    np.savez_compressed(os.path.join(DATA,"splits.npz"), **splits)
    print("Split sizes (train / test):")
    for name in ["random","cold_drug","cold_target"]:
        tr,te = splits[f"{name}_train"], splits[f"{name}_test"]
        extra=""
        if name=="cold_drug":
            ov=set(df.iloc[tr]["Drug_ID"])&set(df.iloc[te]["Drug_ID"]); extra=f" | drug overlap: {len(ov)} (want 0)"
        if name=="cold_target":
            ov=set(df.iloc[tr]["Target_ID"])&set(df.iloc[te]["Target_ID"]); extra=f" | kinase overlap: {len(ov)} (want 0)"
        print(f"  {name:12s}: {len(tr):6d} / {len(te):6d}{extra}")

if __name__ == "__main__":
    main()
