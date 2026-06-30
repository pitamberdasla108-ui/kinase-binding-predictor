#!/usr/bin/env python3
"""plot_results.py — predicted vs true KIBA scatter for all three splits."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
splits = [("random","Random split"),("cold_drug","Cold-drug (unseen molecules)"),
          ("cold_target","Cold-target (unseen kinases)")]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax,(key,title) in zip(axes, splits):
    d = np.load(os.path.join(RES, f"preds_{key}.npz"))
    y, p = d["y"], d["pred"]
    r = pearsonr(p, y)[0]
    ax.scatter(y, p, s=3, alpha=0.15, color="#2b6cb0", edgecolors="none")
    lo, hi = min(y.min(),p.min()), max(y.max(),p.max())
    ax.plot([lo,hi],[lo,hi], "r--", lw=1)
    ax.set_xlabel("True KIBA score"); ax.set_ylabel("Predicted KIBA score")
    ax.set_title(f"{title}\nPearson r = {r:.3f}", fontsize=11)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
plt.suptitle("Kinase Binding Affinity Prediction (ESM-2 + Morgan FP -> MLP)", fontsize=13, y=1.02)
plt.tight_layout()
out = os.path.join(RES, "scatter_all_splits.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
print("Saved:", out)
