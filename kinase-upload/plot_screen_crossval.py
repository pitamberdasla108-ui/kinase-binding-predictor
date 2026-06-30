#!/usr/bin/env python3
"""plot_screen_crossval.py — ML vs docking on the top-5 virtual-screen hits."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE="/home/pitamber_das/kinase-ml"
ml = pd.read_csv(os.path.join(HERE,"results","cabl_virtual_screen.csv")).head(5).set_index("Drug_ID")["pred_KIBA"]
dock = pd.read_csv("/home/pitamber_das/gpudock/ml_hits/dock_results.csv").set_index("name")["docking_kcal"]

names = list(ml.index)
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
a1.barh(range(len(names)),[ml[n] for n in names],color="#2b6cb0")
a1.set_yticks(range(len(names))); a1.set_yticklabels(names,fontsize=8); a1.invert_yaxis()
a1.set_xlabel("ML predicted KIBA"); a1.set_xlim(14,15); a1.set_title("ML model ranking")
a2.barh(range(len(names)),[dock[n] for n in names],color="#c05621")
a2.set_yticks(range(len(names))); a2.set_yticklabels(names,fontsize=8); a2.invert_yaxis()
a2.set_xlabel("Docking kcal/mol (lower=stronger)"); a2.set_title("Docking ranking (independent method)")
plt.suptitle("Virtual screen cross-validation: ML vs docking on top-5 c-Abl hits\n(methods diverge on small polyphenols, agree on drug-like CHEMBL1684800)",y=1.04,fontsize=11)
plt.tight_layout()
out=os.path.join(HERE,"results","screen_crossval.png")
plt.savefig(out,dpi=130,bbox_inches="tight"); print("Saved:",out)
