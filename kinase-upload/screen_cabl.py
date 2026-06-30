#!/usr/bin/env python3
"""screen_cabl.py — Virtual screen: predict c-Abl binding for the ~1039 KIBA drugs
NOT measured against c-Abl. Honest computational triage, not a discovery claim."""
import os, numpy as np, pandas as pd, torch
import torch.nn as nn

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
CABL="P00519"

class MLP(nn.Module):
    def __init__(s,in_dim=2688,hidden=(1024,512,128),p=0.3):
        super().__init__(); L=[]; d=in_dim
        for h in hidden: L+=[nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(p)]; d=h
        L+=[nn.Linear(d,1)]; s.net=nn.Sequential(*L)
    def forward(s,x): return s.net(x).squeeze(-1)

def main():
    df = pd.read_csv(os.path.join(DATA,"kiba_raw.csv"))
    prot = dict(np.load(os.path.join(DATA,"prot_emb.npz")))
    drug = dict(np.load(os.path.join(DATA,"drug_fp.npz")))
    pemb = prot[CABL]

    cabl_drugs = set(df[df["Target_ID"]==CABL]["Drug_ID"])
    smi_map = df.drop_duplicates("Drug_ID").set_index("Drug_ID")["Drug"].to_dict()
    untested = [d for d in drug if d not in cabl_drugs]
    print(f"Screening {len(untested)} untested drugs against c-Abl...")

    model = MLP().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(HERE,"results","model_random.pt"),map_location=DEVICE))
    model.eval()

    X = np.stack([np.concatenate([pemb, drug[d].astype(np.float32)]) for d in untested])
    preds=[]
    with torch.no_grad():
        for i in range(0,len(X),512):
            xb = torch.from_numpy(X[i:i+512]).float().to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)

    res = pd.DataFrame({"Drug_ID":untested,"pred_KIBA":preds})
    res["SMILES"] = res["Drug_ID"].map(smi_map)
    res = res.sort_values("pred_KIBA", ascending=False).reset_index(drop=True)

    measured = df[df["Target_ID"]==CABL]["Y"]
    print(f"\nMEASURED c-Abl scores: median {measured.median():.2f}, 90th pct {measured.quantile(0.9):.2f}, max {measured.max():.2f}")
    print(f"PREDICTED (untested):  median {res['pred_KIBA'].median():.2f}, max {res['pred_KIBA'].max():.2f}\n")

    print("=== Top 15 predicted c-Abl binders (untested = computational hypotheses) ===")
    print(res.head(15)[["Drug_ID","pred_KIBA"]].to_string(index=False))

    out = os.path.join(HERE,"results","cabl_virtual_screen.csv")
    res.to_csv(out, index=False)
    thr = measured.quantile(0.9)
    print(f"\nSaved: {out}")
    print(f"Predicted above c-Abl 90th-percentile ({thr:.1f}): {(res['pred_KIBA']>thr).sum()} compounds")

if __name__ == "__main__":
    main()
