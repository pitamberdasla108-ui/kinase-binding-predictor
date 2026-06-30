#!/usr/bin/env python3
"""train.py — MLP to predict KIBA binding affinity. Reports RMSE, Pearson, CI."""
import os, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PairDataset(Dataset):
    def __init__(self, df, idx, prot, drug):
        self.rows = df.iloc[idx].reset_index(drop=True); self.prot, self.drug = prot, drug
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = np.concatenate([self.prot[r["Target_ID"]], self.drug[r["Drug_ID"]].astype(np.float32)])
        return torch.from_numpy(x), torch.tensor(r["Y"], dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, in_dim=2688, hidden=(1024,512,128), p=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]; d = h
        layers += [nn.Linear(d,1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def concordance_index(y, f):
    y = np.asarray(y); f = np.asarray(f); n = len(y)
    rng = np.random.default_rng(0)
    if n > 4000:
        sel = rng.choice(n, 4000, replace=False); y, f = y[sel], f[sel]; n = 4000
    num = den = 0
    for i in range(n):
        dy = y[i]-y; mask = dy > 0; den += mask.sum()
        num += (f[i] > f[mask]).sum() + 0.5*(f[i] == f[mask]).sum()
    return num/den if den else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random","cold_drug","cold_target"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(DATA,"kiba_raw.csv"))
    prot = dict(np.load(os.path.join(DATA,"prot_emb.npz")))
    drug = dict(np.load(os.path.join(DATA,"drug_fp.npz")))
    sp = np.load(os.path.join(DATA,"splits.npz"))
    tr_idx, te_idx = sp[f"{args.split}_train"], sp[f"{args.split}_test"]
    valid = df["Drug_ID"].isin(drug.keys()).values
    tr_idx = tr_idx[valid[tr_idx]]; te_idx = te_idx[valid[te_idx]]

    tr_ds = PairDataset(df, tr_idx, prot, drug); te_ds = PairDataset(df, te_idx, prot, drug)
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    model = MLP().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=args.lr); lossfn = nn.MSELoss()
    print(f"Split={args.split} | train={len(tr_ds)} test={len(te_ds)} | device={DEVICE}")
    for ep in range(1, args.epochs+1):
        model.train(); t0 = time.time(); tot = 0
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = lossfn(model(x), y); loss.backward(); opt.step()
            tot += loss.item()*len(y)
        tr_rmse = (tot/len(tr_ds))**0.5
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:2d} | train RMSE {tr_rmse:.4f} | {time.time()-t0:.1f}s")

    model.eval(); preds, ys = [], []
    with torch.no_grad():
        for x, y in te_dl:
            preds.append(model(x.to(DEVICE)).cpu().numpy()); ys.append(y.numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
    rmse = float(np.sqrt(((preds-ys)**2).mean())); r = float(pearsonr(preds, ys)[0]); ci = float(concordance_index(ys, preds))
    print(f"\n=== TEST ({args.split}) ===\n  RMSE : {rmse:.4f}\n  Pearson r : {r:.4f}\n  CI : {ci:.4f}")
    os.makedirs(os.path.join(HERE,"results"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(HERE,"results",f"model_{args.split}.pt"))
    with open(os.path.join(HERE,"results",f"metrics_{args.split}.txt"),"w") as fh:
        fh.write(f"split={args.split}\nRMSE={rmse:.4f}\nPearson={r:.4f}\nCI={ci:.4f}\n")
    np.savez(os.path.join(HERE,"results",f"preds_{args.split}.npz"), y=ys, pred=preds)

if __name__ == "__main__":
    main()
