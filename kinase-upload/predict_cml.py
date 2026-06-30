#!/usr/bin/env python3
"""predict_cml.py — Run the trained model on the 4 CML drugs vs c-Abl (P00519),
a TRUE out-of-dataset prediction (these drugs are NOT in KIBA), and compare to
the docking ranking from the GPU virtual screening project."""
import os, numpy as np, torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CABL = "P00519"

CML_DRUGS = {
  "imatinib":  "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
  "nilotinib": "Cc1ccc(cc1Nc1nccc(n1)-c1cccnc1)C(=O)Nc1cc(cc(c1)C(F)(F)F)n1ccnc1C",
  "dasatinib": "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
  "ponatinib": "Cc1ccc(C(=O)Nc2ccc(CN3CCN(C)CC3)c(C(F)(F)F)c2)cc1C#Cc1cnc2cccnn12",
}
DOCKING = {"ponatinib": -20.88, "nilotinib": -19.71, "dasatinib": -18.96, "imatinib": -17.94}

class MLP(nn.Module):
    def __init__(self, in_dim=2688, hidden=(1024,512,128), p=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]; d = h
        layers += [nn.Linear(d,1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def fp_of(smi):
    m = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.float32); DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    prot = dict(np.load(os.path.join(DATA, "prot_emb.npz")))
    if CABL not in prot:
        print(f"ERROR: {CABL} not in cached embeddings."); return
    pemb = prot[CABL]

    model = MLP().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(HERE,"results","model_random.pt"), map_location=DEVICE))
    model.eval()

    # batch all 4 together so BatchNorm sees >1 sample
    names = list(CML_DRUGS.keys())
    X = np.stack([np.concatenate([pemb, fp_of(CML_DRUGS[n])]) for n in names])
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(DEVICE)).cpu().numpy()
    ml = {n: float(p) for n, p in zip(names, preds)}

    print("=== c-Abl: ML prediction vs docking (out-of-dataset drugs) ===\n")
    print(f"{'drug':10s} {'ML pred (KIBA)':>15s} {'docking (kcal/mol)':>20s}")
    for n in sorted(ml, key=lambda k: -ml[k]):
        print(f"{n:10s} {ml[n]:15.2f} {DOCKING[n]:20.2f}")

    # rankings
    ml_rank = [n for n in sorted(ml, key=lambda k: -ml[k])]          # higher KIBA = stronger
    dock_rank = [n for n in sorted(DOCKING, key=lambda k: DOCKING[k])]  # more negative = stronger
    print("\nML ranking (strongest->weakest):    ", " > ".join(ml_rank))
    print("Docking ranking (strongest->weakest):", " > ".join(dock_rank))

    # Spearman between the two rankings
    from scipy.stats import spearmanr
    ml_scores = [ml[n] for n in names]
    dock_scores = [-DOCKING[n] for n in names]  # flip sign so higher=stronger for both
    rho, p = spearmanr(ml_scores, dock_scores)
    print(f"\nSpearman rank correlation (ML vs docking): rho = {rho:.3f}")

if __name__ == "__main__":
    main()
