#!/usr/bin/env python3
"""plot_cml.py — bar comparison of ML prediction vs docking for the 4 CML drugs."""
import os, numpy as np, torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
CML={"imatinib":"Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
"nilotinib":"Cc1ccc(cc1Nc1nccc(n1)-c1cccnc1)C(=O)Nc1cc(cc(c1)C(F)(F)F)n1ccnc1C",
"dasatinib":"Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
"ponatinib":"Cc1ccc(C(=O)Nc2ccc(CN3CCN(C)CC3)c(C(F)(F)F)c2)cc1C#Cc1cnc2cccnn12"}
DOCK={"ponatinib":-20.88,"nilotinib":-19.71,"dasatinib":-18.96,"imatinib":-17.94}

class MLP(nn.Module):
    def __init__(s,in_dim=2688,hidden=(1024,512,128),p=0.3):
        super().__init__(); L=[]; d=in_dim
        for h in hidden: L+=[nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(p)]; d=h
        L+=[nn.Linear(d,1)]; s.net=nn.Sequential(*L)
    def forward(s,x): return s.net(x).squeeze(-1)
def fp(smi):
    m=Chem.MolFromSmiles(smi); a=np.zeros((2048,),dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048),a); return a

prot=dict(np.load(os.path.join(DATA,"prot_emb.npz"))); pemb=prot["P00519"]
model=MLP().to(DEVICE); model.load_state_dict(torch.load(os.path.join(HERE,"results","model_random.pt"),map_location=DEVICE)); model.eval()
names=list(CML); X=np.stack([np.concatenate([pemb,fp(CML[n])]) for n in names])
with torch.no_grad(): ml=model(torch.from_numpy(X).float().to(DEVICE)).cpu().numpy()
mld={n:float(v) for n,v in zip(names,ml)}

order=sorted(names,key=lambda n:-mld[n])
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
a1.bar(order,[mld[n] for n in order],color="#2b6cb0"); a1.axhline(11.5,ls="--",c="gray",lw=1)
a1.set_title("ML model prediction (never saw these drugs)"); a1.set_ylabel("Predicted KIBA score")
a1.set_ylim(11,14.5); a1.text(0,11.55,"KIBA median ~11.5",fontsize=8,color="gray")
a2.bar(order,[DOCK[n] for n in order],color="#c05621")
a2.set_title("Docking score (physics-based)"); a2.set_ylabel("Docking energy (kcal/mol, lower=stronger)")
for ax in (a1,a2): ax.tick_params(axis='x',rotation=20)
plt.suptitle("CML drugs vs c-Abl: ML model vs docking (out-of-dataset cross-check)",y=1.02,fontsize=12)
plt.tight_layout(); out=os.path.join(HERE,"results","cml_ml_vs_docking.png")
plt.savefig(out,dpi=130,bbox_inches="tight"); print("Saved:",out)
