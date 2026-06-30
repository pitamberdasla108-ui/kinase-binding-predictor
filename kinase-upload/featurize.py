#!/usr/bin/env python3
"""featurize.py — ESM-2 (150M) protein embeddings + Morgan fingerprints, cached."""
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
MAX_LEN, FP_BITS, FP_RADIUS = 1022, 2048, 2

def embed_proteins(uniq):
    print(f"Loading ESM-2 on {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = AutoModel.from_pretrained(ESM_MODEL).to(DEVICE).eval()
    out = {}
    for _, row in tqdm(uniq.iterrows(), total=len(uniq), desc="Embedding kinases"):
        enc = tok(row["Target"][:MAX_LEN], return_tensors="pt", truncation=True, max_length=MAX_LEN)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            rep = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        emb = ((rep*mask).sum(1)/mask.sum(1).clamp(min=1)).squeeze(0).cpu().numpy().astype(np.float32)
        out[row["Target_ID"]] = emb
    return out

def fingerprint_drugs(uniq):
    out, n_fail = {}, 0
    for _, row in tqdm(uniq.iterrows(), total=len(uniq), desc="Fingerprinting drugs"):
        mol = Chem.MolFromSmiles(row["Drug"])
        if mol is None:
            n_fail += 1; continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
        arr = np.zeros((FP_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[row["Drug_ID"]] = arr
    if n_fail: print(f"  WARNING: {n_fail} SMILES failed to parse.")
    return out

def main():
    df = pd.read_csv(os.path.join(DATA, "kiba_raw.csv"))
    uniq_t = df[["Target_ID","Target"]].drop_duplicates("Target_ID").reset_index(drop=True)
    uniq_d = df[["Drug_ID","Drug"]].drop_duplicates("Drug_ID").reset_index(drop=True)
    print(f"Unique kinases: {len(uniq_t)} | unique drugs: {len(uniq_d)}")

    prot = embed_proteins(uniq_t)
    np.savez_compressed(os.path.join(DATA,"prot_emb.npz"), **prot)
    print("Saved prot_emb.npz:", len(prot), "kinases, dim =", next(iter(prot.values())).shape[0])

    drug = fingerprint_drugs(uniq_d)
    np.savez_compressed(os.path.join(DATA,"drug_fp.npz"), **drug)
    print("Saved drug_fp.npz:", len(drug), "drugs, bits =", next(iter(drug.values())).shape[0])

if __name__ == "__main__":
    main()
