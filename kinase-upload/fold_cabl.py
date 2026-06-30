#!/usr/bin/env python3
"""fold_cabl.py — Predict the 3D structure of the c-Abl kinase domain with ESMFold."""
import os, torch, numpy as np
from transformers import AutoTokenizer, EsmForProteinFolding

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE,"results"), exist_ok=True)

# c-Abl1 (P00519) kinase domain — catalytic core (UniProt ~242-495 region)
CABL_KINASE_DOMAIN = (
    "ITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFM"
    "TYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAH"
    "AGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRAC"
    "WQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGV"
)

def main():
    print(f"Folding c-Abl kinase domain ({len(CABL_KINASE_DOMAIN)} residues) on GPU...")
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    model.trunk.set_chunk_size(64)  # memory-efficient chunking

    enc = tok([CABL_KINASE_DOMAIN], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    import time; t0 = time.time()
    with torch.no_grad():
        out = model(enc)
    dt = time.time() - t0
    print(f"Folded in {dt:.1f}s")

    # pLDDT confidence (0-100)
    plddt = out["plddt"][0, :, 1].cpu().numpy()  # CA atom pLDDT
    print(f"Mean pLDDT (confidence): {plddt.mean():.1f}")
    print(f"  high-confidence residues (pLDDT>70): {(plddt>70).sum()}/{len(plddt)}")

    # save PDB
    pdb = model.output_to_pdb(out)[0]
    pdb_path = os.path.join(HERE, "results", "cabl_kinase_domain_esmfold.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb)
    print("Saved structure:", pdb_path)
    np.save(os.path.join(HERE,"results","cabl_plddt.npy"), plddt)

if __name__ == "__main__":
    main()
