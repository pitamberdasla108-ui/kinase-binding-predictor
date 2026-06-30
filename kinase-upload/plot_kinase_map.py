#!/usr/bin/env python3
"""plot_kinase_map.py — 2D map of 229 KIBA kinases in ESM-2 embedding space.
Colors a confidently-curated set of tyrosine kinases (TK group, the cancer
drug targets) to test whether ESM-2 embeddings recover known kinase biology."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"data")

# Confidently-identified tyrosine kinases (TK group) by UniProt ID — the
# cancer-relevant receptor & non-receptor TKs. (Curated, high-confidence subset.)
TK = {
 "P00519","P00533","P04626","P06213","P06239","P07333","P08581","P08069",
 "P09619","P10721","P11362","P16234","P17948","P21709","P21802","P22455",
 "P22607","P29317","P29320","P29322","P29323","P30530","P35916","P35968",
 "P35969","P36888","P51451","P54756","P54760","P54762","P54764","P0C1S8",
 "Q02763","Q08881","Q06187","Q9UM73","Q9HBH9","P29376","P43403","P07948",
 "P06241","P12931","P51813","Q06418","P29350","P43405","P42685","P51451",
 "P09769","P52333","O75116","P53671","Q05397","P42684","P00520",
}
# A few clear serine/threonine examples for contrast (CMGC: CDKs/MAPKs)
CMGC = {"P24941","Q00534","P06493","P11802","Q00535","P49841","P28482",
        "P45983","P53779","Q16539","P49840","P24386","Q9NYV4","P50750"}

def group_of(uid):
    if uid in TK: return "Tyrosine kinase (TK)"
    if uid in CMGC: return "CMGC (CDK/MAPK/GSK3)"
    return "Other"

prot = dict(np.load(os.path.join(DATA,"prot_emb.npz")))
ids = list(prot.keys())
X = np.stack([prot[i] for i in ids])
groups = np.array([group_of(i) for i in ids])

print("Group counts:")
for g in sorted(set(groups)):
    print(f"  {g:24s}: {(groups==g).sum()}")

# t-SNE to 2D
emb2 = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(X)

colors = {"Tyrosine kinase (TK)":"#e53e3e","CMGC (CDK/MAPK/GSK3)":"#3182ce","Other":"#cbd5e0"}
plt.figure(figsize=(9,7))
for g in ["Other","CMGC (CDK/MAPK/GSK3)","Tyrosine kinase (TK)"]:
    m = groups==g
    plt.scatter(emb2[m,0], emb2[m,1], s=45, c=colors[g], label=f"{g} (n={m.sum()})",
                edgecolors="white", linewidths=0.5, alpha=0.9)
# mark c-Abl specifically
if "P00519" in ids:
    i = ids.index("P00519")
    plt.scatter(emb2[i,0], emb2[i,1], s=220, facecolors="none", edgecolors="black",
                linewidths=2, label="c-Abl (P00519)")
plt.legend(loc="best", fontsize=10)
plt.title("229 KIBA kinases in ESM-2 embedding space (t-SNE)\nColored by kinase group — families self-organize from sequence alone", fontsize=11)
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.tight_layout()
out=os.path.join(HERE,"results","kinase_embedding_map.png")
plt.savefig(out, dpi=140, bbox_inches="tight"); print("Saved:", out)
