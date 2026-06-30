# Kinase Inhibitor Binding Predictor (Deep Learning)

A deep-learning model that predicts kinase–inhibitor binding affinity from protein sequence and molecule structure, trained on the KIBA benchmark. The emphasis is on **honest, leakage-free evaluation** and on connecting the model back to a physics-based docking pipeline (see companion project [gpu-virtual-screening](https://github.com/pitamberdasla108-ui/gpu-virtual-screening)).

---

## What this project does

Given a **protein kinase** (amino-acid sequence) and a **small-molecule inhibitor** (SMILES), the model predicts how strongly they bind (the KIBA score). It is trained end-to-end on real bioactivity data, evaluated with splits designed to *avoid data leakage*, and cross-checked against an independent docking method on out-of-dataset drugs.

The point is not just "a model that scores well" — it's demonstrating the difference between an inflated tutorial number and an honest measure of real-world generalization.

---

## Data

- **KIBA benchmark** — 117,657 measured kinase–inhibitor interactions across **229 kinases** and **2,068 inhibitors** (Tang et al. 2014; loaded via Therapeutics Data Commons).
- KIBA scores are continuous (regression target), and notably **compressed**: median ≈ 11.5, most values in a narrow 11–12 band, with rare strong binders up to ~17. This distribution shape matters for evaluation and is discussed below.

## Features

- **Proteins → ESM-2 embeddings.** Each kinase sequence is passed through the **ESM-2 protein language model** (150M parameters, `esm2_t30_150M_UR50D`) and mean-pooled into a 640-dim vector. Computed once per unique kinase and cached.
- **Drugs → Morgan fingerprints.** Each inhibitor SMILES becomes a 2048-bit ECFP4 fingerprint (RDKit). Cached per unique drug.
- Final feature per pair: 640 + 2048 = **2688 dimensions**.

## Model

A multilayer perceptron (PyTorch), trained on an RTX 3090 (CUDA):
---

## Honest evaluation (the core of the project)

The naive approach is a random train/test split — but that **leaks**: the same kinases and drugs appear in both sets, so the model can recognize familiar entities instead of generalizing. This project reports three splits:

| Split | What it tests | RMSE | Pearson r | CI |
|---|---|---|---|---|
| **Random** | familiar kinases & drugs (leaks; upper bound) | 0.49 | 0.81 | **0.82** |
| **Cold-drug** | **unseen molecules** (0 drug overlap) | 0.65 | 0.64 | 0.73 |
| **Cold-target** | **unseen kinases** (0 kinase overlap) | 0.64 | 0.69 | 0.73 |

*(CI = concordance index; 0.5 = random ordering, 1.0 = perfect ranking.)*

**The finding:** performance is strong on the random split (CI 0.82, competitive with published DTA baselines), but drops to ~0.73 on both cold splits. That ~0.09 gap quantifies how much of the random-split score came from recognizing familiar entities versus true generalization. **Reporting this gap honestly — rather than only the inflated random-split number — is the central goal.**

![Predicted vs true KIBA score across splits](results/scatter_all_splits.png)

The model is also visibly best in KIBA's data-rich middle band and less certain on the rare high-affinity binders — a direct consequence of the compressed score distribution.

---

## Cross-validation against docking (out-of-dataset)

The four front-line CML drugs (imatinib, nilotinib, dasatinib, ponatinib) are **not** in KIBA — so predicting their binding to c-Abl is a true out-of-dataset test. Their docking scores come from the companion [GPU virtual screening project](https://github.com/pitamberdasla108-ui/gpu-virtual-screening).

![ML vs docking on CML drugs](results/cml_ml_vs_docking.png)

- The ML model — **which never saw these drugs** — predicts all four as strong c-Abl binders (all well above the KIBA median), and agrees with docking on the top-ranked compound (**ponatinib**).
- The two methods diverge on finer ordering (Spearman ρ = 0.40). With only four molecules and KIBA's compressed range, this is an **illustrative cross-check, not a statistical claim** — but agreement on out-of-dataset binders between two independent methods (physics-based docking and data-driven ML) is a meaningful sanity check.

---

## Protein structure prediction (ESMFold)

To round out the structural picture, the **c-Abl kinase domain** (~274 residues) was folded *de novo* with **ESMFold** on the same RTX 3090:

- Folded in ~9 seconds; **mean pLDDT ≈ 91** (high confidence — AlphaFold's "very high" tier).
- Output is a real atomic structure (2,213 atoms) recovering the canonical bilobal kinase fold, with the ATP-binding cleft where the CML drugs dock.

This demonstrates the structure-prediction step of a modern pipeline running on consumer hardware. (c-Abl has known crystal structures; this is a capability/validation demonstration, honestly framed.)

---

## Virtual screen + docking cross-validation

As a capstone, the trained model screened **1,039 KIBA kinase inhibitors never measured against c-Abl** — a genuine prediction on untested pairs. The top predictions land above the 90th percentile of *measured* c-Abl binders but within the realistic ceiling (a plausible, non-degenerate result), prioritizing ~67 compounds as computational candidates.

The **top-5 ML hits were then independently docked** in the companion docking pipeline:

![Virtual screen cross-validation](results/screen_crossval.png)

**The honest finding:** the two methods *diverge* on the top pick. Docking strongly favors the single large, drug-like inhibitor (CHEMBL1684800, −23.68 kcal/mol) — consistent with docking's known molecular-size bias (documented in the companion project). The ML model, trained on measured affinities, instead ranks smaller polyphenol scaffolds highest. The compound both methods endorse (CHEMBL1684800) is the most defensible joint candidate. This divergence illustrates how physics-based and data-driven methods carry *different biases* — which is exactly why using both is more informative than either alone. These remain computational hypotheses for prioritization, not validated binders.

---

## Repository contents
## How to run

```bash
python load_data.py        # fetch KIBA
python featurize.py        # ESM-2 + fingerprints (uses GPU)
python make_splits.py      # build the three splits
python train.py --split random
python train.py --split cold_drug
python train.py --split cold_target
python predict_cml.py      # docking cross-check
```

## Honest scope & limitations

- Predicts on a fixed benchmark (KIBA); not a substitute for experimental assays.
- The compressed KIBA distribution limits fine-grained ranking of strong binders.
- The docking cross-check uses 4 molecules — illustrative, not statistically powered.
- Foundation models (ESM-2) are used pretrained; the trained component is the MLP head on top of these features.

## References

- Tang et al., *Making sense of large-scale kinase inhibitor bioactivity sets*, J. Chem. Inf. Model. 2014 (KIBA).
- Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* (ESM-2), Science 2023.
- Öztürk et al., *DeepDTA*, Bioinformatics 2018 (benchmark splits).
- Huang et al., *Therapeutics Data Commons*, 2021.
