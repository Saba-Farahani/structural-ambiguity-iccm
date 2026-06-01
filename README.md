# Structural Ambiguity in Wearable Physiological AI
### A Coupling Divergence Framework for Detection and Safe Abstention 
University of California, Irvine  

---

## Overview

Wearable stress classifiers routinely achieve high mean accuracy while concealing severe failures for specific individuals. This repository provides the implementation of **Protocol-Aware ICCM** (Individual Conformal Coupling Monitor), a pre-inference physiological safety agent that detects *structural ambiguity* — windows where signals are clean but their inter-signal coupling structure is inconsistent with the assumptions needed for reliable classification.

**Key results:**
- Structural ambiguity scores significantly predict individual classifier failure on WESAD (*r* = −0.607, *p* = 0.016) and Stress-Predict (*r* = −0.416, *p* = 0.013)
- False stress alerts reduced by **6.9%** (WESAD) and **12.6%** (Stress-Predict)
- No subject experiences more than **2% accuracy degradation** from gating
- Outperforms random abstention at matched coverage on both datasets

---

## Method

Protocol-Aware ICCM operates as a **pre-inference structural validity monitor** positioned before the classifier. It requires no model retraining and is independent of the classifier architecture.

### Hybrid Coupling Vector
For each 60-second window, ICCM computes a protocol-aware coupling vector:

```
v(t) = [ρ_EH, ρ_ET, ρ_HT,          ← Pearson correlation
         ℓ_EH, ℓ_ET, ℓ_HT,          ← Max-lag cross-correlation [1–10s]
         g_E→H, g_H→E, g_T→H]        ← Granger-style directed coupling
```

- **Single-protocol datasets** (WESAD): 6 features (ρ + ℓ), Frobenius distance
- **Multi-protocol datasets** (Stress-Predict): 9 features (ρ + ℓ + g), direction-aware score

### 3-Zone Safety Gate
| Zone | Condition | Action |
|------|-----------|--------|
| Zone 1 | p(t) ≥ α | Classify — structurally supported |
| Zone 2 | α/2 ≤ p(t) < α | Defer — borderline structure |
| Zone 3 | p(t) < α/2 | Abstain — structurally unsupported |

---

## OpenCHA Interface

Protocol-Aware ICCM is designed to operate within the **[OpenCHA](https://github.com/Institute4FutureHealth/CHA)** framework developed at the Institute for Future Health, UCI. OpenCHA provides the conversational health agent interface through which users can query their stress status. ICCM acts as a deterministic pre-inference safety layer within OpenCHA's orchestration pipeline:

```
User Query ("Am I stressed?")
        ↓
   OpenCHA Interface
        ↓
   Protocol-Aware ICCM  ←── Individual coupling baseline
        ↓
   3-Zone Safety Gate
        ↓
 Classify / Defer / Abstain
        ↓
   OpenCHA Response
```

For OpenCHA integration, ICCM plugs into the Orchestrator component as a deterministic physiological health agent operating before the downstream classifier.

---

## Installation

```bash
git clone https://github.com/saba-farahani/structural-ambiguity-iccm.git
cd structural-ambiguity-iccm
pip install -r requirements.txt
```

**Requirements:**
```
numpy
pandas
scipy
scikit-learn
statsmodels
matplotlib
```

---

## Datasets

### WESAD
Download from: https://archive.ics.uci.edu/dataset/465/wesad

Place data at:
```
data/WESAD/S2/S2.pkl
data/WESAD/S3/S3.pkl
...
```

### Stress-Predict
Download from: https://www.kaggle.com/datasets/qiriro/stress

Place data at:
```
data/Stress-Predict-Dataset/Raw_data/S01/EDA.csv
data/Stress-Predict-Dataset/Raw_data/S01/BVP.csv
...
```

---

## Running Experiments

**Full run (ablation + main method):**
```bash
python src/bsn_paper.py \
    --wesad_path data/WESAD \
    --stress_path data/Stress-Predict-Dataset/Raw_data
```

**Main method only (fast):**
```bash
python src/bsn_paper.py \
    --wesad_path data/WESAD \
    --stress_path data/Stress-Predict-Dataset/Raw_data \
    --skip_ablation
```

**Output:**
- `results/wesad_protocol_aware_iccm.csv` — WESAD per-subject results
- `results/sp_protocol_aware_iccm.csv` — Stress-Predict per-subject results
- `figures/fig3_scatter_v3.pdf` — Ambiguity score vs. LOSO accuracy scatter plots

---

## Results

| | WESAD (N=15) | Stress-Predict (N=35) |
|---|---|---|
| Coupling vector | corr + lag (6) | corr + lag + directed (9) |
| ICCM metric | Frobenius | Direction-aware |
| Ambiguity *r* | −0.607 * | −0.416 * |
| FP (population) | 29 | 95 |
| FP (ICCM gated) | **27** | **83** |
| FP reduction | 6.9% | 12.6% |
| >2% accuracy drop | 0 | 0 |

*\* p < 0.05*

---

## Repository Structure

```
structural-ambiguity-iccm/
│
├── src/
│   └── bsn_paper_v2.py       # Main pipeline: coupling vectors, ICCM, evaluation
│
├── figures/
│   ├── Fig1.jpeg              # Motivation: aggregate failure + S14 decoupling
│   ├── Fig2.jpeg              # Architecture: Protocol-Aware ICCM system
│   └── fig3_scatter_v3.png   # Results: ambiguity score vs. LOSO accuracy
│
├── results/                   # Generated CSV outputs (created on run)
├── data/                      # Datasets (not included, see above)
├── requirements.txt
└── README.md
```
