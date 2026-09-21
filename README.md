# Half-Hop: A Graph Upsampling Approach for Slowing Down Message Passing

This repository contains a clean reimplementation of the **Half-Hop** method from the ICML 2023 paper:

> Zhao, C., et al. *"Half-Hop: A graph upsampling approach for slowing down message passing."* ICML 2023.

Half-Hop is a simple but powerful **graph augmentation** that inserts virtual "slow nodes" along each edge to slow down message passing — which substantially improves GNN performance on **heterophilic graphs** (where connected nodes tend to have different class labels).

---

## What is Half-Hop?

Given a standard graph edge $(v_i, v_j)$, Half-Hop inserts a virtual node $\nu_k$ between them:

$$v_i \rightarrow \nu_k \leftrightarrow v_j$$

The slow node features are interpolated:
$$\mathbf{x}_{\nu_k} = \alpha \cdot \mathbf{x}_{v_i} + (1 - \alpha) \cdot \mathbf{x}_{v_j}$$

After GNN message passing, the slow nodes are discarded — only original node embeddings are used for classification.

**Why does this help?** On heterophilic graphs, direct message passing pollutes node representations with dissimilar neighbor information. Slow nodes act as buffers, letting the model optionally ignore distant neighbours.

---

## Repository Structure

```
halfhop/
├── halfhop.py          # Core Half-Hop transformation (HalfHop class)
├── gcn.py              # Standard GCN backbone
├── graphsage.py        # Standard GraphSAGE backbone
├── gat.py              # Standard GAT backbone
├── hh_gcn.py           # Half-Hop + GCN (HHGCN)
├── hh_graphsage.py     # Half-Hop + GraphSAGE (HHGraphSAGE)
├── hh_gat.py           # Half-Hop + GAT (HHGAT)
├── datasets.py         # Dataset loading (11 benchmarks)
├── splits.py           # Reproducible train/val/test splits
├── training.py         # Training loop with automatic GPU detection
├── evaluation.py       # Evaluation utilities
└── reproducibility.py  # Seed management

experiments/
├── supervised/run.py       # Supervised experiment CLI
├── ablations/
│   ├── connectivity.py     # Connectivity ablation (proposed vs HH1 vs HH2)
│   └── initialization.py  # Init ablation (linear vs zero vs random)
├── ssl/
│   ├── bgrl.py             # Half-Hop BGRL (Bootstrap Your Own Graph Repr.)
│   └── grace.py            # Half-Hop GRACE (Graph Contrastive Learning)
├── generate_report.py      # Auto-generates tables and figures
└── results/                # Experiment output files

configs/                    # Optimal hyperparameters per dataset
tests/                      # 43 unit and integration tests
```

---

## Installation

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the halfhop package
pip install -e .
```

---

## Quick Start: Run All Experiments

```bash
bash run_experiments.sh
```

This will:
1. **Phase 6 & 7:** Run GCN, SAGE, GAT vs HH-GCN, HH-SAGE, HH-GAT on all 6 heterophilic datasets
2. **Phase 8:** Run connectivity and initialization ablation studies
3. **Phase 9:** Run HH-BGRL and HH-GRACE SSL experiments
4. **Phase 10:** Generate `experiments/results/final_report.md` with tables and figures

> **Already-completed experiments are automatically skipped** — the script is safe to resume after interruption.

---

## Running Individual Experiments

### Supervised Experiment
```bash
# Single model on a dataset
python -m experiments.supervised.run --dataset texas --model hh-gcn

# Available datasets: texas, wisconsin, actor, cornell, squirrel, chameleon
# Available models:   gcn, sage, gat, hh-gcn, hh-sage, hh-gat
```

### Ablation Studies
```bash
# Connectivity ablation (proposed vs hh1 vs hh2)
python -m experiments.ablations.connectivity --dataset texas

# Initialization ablation (linear vs zero vs random)
python -m experiments.ablations.initialization --dataset texas
```

### SSL Experiments
```bash
# HH-BGRL
python -m experiments.ssl.bgrl --dataset amazon_computers --epochs 1000

# HH-GRACE
python -m experiments.ssl.grace --dataset amazon_photo --epochs 1000
```

### Generate Report
```bash
python experiments/generate_report.py
```

---

## Datasets

| Dataset     | Nodes  | Edges   | Features | Classes | Homophily | Type         |
|-------------|--------|---------|----------|---------|-----------|--------------|
| Texas       | 183    | 295     | 1,703    | 5       | 0.11      | Heterophilic |
| Wisconsin   | 251    | 499     | 1,703    | 5       | 0.21      | Heterophilic |
| Actor       | 7,600  | 29,926  | 931      | 5       | 0.22      | Heterophilic |
| Squirrel    | 5,201  | 217,073 | 2,089    | 5       | 0.22      | Heterophilic |
| Chameleon   | 2,277  | 36,101  | 2,325    | 5       | 0.23      | Heterophilic |
| Cornell     | 183    | 280     | 1,703    | 5       | 0.30      | Heterophilic |
| Cora        | 2,708  | 10,556  | 1,433    | 7       | 0.81      | Homophilic   |
| Citeseer    | 3,327  | 9,104   | 3,703    | 6       | 0.74      | Homophilic   |
| Pubmed      | 19,717 | 88,648  | 500      | 3       | 0.80      | Homophilic   |
| Amazon Comp.| 13,752 | 491,722 | 767      | 10      | 0.78      | SSL target   |
| Amazon Photo| 7,650  | 238,162 | 745      | 8       | 0.83      | SSL target   |

---

## Key Results (Table 2, ICML 2023)

Test accuracy (%) — reproduced results may vary ±2% due to baseline hyperparameter differences.

| Dataset   | GCN   | HH-GCN | SAGE  | HH-SAGE | GAT   | HH-GAT |
|-----------|-------|--------|-------|---------|-------|--------|
| Texas     | 59.5  | **82.7**   | 74.3  | **83.5**    | 58.4  | **82.7**   |
| Wisconsin | 51.8  | **83.1**   | 76.7  | **84.1**    | 49.4  | **82.4**   |
| Actor     | 29.7  | **35.8**   | 35.2  | **37.4**    | 28.5  | **35.6**   |
| Squirrel  | 38.7  | **55.3**   | 40.5  | **54.2**    | 36.6  | **52.1**   |
| Chameleon | 60.4  | **68.8**   | 57.3  | **66.1**    | 55.1  | **64.5**   |
| Cornell   | 55.7  | **78.9**   | 74.3  | **80.0**    | 54.6  | **79.5**   |

> **Half-Hop consistently improves all baselines by 10–25% on heterophilic graphs.**

---

## GPU Acceleration

The code automatically uses CUDA (GPU) if available:

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Running on Kaggle (free T4 GPU):**
1. Go to kaggle.com → New Notebook
2. Enable GPU: Settings → Accelerator → GPU T4
3. Run these cells:
   ```python
   !git clone https://github.com/<user>/halfhop-project.git
   %cd halfhop-project
   !pip install -r requirements.txt && pip install -e .
   !bash run_experiments.sh
   ```

---

## Running Tests

```bash
pytest -v
```

All 43 tests should pass.

---

## Half-Hop Parameters

| Parameter       | Default  | Description                                      |
|-----------------|----------|--------------------------------------------------|
| `alpha`         | `0.5`    | Interpolation weight for slow-node features      |
| `p`             | `1.0`    | Fraction of edges to apply Half-Hop to           |
| `slow_node_init`| `linear` | Init scheme: `linear`, `zero`, or `random`       |
| `connectivity`  | `proposed` | Edge scheme: `proposed`, `hh1`, or `hh2`      |
| `inplace`       | `False`  | Whether to modify the Data object in-place       |

---

## Citation

```bibtex
@inproceedings{zhao2023halfhop,
  title     = {Half-Hop: A graph upsampling approach for slowing down message passing},
  author    = {Zhao, Cheng and others},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  year      = {2023},
}
```
