# Geometry-Aware Boundary Half-Hop: 3D Point Cloud Part Segmentation

This repository contains the official code, experiments, and benchmark results for **Geometry-Aware Boundary Half-Hop**, an extension of the Half-Hop framework ([Azabou et al., ICML 2023](https://proceedings.mlr.press/v202/azabou23a.html)) to 3D geometric point cloud manifolds.

For the exhaustive mathematical formulation, geometric theory, and empirical findings, see **[EXTENSION.md](EXTENSION.md)**.

---

## 1. Overview & Method

While standard Half-Hop slows down message passing across abstract relational graphs (e.g., citation and webpage networks), applying uniform edge subdivision to physical 3D manifolds induces **receptive-field starvation** across smooth continuous surfaces.

**Geometry-Aware Boundary Half-Hop** resolves this by introducing a **dual-gated geometric boundary detector**:
1. **Pauly Surface Variation ($\sigma_n$):** Measures local curvature and planar deviation from the local covariance matrix.
2. **Surface Normal Disparity ($\Delta \mathbf{n}_{ij}$):** Detects sharp dihedral angles and structural part joints.

By selectively inserting virtual slow nodes ($\nu_k$) strictly along physical seams and part joints (throttling $\le 5\%$ of edges), our method eliminates cross-part feature bleeding while preserving fast 1-hop message passing across smooth surfaces.

---

## 2. Key Empirical Findings (ShapeNet-Part)

- **Table:** Boosts overall mIoU from $74.32\%$ to **$76.19\%$** (**+1.87%**) and boundary fidelity from $54.63\%$ to **$56.73\%$** (**+2.10%** $\text{mIoU}_{\text{bnd}}$).
- **Airplane:** Increases overall mIoU to **$78.23\%$** (**+0.28%**) and boundary fidelity from $57.41\%$ to **$58.44\%$** (**+1.03%** $\text{mIoU}_{\text{bnd}}$).
- **Chair Anti-Starvation Sweep:** Pruning slow nodes to the top $2\%$ sharpest geometric seams ($Q=0.98$) achieves peak accuracy (**$85.28\%$** mIoU, **+0.52%** over unconstrained gating).

---

## 3. Repository Structure

```
halfhop-project/
├── halfhop/                         # Core Transformation Engine
│   ├── __init__.py                  # Clean extension exports
│   ├── boundary_halfhop.py          # BoundaryHalfHop class (dual-gated boundary slowing)
│   ├── geometry_boundary.py         # Curvature, normal disparity & crease detection
│   ├── halfhop.py                   # Base Half-Hop graph transform and slow node insertion
│   ├── selective_halfhop.py         # Selective edge subset rewiring & topological primitives
│   ├── reproducibility.py          # PRNG seeding across Torch, NumPy, and Python
│   └── utils.py                     # Tensor helpers and utilities
│
├── experiments/
│   ├── shapenet_partseg/            # ShapeNet-Part Execution Pipeline
│   │   ├── model.py                 # Multi-scale EdgeConv part segmentation architecture
│   │   ├── run_partseg.py           # Main training & evaluation driver (none, boundary, random)
│   │   ├── run_rigid_benchmark.sh   # Multi-category benchmark runner (Table, Airplane, Chair)
│   │   ├── sweep_thresholds_kaggle.sh # Quantile threshold ablation runner
│   │   ├── run_kaggle.sh            # Single GPU Kaggle execution script
│   │   └── run_kaggle_2gpu.sh       # Distributed multi-GPU execution script
│   └── results/                     # Empirical Benchmark Results
│       ├── partseg.jsonl            # ShapeNet-Part experimental runs
│       ├── rigid_partseg_benchmark.jsonl # Rigid object category evaluation runs
│       └── threshold_sweep_chair.jsonl   # Chair quantile threshold sweep logs
│
├── tests/                           # Unit and Integration Test Suite
│   ├── test_boundary_halfhop.py     # Geometry boundary score, curvature, and normal tests
│   ├── test_halfhop.py              # Base topological invariants and slow node tests
│   └── test_selective_halfhop.py    # Edge rewiring, replacement, and mask tests
│
├── EXTENSION.md                     # Comprehensive Technical Specification and Paper
├── pyproject.toml                   # Build configuration and package metadata
├── requirements.txt                 # Project dependencies
└── pytest.ini                       # PyTest configuration
```

---

## 4. Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install package in editable mode
pip install -e .
```

---

## 5. Running Tests

Run the complete test suite to verify topological invariants, edge rewiring, and differential geometry operators:

```bash
pytest
```

---

## 6. Running Experiments

### Train on ShapeNet-Part

```bash
# Baseline EdgeConv (no Half-Hop)
python -m experiments.shapenet_partseg.run_partseg --category Table --condition none --epochs 100

# Geometry-Aware Boundary Half-Hop (proposed)
python -m experiments.shapenet_partseg.run_partseg --category Table --condition boundary --epochs 100

# Random Half-Hop Control (matched edge budget)
python -m experiments.shapenet_partseg.run_partseg --category Table --condition random --epochs 100
```

### Threshold Sensitivity Sweep

```bash
python -m experiments.shapenet_partseg.run_partseg --category Chair --sweep
```

### Batch Rigid Category Benchmark

```bash
bash experiments/shapenet_partseg/run_rigid_benchmark.sh
```
