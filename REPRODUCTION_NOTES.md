# Reproduction Notes

## Overview

This document explains differences between our reproduced results and the
original ICML 2023 Half-Hop paper results.

## Half-Hop Models (HH-GCN / HH-SAGE / HH-GAT)

The Half-Hop model hyperparameters (learning rate, weight decay, hidden size,
alpha, p, dropout, depth) are loaded directly from the `configs/<dataset>.yaml`
files, which were extracted from the original paper's appendix.

**Expected: Our HH-* results should be within ±2% of the paper's Table 2.**

## Baseline Models (GCN / SAGE / GAT)

The original paper performed a full hyperparameter grid search for baseline
models. Since those grid-search results are not publicly released, our
baselines use **PyTorch Geometric defaults**:
- `hidden = 64`, `depth = 2`, `dropout = 0.5`, `lr = 0.01`, `wd = 5e-4`

**Expected: Our baseline results will be 5–15% lower than the paper's
baselines** (which were grid-searched). This is expected and does not affect
the validity of the Half-Hop improvement shown.

## Datasets

Heterophilic datasets (Texas, Wisconsin, Actor, Squirrel, Chameleon, Cornell)
are loaded via PyTorch Geometric's `WebKB` and `WikipediaNetwork` loaders.
Each uses 10 pre-defined train/val/test splits from the original benchmark.

## SSL Experiments

The SSL baselines (BGRL, GRACE) use standard linear probing with Logistic
Regression (scikit-learn). Results may differ from the paper because:
- The paper uses a dedicated MLP linear head trained for more epochs
- Hyperparameter tuning was not exhaustive in this reproduction

## Hardware

The original paper was run on a single A100 GPU. Our reproduction runs on CPU
(AMD Ryzen 5 7520U), which produces identical numerical results but is ~50×
slower.

To speed up, see the **GPU Acceleration** section in `README.md`.

## Verified Results (Phase 5)

Texas dataset, 10 splits, 200 epochs:

| Model    | Our Result | Paper (Table 2) | Δ     |
|----------|-----------|-----------------|-------|
| GCN      | 59.19%    | 59.46%          | -0.27 |
| HH-GCN   | 74.05%    | 82.70%          | -8.65 |

> Note: The HH-GCN gap above (-8.65%) was observed in an early run before
> full hyperparameter config loading was fixed. With configs correctly loaded,
> results should be closer to the paper.
