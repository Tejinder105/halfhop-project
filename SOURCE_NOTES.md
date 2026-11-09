# SOURCE_NOTES.md
# Half-Hop: Source Analysis and Implementation Notes

## Citation

```bibtex
@article{azabou2023half,
  title={Half-Hop: A graph upsampling approach for slowing down message passing},
  author={Azabou, Mehdi and Ganesh, Venkataramana and Thakoor, Shantanu and Lin, Chi-Heng
          and Sathidevi, Lakshmi and Liu, Ran and Valko, Michal and Veličković, Petar and Dyer, Eva L},
  journal={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2023}
}
```

**Official Repository:** https://github.com/nerdslab/halfhop  
**Paper:** https://openreview.net/forum?id=lXczFIwQkv

---

## 1. Half-Hop Mathematical Definition

### Single Edge (Equation 1)

For a directed edge `e_ij` where `i ≠ j` (not a self-loop):

```
V' = V ∪ {νk}
E' = (E \ {e_ij}) ∪ {e_{i→k}, e_{j→k}, e_{k→j}}
```

Notation: `vi → νk ↔ vj`

Both source and target nodes send messages **to** the slow node.  
Only the slow node passes information **toward** the target (original direction preserved).

### Feature Interpolation

The slow node `νk` for edge `e_ij` is initialized as:

```
x_k = α * x_i + (1 - α) * x_j
```

where `x_i` is the **source** feature, `x_j` is the **target** feature, and `α ∈ [0, 1]`.

**Note:** The paper formula (page 3) writes `x_k = (1-α)x_j + α*x_i`, which is identical.  
The official code does: `x_slow = alpha * x[src] + (1-alpha) * x[dst]` — confirmed identical.

- `α = 0`: slow node initialized to target features (x_j)
- `α = 1`: slow node initialized to source features (x_i)
- `α = 0.5`: equal mix (default, most common setting)

### Probabilistic Half-Hop (p < 1)

Node-level sampling: for each target node `vi ∈ V`, select it with probability `p`.  
Half-hop ALL incoming edges of selected nodes.

**This is node-level, NOT edge-level sampling.**

Let `S` = set of selected nodes, `E_S` = all edges with target in `S`.  
The new graph `(V', E')` is sampled as: `(V', E') ~ hh_α(G; p)`

When `p = 1`: fully deterministic transformation, denoted `HH_α(G)`.

---

## 2. Slow-Node Connectivity

### Proposed (Default)

For edge `vi → vj`, introduce slow node `νk`:
- `vi → νk`  (source to slow)
- `vj → νk`  (target to slow — backward edge)
- `νk → vj`  (slow to target — original direction)

### Ablation Variants (Appendix B.1)

**HH(1):** Remove the backward edge from target to slow:
- `vi → νk → vj`
- (No `vj → νk` edge)

**HH(2):** Add extra edge from slow back to source:
- `vi ↔ νk ↔ vj`
- Adds `νk → vi` (creates new path `vj → vi` not in original graph)

**Results (Table 1, GCN on heterophilic datasets):**

| Dataset | HH (proposed) | HH(1) | HH(2) |
|---------|--------------|-------|-------|
| Texas   | 71.71 ± 8.76 | 68.8 ± 6.50 | 58.47 ± 5.56 |
| Actor   | 33.35 ± 1.00 | 32.17 ± 0.84 | 31.93 ± 1.26 |
| Cornell | 63.42 ± 5.62 | 57.66 ± 6.89 | 42.16 ± 6.57 |

**Conclusion:** Proposed connectivity is optimal.

---

## 3. Self-Loop Treatment

Self-loops (`i == j`) are **excluded** from Half-Hop.  
They are preserved as-is in the new edge set.

```python
self_loop_mask = edge_index[0] == edge_index[1]
edge_index_self_loop = edge_index[:, self_loop_mask]
edge_index = edge_index[:, ~self_loop_mask]  # only non-self-loops are half-hopped
```

---

## 4. Slow-Node Mask

A boolean tensor `slow_node_mask` of shape `[total_nodes]`:
- `False` for original nodes (indices 0..N-1)
- `True` for slow nodes (indices N..N+N_slow-1)

After message passing, predictions use ONLY original nodes:
```python
output = model(data)  # shape [N + N_slow, out_channels]
output = output[~data.slow_node_mask]  # shape [N, out_channels]
```

---

## 5. Slow-Node Feature Initialization Ablations (Appendix B.2)

| Method | Description |
|--------|-------------|
| `linear` (default) | `xk = α*xi + (1-α)*xj` |
| `zero` | `xk = 0` |
| `random` | `xk ~ Uniform(0, 1)` |

**Results (Table 2, HH-GCN):**

| Dataset | linear | zero | random |
|---------|--------|------|--------|
| Texas   | 72.88 ± 7.17 | 61.80 ± 5.91 | 53.33 ± 5.27 |
| Actor   | 33.39 ± 1.29 | 28.93 ± 2.83 | 24.70 ± 1.16 |
| Cornell | 63.33 ± 5.70 | 49.55 ± 7.06 | 37.48 ± 6.93 |

---

## 6. Supervised Learning Procedure

**Train/val/test protocol:**
1. Train on training nodes only (loss computed on `train_mask`)
2. Select best epoch / configuration using validation accuracy only
3. Report test accuracy at best validation epoch

**Heterophilic datasets (10 pre-set splits from Pei et al. 2020):**
- Use `data.train_mask[:, split_id]`, `data.val_mask[:, split_id]`, `data.test_mask[:, split_id]`
- Report mean ± std over 10 splits

**Homophilic datasets (20 random splits):**
- 60:20:20 train/val/test split (Amazon/Coauthor)
- 20 pre-set masks for WikiCS

**Hyperparameter search:** randomized grid search on train+val sets only.

---

## 7. SSL Procedure

**Pre-training:** Full graph (transductive), two augmented views:
```
G1 ~ hh_α(G; p1)   with feature masking probability pf,1 and edge masking pe,1
G2 ~ hh_α(G; p2)   with feature masking probability pf,2 and edge masking pe,2
```

**Evaluation:** Linear evaluation protocol
- Freeze encoder weights
- Train linear classifier (L2-regularized logistic regression, scikit-learn, liblinear solver)
- Split: 10:10:80 train/val/test

**Contrastive loss:** Only original nodes used (slow nodes discarded before loss computation).

---

## 8. Datasets

### Heterophilic (from Pei et al. 2020, 10 pre-set splits)

| Dataset | Nodes | Edges | Classes | Homophily | PyG Class |
|---------|-------|-------|---------|-----------|-----------|
| Texas | 183 | 295 | 5 | 0.11 | `WebKB(name='Texas')` |
| Wisconsin | 251 | 488 | 5 | 0.21 | `WebKB(name='Wisconsin')` |
| Actor (Film) | 7,600 | 26,752 | 5 | 0.22 | `Actor` |
| Squirrel | 5,201 | 198,493 | 5 | 0.22 | `WikipediaNetwork(name='squirrel')` |
| Chameleon | 2,277 | 31,421 | 5 | 0.23 | `WikipediaNetwork(name='chameleon')` |
| Cornell | 183 | 280 | 5 | 0.30 | `WebKB(name='Cornell')` |

**Note:** Paper calls "Actor" as "Film" in Table 4. PyG uses `Actor` loader.

**Note on Squirrel/Chameleon:** The paper uses the unfiltered versions (with duplicate nodes) from Pei et al. 2020. Some later papers use filtered versions. We use the original splits.

### Homophilic (20 random splits, 60:20:20)

| Dataset | Nodes | Edges | Features | Classes | Homophily | PyG Class |
|---------|-------|-------|----------|---------|-----------|-----------|
| Amazon Photos | 7,650 | 119,081 | 745 | 8 | 0.84 | `Amazon(name='photo')` |
| Amazon Computers | 13,752 | 245,861 | 767 | 10 | 0.79 | `Amazon(name='computers')` |
| Coauthor CS | 18,333 | 81,894 | 6,805 | 15 | 0.83 | `Coauthor(name='cs')` |
| Coauthor Physics | 34,493 | 247,962 | 8,415 | 5 | 0.92 | `Coauthor(name='physics')` |
| WikiCS | 11,701 | 216,123 | 300 | 10 | 0.66 | `WikiCS` |

---

## 9. Model Architectures

### GCN
- Input → [GCNConv + ReLU + Dropout] × (depth-1) → GCNConv
- Default depth: 2; hidden: 64; dropout: 0.5
- PyG `GCNConv`

### GraphSAGE
- Same structure with `SAGEConv`
- Includes self-loop weight by default (PyG SAGEConv behavior)

### GAT
- Same structure with `GATConv`
- Default 8 attention heads (intermediate), 1 head (final layer)

### HH-GCN / HH-GraphSAGE / HH-GAT
```
data → HalfHop → GNN → filter[~slow_node_mask] → original node predictions
```

---

## 10. Reported Hyperparameters (Table 5)

### Best hyperparameters (HH models, heterophilic datasets)

| Dataset | Model | lr | wd | depth | hidden | dropout | α | p |
|---------|-------|----|----|-------|--------|---------|---|---|
| Texas | HH-GCN | 0.0291 | 0.0096 | 2 | 64 | 0.8058 | 0.0043 | 0.9526 |
| Texas | HH-SAGE | 0.0170 | 0.0053 | 2 | 64 | 0.1967 | 0.9397 | 0.7140 |
| Texas | HH-GAT | 0.0328 | 0.0066 | 2 | 32 | 0.1288 | 0.0902 | 0.9841 |
| Wisconsin | HH-GCN | 0.0105 | 0.0002 | 3 | 128 | 0.6612 | 0.9937 | 0.7140 |
| Wisconsin | HH-SAGE | 0.0202 | 0.0042 | 3 | 64 | 0.3462 | 0.0100 | 0.6177 |
| Wisconsin | HH-GAT | 0.0539 | 0.0068 | 3 | 16 | 0.2141 | 0.0026 | 0.9797 |
| Actor | HH-GCN | 0.0313 | 0.0087 | 3 | 64 | 0.5511 | 0.0369 | 0.5466 |
| Actor | HH-SAGE | 0.0133 | 0.0090 | 3 | 32 | 0.3737 | 0.0116 | 0.8368 |
| Actor | HH-GAT | 0.0009 | 0.0001 | 3 | 128 | 0.8708 | 0.0549 | 0.9594 |
| Squirrel | HH-GCN | 0.0053 | 0.0001 | 3 | 128 | 0.2455 | 0.0145 | 0.8257 |
| Squirrel | HH-SAGE | 0.0296 | 0.0001 | 2 | 128 | 0.8668 | 0.9474 | 0.5198 |
| Squirrel | HH-GAT | 0.0027 | 0.0001 | 3 | 64 | 0.5131 | 0.9277 | 0.1549 |
| Chameleon | HH-GCN | 0.0318 | 0.0057 | 2 | 128 | 0.8040 | 0.0510 | 0.9986 |
| Chameleon | HH-SAGE | 0.0225 | 0.0001 | 2 | 32 | 0.7175 | 0.9834 | 0.6226 |
| Chameleon | HH-GAT | 0.0012 | 0.0008 | 3 | 64 | 0.0439 | 0.9766 | 0.9386 |
| Cornell | HH-GCN | 0.0505 | 0.0055 | 2 | 32 | 0.4123 | 0.0145 | 0.9660 |
| Cornell | HH-SAGE | 0.0697 | 0.0018 | 2 | 64 | 0.0697 | 0.8807 | 0.5660 |
| Cornell | HH-GAT | 0.0572 | 0.0070 | 2 | 64 | 0.0572 | 0.0710 | 0.9979 |

**Note:** Baseline (GCN/SAGE/GAT without HalfHop) hyperparameters are NOT reported in Table 5. 
For baselines, we use the same search space and report best-val results.

---

## 11. Reported Results (Table 2, Heterophilic — Verified from Published ICML 2023 Proceedings PMLR 202)

Mean accuracy ± std over 10 splits (camera-ready proceedings, page 6):

| Dataset | GCN† | HH-GCN | GAT† | HH-GAT | GraphSAGE† | HH-GraphSAGE | MLP† |
|---|---|---|---|---|---|---|---|
| Texas | 55.14 ± 5.16 | 71.89 ± 3.46 | 52.16 ± 6.63 | 80.54 ± 4.80 | 82.43 ± 6.14 | 85.95 ± 6.42 | 80.81 ± 4.75 |
| Wisconsin | 51.76 ± 3.06 | 79.80 ± 4.30 | 49.41 ± 4.09 | 83.53 ± 3.84 | 81.18 ± 5.56 | 85.88 ± 3.99 | 85.29 ± 3.31 |
| Actor | 27.32 ± 1.10 | 35.12 ± 1.06 | 27.44 ± 0.89 | 36.70 ± 0.92 | 34.23 ± 0.99 | 36.82 ± 0.77 | 36.63 ± 0.70 |
| Squirrel | 31.52 ± 0.71 | 47.19 ± 1.21 | 36.77 ± 1.68 | 46.35 ± 1.86 | 41.61 ± 0.74 | 45.25 ± 1.52 | 28.77 ± 1.56 |
| Chameleon | 38.44 ± 1.92 | 60.24 ± 1.93 | 48.36 ± 1.58 | 61.12 ± 1.83 | 58.73 ± 1.68 | 62.98 ± 3.35 | 46.21 ± 2.99 |
| Cornell | 60.54 ± 5.30 | 63.24 ± 5.43 | 61.89 ± 5.05 | 72.70 ± 4.26 | 75.95 ± 5.01 | 74.60 ± 6.06 | 81.89 ± 6.40 |

*Note: Baselines marked with † are cited from Yan et al. (2021) "Two sides of the same coin: Heterophily and oversmoothing in GNNs".*
*Competitor models from literature in Table 2:*
- *MixHop: Texas 77.84, Wisconsin 75.88, Actor 32.22, Squirrel 43.80, Chameleon 60.50, Cornell 73.51*
- *GGCN: Texas 84.86, Wisconsin 86.86, Actor 37.54, Squirrel 55.17, Chameleon 71.14, Cornell 85.68*
- *H2GCN: Texas 84.86, Wisconsin 87.65, Actor 35.70, Squirrel 36.48, Chameleon 60.11, Cornell 82.70*

---

## 12. SSL Augmentation Hyperparameters (Table 6)

All datasets use the same SSL augmentation params:
- `phh,1 = phh,2 = 0.75`
- `α1 = α2 = 0.50`

Per-dataset feature/edge masking probabilities documented in paper Table 6.

---

## 13. Differences Between Paper and Official Repository

| Aspect | Paper | Official Repository | Our Implementation |
|--------|-------|--------------------|--------------------|
| Feature interpolation formula notation | `(1-α)xj + αxi` | `alpha*x[src] + (1-alpha)*x[dst]` | Same as official |
| p=1 handling | Described as deterministic | Sets `edge_index_to_keep=None`, concat with `torch.cat` handles None in list | Fixed to handle None |
| p<1 sampling | Node-level target sampling | Uses `torch_geometric.utils.subgraph` | Aligned with official |
| In-place feature update | Not specified | Uses `mul_` and `add_` in-place ops | Use equivalent safe ops |
| Default inplace | Not specified | `inplace=True` | `inplace=True` (default) |
| Baseline hyperparameters | Not reported in paper | Not provided in repo | Search same space |
| `slow_node_mask` dtype | Not specified | `bool` tensor | `bool` tensor |

---

## 14. Python 3.14 Compatibility Notes

- `torch.jit.script` is deprecated in Python 3.14+ — produces `FutureWarning`. We do not use `jit.script`.
- PyG's `typing._eval_type` usage produces `DeprecationWarning` on Python 3.14. Harmless; not our code.
- No Python version downgrade required. All core functionality works on Python 3.14.

---

## 15. Remaining Nondeterminism

Despite `set_seed()`, the following may cause non-reproducibility:
- CUDA operations (not applicable — CPU-only)
- PyG's sparse operations may use non-deterministic algorithms on some hardware
- Dataset download ordering (cached after first run)
- For p<1: stochastic node sampling uses `torch.rand` seeded by `torch.manual_seed`
