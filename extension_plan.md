# Geometry-Aware Adaptive Half-Hop for 3D Point-Cloud Graphs

Planning document. Supersedes the mesh-centric draft.
No geometry code exists yet.

**One-line thesis.** Half-Hop mitigates over-smoothing by slowing message
propagation, but it chooses edges blindly. On 3D point-cloud graphs, edges that
straddle a geometric discontinuity are the ones worth slowing. Conditioning the
selection on local surface geometry should buy more anti-over-smoothing effect per
unit of augmentation budget.

**What changed from the first draft, and why.** The first draft kept the existing
ModelNet mesh pipeline at `depth=2` with mean pooling and tried to improve on a
+0.99% effect. That framing cannot produce a meaningful result: at depth 2 there is
no over-smoothing to prevent, global mean pooling discards exactly the spatial
selectivity geometry introduces, and the effect size sits below the measurement noise
of the test set. This version keeps the point-cloud representation (which fixes
geometry estimation) and moves the headline experiment to **accuracy versus depth**,
where the mechanism is actually load-bearing and effect sizes are large enough to
measure. Segmentation is the second track, because that is where the method's
premise can be validated directly rather than assumed.

---

## 0. Starting position and hard constraints

What exists and works:

| Component | Location | State |
|---|---|---|
| Half-Hop transform | `halfhop/halfhop.py` (`HalfHop`) | Complete, tested; 3 connectivity + 3 init variants |
| GCN / GraphSAGE / GAT | `halfhop/{gcn,graphsage,gat}.py` | Complete, `depth` is already a constructor arg |
| HH-wrapped node classifiers | `halfhop/hh_*.py` | Complete — the `out[~slow_node_mask]` pattern Track B needs |
| Heterophilic reproduction | `experiments/supervised/run.py` | Complete, 10 splits, matches paper trends |
| ModelNet **mesh** pipeline | `experiments/modelnet/run.py` | Minimal: 1 seed, 2 models, no GAT, depth 2 |
| Report | `report/halfhop_final_report_exact.tex` | Complete; names this extension as future work |

Existing 3D numbers (seed 0, p=0.5, α=0.5, depth 2): GCN 67.62 → 68.61, GraphSAGE
69.82 → 69.27. **Treat these as noise, not as a result.** With 908 ModelNet10 test
graphs at ~68%, one run's binomial standard error is
\(\sqrt{0.68 \cdot 0.32 / 908} \approx 1.55\%\) — larger than both observed effects.
They stay in the report as the honest starting point; they are not the baseline the
new claim is built on.

**Compute.** The local `.venv` contains only `pip` and `setuptools` — no torch, no PyG.
All training runs on Kaggle 2×T4 (`experiments/modelnet/run_kaggle_2gpu.sh`), ~9h per
session under a weekly quota. Compute is the scarcest resource here, so every stage
below carries a run count and the sweep driver must be resumable.

**Welcome consequence of the point-cloud choice:** at a fixed 1024 points per cloud, a
naive `torch.cdist` is 1024² floats ≈ 4 MB, so kNN needs no chunking and no
`torch-cluster` (absent locally and typically on Kaggle images). The representation
switch removes the only fragile binary dependency in the plan.

---

## 1. Hypotheses, stated so each can fail

**H1 — Depth (headline).** Vanilla GCNs on kNN point-cloud graphs lose accuracy as
depth grows. Half-Hop recovers some of that loss; geometry-aware Half-Hop recovers
more at equal augmentation budget. The claim lives in the *gap between two curves*,
not in a single number.

**H2 — Mechanism.** Geometry-aware Half-Hop reduces feature homogenisation
*selectively* across high-geometry-score edges, whereas random Half-Hop reduces it
uniformly. Measured as per-layer neighbour cosine similarity, bucketed by score.

**H3 — Proxy validity.** The geometric score is a usable stand-in for label
heterophily. Testable only where per-point labels exist, i.e. Track B:
\(\mathrm{corr}\big(g_{ij},\ \mathbb{1}[y_i \neq y_j]\big)\) should be clearly positive.

H1 is the result. H2 is what makes it a mechanism rather than a coincidence. H3 is the
justification, and it is the one that can be checked *before* training anything.

### Falsification arms, built in from the start

- **Low-score selection** performing like high-score selection ⇒ geometry carries no
  usable signal.
- **Score-shuffle** (permute \(g\) within a graph, same histogram, destroyed layout)
  performing like high-score ⇒ only the score *distribution* mattered, not *where* the
  edges are.
- **Degree** or **edge-length** selection matching geometry ⇒ "geometry" was a proxy
  for a trivial structural quantity. Non-obvious risk: on kNN graphs edge length
  correlates with local sampling density, which correlates with curvature.
- **Geometry-as-node-features** under *random* Half-Hop matching geometry-aware
  selection ⇒ the contribution reduces to "we fed the model geometry", which is not a
  new augmentation method. Run this arm early; it is the cheapest way to kill the project.

---

## 2. G0 — The go/no-go gate (run this before writing any geometry code)

Implemented in `experiments/pointcloud/run.py`. Run `python -m experiments.pointcloud.run --sweep`.
Point clouds are sampled once (seed 0) and cached under `data/ModelNetPC/`; the run
seed controls the split and the weights only. Dropout defaults to 0.

Train the baseline GCN on 1024-point kNN graphs at **depth ∈ {2, 4, 8, 16}**, 3 seeds.
Twelve runs. No geometry, no Half-Hop, no new modules beyond the point-cloud loader.

| Outcome | Action |
|---|---|
| Accuracy degrades clearly with depth | Over-smoothing is real here. H1 has a large, measurable target. Proceed to Track A. |
| Depth 16 ≈ depth 2 | There is no disease to cure in this pipeline. Abandon the depth framing and go straight to Track B. |

I expect degradation — a vanilla GCN at depth 16 with no residual connections collapses
on essentially any graph — so this is likely a green light. But it costs twelve runs to
know rather than assume, and it fixes the depth at which every later experiment runs
(use the depth where the baseline has clearly degraded but not collapsed to chance).

**G0 also sets the error bars.** Record seed variance at each depth; it determines the
seed count for every table that follows.

---

## 3. Blockers in the current code

These silently invalidate comparisons and must be fixed first.

### 3.1 Half-Hop samples *nodes*, not *edges*

```python
# halfhop/halfhop.py:101-109
else:
    node_mask = torch.rand(data.num_nodes, device=device) < self.p
    _, _, edge_mask = subgraph(
        node_mask,
        torch.stack([edge_index[1], edge_index[1]], dim=0),
        return_edge_mask=True,
    )
    edge_index_to_halfhop = edge_index[:, edge_mask]
    edge_index_to_keep = edge_index[:, ~edge_mask]
```

`p` selects *target nodes*, then half-hops **all incoming edges** of each selected
target. A geometry rule is inherently per-edge, so the published node-level Half-Hop
differs from it in *granularity as well as criterion* — precisely the confound this
project exists to avoid. Expected edge fraction matches (\(E[\text{frac}] = p\)) but the
correlation structure does not.

Fix: add an injectable selector to `HalfHop` (see §5). The scientific control is
**edge-level random at matched budget**. Node-level HH stays in the paper as
"Half-Hop as published".

### 3.2 `data.x` is an alias of `data.pos`

```python
# experiments/modelnet/run.py:34-37
def __call__(self, data):
    if data.x is None:
        data.x = data.pos
    return data
```

Same tensor object. After `HalfHop`, `x` has `N + N_slow` rows while `pos` still has `N`,
so any geometry code indexing `pos` by post-Half-Hop node id is wrong. Fix:
`data.x = data.pos.clone()`, and give slow nodes the interpolated position
\(\alpha p_i + (1-\alpha) p_j\) so visualisation works.

### 3.3 The coordinate-feature confound — the subtle one

With `x = pos` and `slow_node_init='linear'`, a slow node's features *are* the 3D
midpoint of its two endpoints. So on a point cloud, Half-Hop is simultaneously
(a) slowing propagation and (b) **inserting interpolated points**. You cannot attribute
any gain to the mechanism you are claiming, because the two operations are identical
under coordinate features.

Worse for the hypothesis: on a smooth patch the midpoint lies essentially on the
surface and adds nothing; at a crease it lies *off* the surface, inside the solid. A
geometry-aware rule that targets creases therefore preferentially inserts off-surface
artefacts at exactly the informative regions. This could plausibly *hurt*, and it is a
mechanism worth knowing about either way.

Disentangle with two arms that cost almost nothing:
- `slow_node_init='zero'` — pure propagation delay, no interpolated geometry.
- richer input features so \(x \neq \text{pos}\) (e.g. `x = [pos, normal]`), which is the
  standard point-cloud setup anyway.

### 3.4 Smaller freezes

| Issue | Location | Fix |
|---|---|---|
| No GAT in the 3D runner | `experiments/modelnet/run.py:89-99` | Add `gat` / `hh-gat`; needed for the attention baseline (§8) |
| Results are append-only prose lines | `run.py:302-318` | One JSON line per run with the full config; a depth × method × seed matrix is unreadable as text |
| `if val_acc >= best_val` | `run.py:288` | Use `>`; `>=` biases selection toward late epochs |
| Val split varies with seed | `run.py:243` | Keep it — split variance *should* be in error bars — but state it in the report |
| `depth` not swept | CLI already has `--depth` | Just needs the sweep driver |

### 3.5 Layout collision

The folder sketch from the original brief has a top-level `data/` package, but `data/`
is already the PyG download root (`--data-dir data/ModelNet`). It also proposes
`models/`, duplicating `halfhop/gcn.py`. §5 adapts to existing conventions instead.

---

## 4. Design decisions

Settled once, in writing, rather than discovered mid-experiment.

**D1 — Point sampling.** `NormalizeScale()` → `SamplePoints(1024, include_normals=True)`
→ kNN graph, all in `pre_transform` so the sampling is **fixed and cached**. PyG
normally applies `SamplePoints` as a runtime transform for augmentation; here a fixed
sample is required so that cached geometry stays valid and the graph is identical
across arms. Note that `include_normals=True` gives face-derived normals — effectively
oracle normals, which removes a noise source. Keep the PCA path too (real scans have no
faces), and compare them: *does the method need accurate normals?* is a cheap, genuinely
interesting ablation.

**D2 — Neighbourhoods.** kNN over positions, \(k \in \{8, 16, 32\}\). On uniformly
sampled clouds this is well-posed, unlike kNN over raw CAD mesh vertices, where density
tracks tessellation detail rather than surface area. Naive `cdist` at 1024 points is
cheap enough that no `torch-cluster` dependency is needed.

**D3 — Directed arcs vs undirected pairs.** All proposed scores are symmetric, so top-K
over directed arcs selects both arcs of a pair together while random node-level
sampling does not. Select over **undirected pairs**, half-hop both arcs, and apply the
identical granularity to every arm including random.

**D4 — Exact top-K, not Bernoulli, for the headline.** \(\text{Bernoulli}(p_{ij})\) does
not fix the number of half-hopped edges, so a probabilistic arm differs from random HH
in *both* which edges and how many. Use \(K = \mathrm{round}(\rho \cdot |E_{\text{und}}|)\)
per graph with \(\rho\) matched across arms. Note \(\rho = 1.0\) makes all arms identical
by construction — a free sanity check on the harness. The adaptive/probabilistic form
\(p_{ij} = p_{\min} + (p_{\max}-p_{\min})\tilde g_{ij}\) is a later variant, reported once
top-K has established or refuted the effect.

**D5 — Random tie-breaking is mandatory.** Even on sampled clouds, flat regions give
\(\lambda_1 \approx 0\) and near-zero normal disagreement, so expect heavy mass near
\(g=0\). A naive `topk` breaks ties by tensor index order, which follows point sampling
order — an artefact, not a signal. Add \(\epsilon \cdot \mathrm{Uniform}(0,1)\) below the
smallest meaningful gap, and log the tie fraction.

**D6 — Per-graph rank normalisation** of \(g\), not global min-max. The budget is
per graph, and rank normalisation is robust to the spiky distributions of D5.

**D7 — Geometry computed once, in `pre_transform`, cached.** Per-epoch kNN plus
eigendecomposition over thousands of clouds would dominate training. Changing
`pre_transform` invalidates the PyG processed cache, so budget one session for the
reprocess, then **save the processed directory as a Kaggle Dataset** so it never
recomputes. Geometry is deterministic; the only per-epoch randomness is tie-breaking
and the Bernoulli draw in the adaptive variant.

**D8 — "Adaptive" means geometry-conditioned, not learned.** A learned
\(p_{ij} = \sigma(\mathrm{MLP}(g_{ij}))\) changes two things at once. It is §10, not v1.

---

## 5. Module layout

```
halfhop/
├── halfhop.py              # EXTEND: injectable selector; defaults byte-identical to today
├── geometry/               # NEW
│   ├── neighbors.py        # knn (naive at 1024 pts; chunked fallback for larger)
│   ├── descriptors.py      # covariance -> eigh -> normals, curvature, planarity, linearity
│   ├── edge_scores.py      # node geometry -> g_ij; plus degree/length/random controls
│   ├── selection.py        # topk_with_random_tiebreak, bernoulli_from_score, shuffle_scores
│   └── transforms.py       # ComputeGeometry pre_transform -> data.normal/curv/edge_score
├── dropedge.py             # NEW: random + geometry-aware DropEdge (competing operator, §8)
experiments/
├── pointcloud/
│   ├── run.py              # NEW: Track A — SamplePoints + kNN + depth sweep + pooling
│   └── sweep.py            # NEW: resumable driver, writes JSONL
├── segmentation/
│   └── run.py              # NEW: Track B — ShapeNet-Part, per-point loss, mIoU
├── analysis/
│   ├── smoothing.py        # NEW: per-layer cosine similarity, bucketed by score (H2)
│   ├── proxy.py            # NEW: corr(g_ij, label disagreement) (H3)
│   └── aggregate.py        # NEW: JSONL -> paired stats + report tables
├── modelnet/run.py         # KEEP: mesh baseline, reported as-is
visualize/
├── geometry.py             # normals/curvature maps on a cloud
└── halfhop_edges.py        # which edges were selected, overlaid on the object
tests/
└── test_geometry.py        # analytic ground truths
```

Extend `HalfHop` rather than forking a `geometry_halfhop.py`: the slow-node construction
(connectivity schemes, batch/`ptr` handling, self-loop isolation) is intricate and
already correct, and a copy will drift. Every current default must be preserved so the
heterophilic results stay reproducible — `tests/test_halfhop.py` is the regression guard.

```python
# halfhop/halfhop.py
class HalfHop:
    def __init__(self, alpha=0.5, p=1.0, inplace=True,
                 slow_node_init='linear', connectivity='proposed',
                 selection='node',        # 'node' (current default) | 'edge' | 'score'
                 score_key=None,          # data attribute holding per-edge g
                 score_mode='high',       # 'high' | 'low' | 'shuffle'
                 budget=None,             # exact fraction rho for 'edge'/'score'
                 undirected_pairs=True):
```

Assert with a seeded equality test that `selection='node'` plus current defaults
reproduces today's output exactly.

---

## 6. Track A — Point-cloud classification, depth as the axis

### A1. Geometry primitives, verified against closed forms · 0 runs

centroid → covariance → batched `torch.linalg.eigh` on `(N,3,3)` → \(n_i = v_{\min}\),
\(\kappa_i = \lambda_1/(\lambda_1+\lambda_2+\lambda_3+\epsilon)\). Also emit
`planarity = (λ₂−λ₁)/λ₃` and `linearity = (λ₃−λ₂)/λ₃` — free, and a second descriptor
family for the ablation.

| Test input | Expected |
|---|---|
| Points on a plane | κ = 0, n ⟂ plane |
| Points on a line | linearity → 1, κ = 0 |
| Isotropic Gaussian ball | κ → 1/3 |
| Unit-sphere patch, radius r | κ grows with r/R, known to first order |
| Cube corner, 3 orthogonal faces | κ ≈ 1/3, no stable normal |
| Duplicate / single point | no NaN, flagged degenerate |
| PCA normals vs `include_normals` | high \(|n_{\text{pca}} \cdot n_{\text{face}}|\) away from creases |

Exit: all tests pass, no NaNs on a full dataset pass, degenerate fraction logged.
Do the cube-corner case numerically by hand before trusting the code — it is what makes
the \(\lambda_1\) degeneracy of D5 obvious before it quietly biases a top-K selection.

### A2. Geometry sanity on real objects · 0 runs

Colour-map curvature and normal disagreement on a chair, an aeroplane, a monitor. Plot
score histograms and the tie mass.

Exit criterion as a defensible claim: *high-score edges coincide with creases, corners
and part boundaries; flat regions are low-score.* If the map instead tracks sampling
density, the descriptor is broken — fix \(k\) or the normalisation before proceeding.

### A3. Scores and selection · 0 runs

| Score | Definition | Role |
|---|---|---|
| `normal` | \(1 - |n_i^\top n_j|\) | primary |
| `curv` | \((\kappa_i + \kappa_j)/2\) | primary |
| `dcurv` | \(|\kappa_i - \kappa_j|\) | primary |
| `combined` | rank-average of `normal`, `curv` | primary |
| `length` | \(\|p_i - p_j\|\) | **control** |
| `degree` | \(\deg_i + \deg_j\) | **control** |
| `random` | uniform | **control** |

Exit: all arms select exactly equal counts on a fixed graph; `shuffle` preserves the
histogram but not the layout; selection is seed-reproducible.

### A4. The headline experiment — accuracy vs depth · ~90 runs

Depth ∈ {2, 4, 8, 16} × 3 seeds × arms:

1. baseline, no augmentation
2. HH, edge-random, ρ = 0.5
3. HH, geometry-high (`normal`), ρ = 0.5
4. HH, geometry-low ← falsification
5. HH, score-shuffle ← falsification
6. DropEdge, random ← competing operator (§8)
7. DropEdge, geometry-aware ← does geometry conditioning transfer across operators?

Deliverable: the central figure of the paper — accuracy against depth, one line per arm,
error bars over seeds. The claim is the *gap between curves 2 and 3*, widening with depth.

### A5. Budget curve · ~30 runs

ρ ∈ {0.1, 0.25, 0.5, 0.75, 1.0} × {random, geometry-high} × 3 seeds, at the best depth
from A4.

Expect the contrast to be **largest at low ρ**, and this is worth stating in advance as
a prediction. Geometric discontinuities are a minority of edges — perhaps 10% on a
typical object. At ρ = 0.5 geometry takes all the creases while random already takes
half of them, so the arms differ in crease coverage by only ~5% of edges. At ρ = 0.1
geometry takes ~10% that are nearly all creases while random takes 10% of which ~1% are
creases — a ~9% difference in coverage. **Low budget is where the hypothesis has teeth**,
so do not inherit the ρ = 0.5 that was originally chosen to fit a T4.

### A6. Confound and attribution arms · ~30 runs

At the best depth and budget, 3 seeds each:
`degree` · `length` · geometry-as-node-features with random HH ·
`slow_node_init='zero'` (§3.3) · PCA normals vs face normals (D1) · k ∈ {8, 16, 32}.

### A7. Mechanism analysis (H2) · analysis only

Per-layer neighbour similarity
\(S^{(l)} = \frac{1}{|E|}\sum_{(i,j)\in E}\cos(h_i^{(l)}, h_j^{(l)})\),
computed **separately over high-score and low-score edge buckets**, for baseline /
random HH / geometry HH at depth 16.

This is where the paper's actual thesis lives: geometry-aware Half-Hop should slow
mixing *selectively*, while random HH slows it uniformly. That is a qualitatively
different claim from a delta in accuracy, and it survives a null H1. Depth 16 is deep
enough for the trend to be visible — the original `depth=2` could never show it.

### A8. Qualitative figure · 0 runs

Render selected edges on the object. The report currently has no such figure, and it is
what convinces a reader the method is geometrically meaningful rather than numerically
lucky.

### A9. Robustness · ~24 runs, budget permitting

Coordinate noise σ ∈ {0.01, 0.02, 0.05}; point dropout {10, 20, 30}%; density variation.
Be ready to lose here: PCA descriptors degrade under noise while random selection is
noise-free by construction. A method that needs clean geometry is still a result, but it
has to be stated rather than omitted.

---

## 7. Track B — Part segmentation, where the premise can be validated

Run from A4 onward, not deferred to the end. ShapeNet-Part is available directly as
`torch_geometric.datasets.ShapeNet(root, categories, include_normals=True)`: 16
categories, 50 parts, per-point labels, normals included.

Why this track matters more than its compute cost suggests:

1. **The mechanism applies.** Per-point labels make this genuine node classification —
   Half-Hop's actual published setting. Points either side of a seat/leg boundary have
   different labels *and* different local geometry at the same place. No global pooling
   to wash out the spatial selectivity.
2. **H3 becomes measurable.** Compute \(\mathrm{corr}(g_{ij}, \mathbb{1}[y_i \neq y_j])\)
   directly (`analysis/proxy.py`). If geometric disagreement predicts label disagreement,
   the method has a principled reason to work, stated before any training. In
   classification this correlation is undefined, which is exactly why that framing
   cannot close the argument.
3. **Measurement is far better.** ~2,870 test shapes × 2,048 points. mIoU differences of
   0.3% are routinely treated as real, versus the ~1.5% noise floor on ModelNet10
   classification.
4. **Minimal new code.** Per-point prediction reuses the existing
   `out[~data.slow_node_mask]` pattern from `halfhop/hh_gcn.py` more naturally than the
   pooled classifier does.

Staging: run `analysis/proxy.py` **first** — it needs no training at all and gates the
rest of the track. Then 2 categories (Airplane, Chair) for development, full 16 for the
final table. Arms: baseline / random HH / geometry HH at matched budget, 3 seeds.

---

## 8. Baselines and controls

The three objections most likely to sink this, and what answers each.

**"Attention already does this, and learns it."** GAT, transformers and EdgeConv/DGCNN
all down-weight edges, and DGCNN — which recomputes kNN per layer and reaches ~93% on
ModelNet40 — is already geometry-aware dynamic rewiring. Geometry-aware Half-Hop is a
hand-designed, non-learned, *hard* version of soft edge reweighting.

There is a real answer: Half-Hop changes the graph's **effective diameter** and delays
propagation, which reweighting cannot do. But it must be argued and tested, so include
a **GAT + geometric edge features** arm. If attention with \(\kappa_i, \kappa_j,
n_i^\top n_j\) as edge features matches geometry-aware Half-Hop, the contribution is
positioning, not method.

**"DropEdge already fights over-smoothing with edge-level stochasticity."** This is the
most directly comparable prior method, so it is a baseline (arm 6 in A4) — and a
geometry-aware DropEdge (arm 7) turns the objection into a strength: if geometry
conditioning helps *both* operators, the contribution is the conditioning principle,
not one augmentation.

**"Just use residual connections / GCNII / PairNorm."** Residuals are cheap to add and
must appear as a depth-robustness baseline. Honest framing: these are architectural
fixes, Half-Hop is an augmentation, and the interesting question is whether geometry
conditioning adds anything *on top of* a residual backbone. Include a
`residual + geometry-HH` cell at depth 16.

---

## 9. Validity threats and their answers

| Threat | Answer |
|---|---|
| Unequal number of half-hopped edges across arms | Exact top-K (D4); ρ = 1.0 identity check (A5) |
| Node-level vs edge-level selection granularity | Edge-level random baseline (§3.1) |
| Index-order bias from score ties | Random tie-breaking (D5); report tie fraction |
| Propagation delay confounded with point interpolation | `slow_node_init='zero'` arm (§3.3) |
| Geometry is a proxy for degree | `degree` arm (A6) |
| Geometry is a proxy for edge length / sampling density | `length` arm (A6); uniform sampling (D1) |
| "You just added geometry to the features" | geometry-as-features arm (A6) — run early |
| Score distribution vs score layout | `shuffle` arm (A4) |
| Attention would do it better | GAT + geometric edge features (§8) |
| DropEdge would do it just as well | DropEdge arms (A4) |
| Residuals make it moot | residual backbone cell (§8) |
| Effect smaller than seed noise | G0 error bars first; paired stats (§10) |
| GCN-specific quirk | GraphSAGE and GAT at best settings |
| Over-smoothing never actually occurs here | G0 gate (§2) |

---

## 10. Statistics protocol

Fixed before looking at any result, so the analysis is not chosen after the fact.

- 3 seeds for exploratory stages, **5 for every reported table**, more if G0 shows
  σ > 1%.
- Seeds control weight init, Half-Hop sampling, tie-breaking *and* the train/val split
  (`run.py:243`), so error bars include split variance. Say so in the report.
- Model selection on best validation metric; report test at that epoch; use `>` not `>=`.
- Compare **paired by seed**: \(\Delta_s = \mathrm{acc}_{\text{geom}}(s) -
  \mathrm{acc}_{\text{rand}}(s)\), with a bootstrap CI over seeds. With 5 seeds a CI is
  honest; a p-value is theatre. For a shared test set, McNemar on per-sample
  correctness has more power than the naive binomial SE suggests.
- Report every arm that was run. No dropping arms that failed to cooperate.
- One JSONL row per run with the full config, so `aggregate.py` can rebuild every table
  and nothing depends on remembering which text file was which.
- Tag dev-tier runs (reduced epochs/subset) in the JSONL so they can never leak into a
  reported table.

---

## 11. Compute budget

| Stage | Runs | Gate |
|---|---:|---|
| G0 depth gate | 12 | **blocks everything** |
| A1–A3 geometry + selection | 0 | local, no GPU |
| A4 headline depth × arms | ~90 | the paper's main figure |
| A5 budget curve | ~30 | conditional on A4 separation |
| A6 confounds | ~30 | required for publication |
| A7 mechanism | analysis only | reuses A4 checkpoints |
| Track B proxy check | 0 | **gates Track B**, no GPU |
| Track B training | ~18 | 2 categories dev, then full |
| A9 robustness | ~24 | truncate first if quota runs short |
| §8 competing baselines | ~20 | required for publication |
| **Total** | **~225** | versus ~1125 for the original full matrix |

Levers, both built in G0:

- **Dev tier**: fewer epochs, smaller subset, for debugging only.
- **Cache the processed dataset as a Kaggle Dataset** once the geometry `pre_transform`
  is final, so reprocessing never eats a session again.
- `sweep.py` must **skip configs already present in the JSONL**. Kaggle sessions die,
  and a sweep that restarts from zero never finishes.

---

## 12. What the paper can claim, by outcome

**If H1 holds.** Half-Hop's uniform edge selection is suboptimal on geometric graphs.
Conditioning selection on local surface heterogeneity improves depth-robustness of
point-cloud GNNs at equal augmentation budget, survives degree/length/feature controls,
and transfers to a second augmentation operator (DropEdge).

**If H1 is null but H2 holds.** Geometry-conditioned Half-Hop gives *spatially targeted*
control of propagation — demonstrated by per-layer similarity measured separately over
high- and low-score edges — which does not translate into classification accuracy under
global pooling. A real negative result with a mechanism, pointing at where it should
work. Track B is then the natural follow-up and may already contain the positive result.

**If H3 is near zero** (geometric disagreement does not predict label disagreement), the
premise is refuted cheaply and honestly, with no GPU time spent. That is a legitimate
finding about geometric priors for graph augmentation, and far more than the current
single-seed table says.

---

## 13. Next three actions

1. **Fix §3.1–3.4** — injectable edge-level selector in `HalfHop`, `pos`/`x` aliasing,
   GAT in the 3D runner, JSONL logging. Verify `tests/test_halfhop.py` passes unchanged.
   No geometry yet.
2. **Build `experiments/pointcloud/run.py`** (`NormalizeScale` → `SamplePoints(1024,
   include_normals=True)` → naive kNN → GCN → pool) and **run G0**: depth {2,4,8,16},
   3 seeds, one Kaggle session. This answers whether the project has a target at all.
3. **In parallel, locally** (the machine cannot train anything anyway): the cube-corner
   eigendecomposition by hand, then `geometry/descriptors.py` against the A1 analytic
   tests, then `analysis/proxy.py` on ShapeNet to get the H3 correlation — the single
   cheapest piece of evidence for or against the whole premise.

Actions 1–2 are the critical path and involve no geometry. Action 3 is free and can
invalidate the premise before any compute is spent.
