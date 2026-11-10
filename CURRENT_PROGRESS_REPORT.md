# Half-Hop Repository: Work Completed So Far

**Repository reviewed:** `halfhop-project`  
**Review date:** 2026-11-10  
**Purpose:** Consolidated record of the implementation, experiments, plans, results, gains, limitations, and remaining work currently visible in the repository.

> This report is an evidence-based snapshot of the files and commit history. It records negative results and protocol limitations as well as successful results. Percentages below are reported values from the repository's existing reports; they are not newly rerun by this documentation change.

## 1. Executive Summary

The project has progressed from a basic Half-Hop implementation to a broad research and reproduction suite:

1. Reimplemented the ICML 2023 Half-Hop method for heterophilic graph learning.
2. Added GCN, GraphSAGE, and GAT baselines plus Half-Hop wrappers.
3. Added reproducible datasets, masks/splits, training, evaluation, and seed handling.
4. Added connectivity and slow-node-initialization ablations.
5. Added supervised and self-supervised experiment runners, including BGRL and GRACE prototypes.
6. Added Kaggle-oriented multi-GPU runners and dataset/cache optimizations.
7. Added ModelNet mesh and point-cloud experiments.
8. Added geometry-aware Half-Hop research on ShapeNet-Part, including a rigorous Stage 0 proxy gate and Stage 1 results.
9. Added selective Half-Hop, self-loop/ego-preservation controls, DropEdge, depth-matched dilation, MultiDelay, and AdaHop.
10. Added extensive result reports, raw result artifacts, tests, and a LaTeX report.

The most important scientific conclusion is now more nuanced than “Half-Hop always improves GNNs”:

- On heterophilic citation/web graphs, Half-Hop can improve over untuned plain GCN, but tuned ego-preserving alternatives such as Lazy Walk can match or exceed it without graph expansion.
- At greater depth, MultiDelay can outperform standard one-delay Half-Hop on several small heterophilic benchmarks.
- On ModelNet10 point-cloud classification, Half-Hop is harmful under the reported six-layer protocol; receptive-field starvation and feature interpolation away from the surface are plausible mechanisms.
- Geometry contains a moderate signal for identifying label-disagreeing edges, but geometry-aware selection is not universally better during downstream training.

## 2. Current Repository State

### 2.1 Repository and working tree

- The active Git repository is [`halfhop-project`](./).
- Branch: `main`, tracking `origin/main`.
- The repository has a long experiment-driven history from 2026-09-22 through 2026-11-10.
- At review time, the working tree contains one pre-existing modification:
  - [`experiments/results/modelnet10_report.md`](./experiments/results/modelnet10_report.md)
- This report does not overwrite or revert that modification.
- The repository also contains generated data, archives, reports, and a local `.venv`; these were treated as project context rather than source deliverables.

### 2.2 Project size

- 83 tracked Python files.
- 15 tracked Markdown files.
- 17 tracked test files.
- Core package requirement: Python `>=3.10`.
- Main dependencies: PyTorch, PyTorch Geometric, NumPy, SciPy, pandas, matplotlib, scikit-learn, PyYAML, pytest, and tqdm.

## 3. Core Implementation Completed

### 3.1 Half-Hop transformation

Implemented in [`halfhop/halfhop.py`](./halfhop/halfhop.py).

The transformation now supports:

- Proposed connectivity:
  - source → slow node
  - target → slow node
  - slow node → target
- `alpha` interpolation:
  - `x_slow = alpha * x_source + (1 - alpha) * x_target`
- Node-level sampling matching the published method.
- Exact edge-level random selection for controlled comparisons.
- Score-based edge selection, high-score or low-score.
- Self-loop preservation and exclusion from Half-Hop.
- `slow_node_mask` to remove synthetic nodes before classification/loss.
- Three slow-node initializations:
  - `linear`
  - `zero`
  - `random`
- Three connectivity modes:
  - `proposed`
  - `hh1`
  - `hh2`
- Input validation for probability, interpolation, selection, budget, score mode, initialization, and connectivity.

**Gain:** The project can reproduce the published mechanism while also supporting controlled ablations and later adaptive-selection experiments without duplicating the transformation.

### 3.2 GNN backbones and wrappers

Implemented in:

- [`halfhop/gcn.py`](./halfhop/gcn.py)
- [`halfhop/graphsage.py`](./halfhop/graphsage.py)
- [`halfhop/gat.py`](./halfhop/gat.py)
- [`halfhop/hh_gcn.py`](./halfhop/hh_gcn.py)
- [`halfhop/hh_graphsage.py`](./halfhop/hh_graphsage.py)
- [`halfhop/hh_gat.py`](./halfhop/hh_gat.py)

The repository supports:

- Standard GCN, GraphSAGE, and GAT.
- Corresponding Half-Hop variants.
- Configurable depth, hidden dimensions, dropout, and model-specific parameters.
- Original-node-only predictions after augmented message passing.

**Gain:** Baselines and Half-Hop models share a common implementation surface, making comparisons easier to reproduce and reducing architecture-specific duplication.

### 3.3 Reproducible training and evaluation

The repository includes training, evaluation, dataset, split, and reproducibility utilities referenced by the project documentation and tests.

The intended protocols include:

- Train only on training nodes.
- Select epochs/configurations using validation accuracy.
- Report test accuracy at the selected validation epoch.
- Ten pre-defined splits for heterophilic datasets.
- Twenty random splits for several homophilic datasets.
- Seed control for Python, NumPy, and PyTorch.
- Automatic CPU/GPU selection in experiment runners.

**Gain:** Experiments have a consistent evaluation framework instead of ad-hoc scripts for each dataset.

## 4. Reproduction Work Completed

### 4.1 Datasets and benchmark coverage

The supervised benchmark suite covers:

- Texas
- Wisconsin
- Actor/Film
- Cornell
- Squirrel
- Chameleon

Additional coverage includes:

- Amazon Photo
- Amazon Computers
- Coauthor CS
- Coauthor Physics
- WikiCS
- Amazon-ratings
- ModelNet10
- ModelNet40
- ShapeNet-Part categories

The dataset details, graph statistics, model architecture notes, paper hyperparameters, and reproduction differences are documented in [`SOURCE_NOTES.md`](./SOURCE_NOTES.md).

**Gain:** The project covers both the original heterophilic setting and several out-of-domain extensions.

### 4.2 Paper alignment and known differences

The repository explicitly documents:

- The paper's slow-node interpolation convention.
- Published node-level sampling semantics.
- Self-loop handling.
- The published connectivity variants.
- Slow-node masking.
- Baseline hyperparameter differences.
- SSL protocol differences.
- CPU versus A100 hardware differences.

[`REPRODUCTION_NOTES.md`](./REPRODUCTION_NOTES.md) records that baseline models use PyTorch Geometric defaults when the original paper's baseline grid-search results are unavailable. It also records that an early Texas HH-GCN result was below the paper target before configuration loading was corrected.

**Gain:** Results can be interpreted with their protocol and hardware context instead of being presented as exact paper replications when they are not.

### 4.3 Ablation results

The implementation and source notes cover:

- Connectivity:
  - Proposed connectivity performed best in the cited Texas, Actor, and Cornell comparisons.
  - `hh1` and `hh2` were weaker controls.
- Slow-node initialization:
  - Linear interpolation outperformed zero and random initialization in the cited Texas, Actor, and Cornell results.

**Gain:** The project tests whether the result depends on the specific graph motif and feature initialization rather than treating Half-Hop as a single opaque operation.

## 5. Experiment Plans and What Each Achieved

### Plan A — Core paper reproduction

**Goal:** Reimplement Half-Hop and compare GCN, GraphSAGE, and GAT with and without Half-Hop on the paper's heterophilic benchmarks.

**Completed work:**

- Core transform and model wrappers.
- Dataset loaders and split handling.
- Supervised runners.
- Hyperparameter/configuration support.
- Result collection and documentation.

**What was gained:**

- A reusable implementation of the paper method.
- Reproducible benchmark infrastructure.
- A documented comparison point against published numbers.

**Important qualification:** Some baseline and SSL values are not directly comparable to the paper because the original grid searches and SSL head/training details were not fully available.

### Plan B — Self-supervised learning

**Goal:** Test Half-Hop in BGRL and GRACE-style graph contrastive learning.

**Completed work:**

- BGRL fixes and an official BGRL + Half-Hop Table 3 runner.
- A BGRL prototype.
- A GRACE implementation/runner.
- Feature-drop and edge-drop experiment support.

**What was gained:**

- The project can investigate Half-Hop beyond supervised node classification.
- The augmentation pipeline can be compared with the paper's SSL setting.

**Qualification:** The repository describes these as prototypes/reproductions with possible differences in linear evaluation heads, epochs, and tuning.

### Plan C — Hyperparameter sensitivity of `alpha` and `p`

**Goal:** Measure how interpolation and augmentation coverage affect HH-GCN.

**Completed work:**

- A 20-cell grid over `alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` and `p ∈ {0.25, 0.50, 0.75, 1.00}`.
- Runs on Texas, Cornell, Wisconsin, and Chameleon.
- JSONL result files and heatmaps.

**Reported result:**

- `alpha=0.1, p=1.0` was the best configuration under that fixed protocol on all four datasets.
- Reported gains over the protocol baseline were:
  - Texas: `+11.36%`
  - Cornell: `+8.92%`
  - Wisconsin: `+9.22%`
  - Chameleon: `+1.73%`

**What was gained:**

- A sensitivity surface rather than a single default setting.
- Evidence supporting an ego-preservation interpretation of small `alpha`.

**Limitation:** The best configuration was selected using mean test accuracy, not validation accuracy. The report correctly marks these improvements as optimistic and not suitable as a final generalization claim.

### Plan D — Half-Hop plus DropEdge

**Goal:** Test whether structural edge dropout and Half-Hop provide complementary benefits in supervised heterophilic learning.

**Completed work:**

- DropEdge transform.
- DropEdge/HH model variants.
- Depth curves at multiple depths.
- Per-dataset summaries and raw JSONL outputs.

**Reported result:**

- Positive synergy appeared in 7 of 12 cells and negative synergy in 5 of 12.
- Mean synergy was only `+0.36%` with approximately `2%` standard deviation.

**What was gained:**

- A direct supervised test of a combination not evaluated in the same way by the original paper.
- A measured negative/mixed result: no consistent super-additive benefit at the tested scale.

**Limitation:** Only one DropEdge probability was tested, and some datasets have very small test sets.

### Plan E — Geometry-aware adaptive Half-Hop on point clouds

**Goal:** Select edges at geometric discontinuities instead of selecting edges blindly, using the hypothesis that geometric boundaries are more likely to be label-disagreeing.

**Design improvement made:**

- The initial shallow ModelNet mesh/global-pooling idea was rejected as too weak: depth two did not provide a strong over-smoothing target and global pooling could wash out spatial selectivity.
- The plan moved to point-cloud graphs and staged gates, documented in [`extension_plan.md`](./extension_plan.md).

**What was gained from the redesign:**

- A falsifiable depth-based hypothesis.
- Explicit controls for random, low-score, shuffled-score, edge-length, and degree selection.
- A pre-training proxy gate before spending substantial GPU time.
- Boundary-aware evaluation rather than relying only on global accuracy.

### Plan F — Stage 0 geometry proxy gate

**Goal:** Verify that local geometry predicts label-disagreeing edges before training a GNN.

**Completed work:**

- Shape-level train/test separation on Airplane, Chair, Lamp, and Table.
- Local normal angle, curvature difference, planarity difference, and linearity difference features.
- Edge-length and density controls.
- Per-shape AUROC, confidence intervals, precision, recall, and lift.

**Reported result:**

- Overall mean per-shape AUROC: `0.6968`.
- Pooled AUROC: `0.6859`.
- Gate G0 threshold: `0.65`.
- G0: passed, but modestly.
- At a 10% budget, normal-angle selection reported approximately:
  - Airplane: `2.39x` lift
  - Chair: `1.95x` lift
  - Lamp: `2.37x` lift
  - Table: `1.85x` lift

**What was gained:**

- Evidence that geometry contains useful, though imperfect, boundary signal.
- Controls showing that planarity/linearity were near chance and that simple edge length can sometimes be competitive.
- A safer basis for downstream geometry experiments.

**Limitation:** The signal is narrow on Table and Chair, and heterophilic edges are rare.

### Plan G — Stage 1 geometry-selected GNN evaluation

**Goal:** Compare baseline, all-edge Half-Hop, random-budget Half-Hop, geometry-selected Half-Hop, oracle selection, and inverted selection.

**Completed work:**

- Controlled experiments across four ShapeNet-Part categories and three seeds.
- Multi-GPU runner and result reporting.
- Gate G1 evaluation.

**Reported result:**

- Geometry-directed Half-Hop beat uniform/random Half-Hop on:
  - Airplane: `+1.92%`
  - Chair: `+0.72%`
  - Lamp: `+0.89%`
- It underperformed on Table by `-1.46%`.
- G1 was reported as cleared.

**What was gained:**

- A positive result in three categories.
- An honest failure mode: geometric discontinuity does not always equal a useful message-passing boundary.
- A stronger understanding of when geometry-directed slowing is useful.

### Plan H — Selective Half-Hop on small heterophilic graphs (Option A)

**Goal:** Determine whether edge-level selective slowing can outperform all-edge Half-Hop and whether the benefit is structural or primarily ego preservation.

**Completed work:**

- Oracle-first selector study.
- Exact edge-budget controls.
- Self-loop reweighting control.
- APPNP control.
- 3,960 reported runs across Texas, Cornell, Wisconsin, and Chameleon.

**Reported result:**

- `oracle_train` was reported as 14–19% worse than all-edge Half-Hop.
- Tuned GCN self-loop weight `w=16` and APPNP matched Half-Hop within 95% bootstrap confidence intervals.

**What was gained:**

- Strong evidence against the claim that selecting only label-disagreeing edges is automatically better on these small graphs.
- Evidence that much of the observed Half-Hop advantage can be explained by preserving ego/self information.
- A lower-cost baseline family for future comparisons.

### Plan I — Phase 1 baseline and backbone diagnostics

**Goal:** Investigate whether reproduction gaps came from model, hyperparameter, or protocol differences.

**Completed work:**

- Wisconsin/Chameleon reproduction diagnostic.
- All-backbone reproduction script.
- Phase 1 controls and results.
- Kaggle execution runners.

**What was gained:**

- Better separation of implementation differences from scientific effects.
- A way to run GCN, GraphSAGE, and GAT under matched conditions.
- More reliable diagnosis of why local values can differ from paper values.

### Plan J — Depth-matched dilation and MultiDelay

**Goal:** Compare path dilation fairly at greater depth and test whether more than one inserted delay is useful.

**Completed work:**

- Depth-matched dilation study.
- Lazy Walk/self-loop-weight controls.
- MultiDelay implementation and runner.
- Results at depth `L=4` over Wisconsin, Texas, Chameleon, and Cornell.

**Reported result:**

- Uniform MultiDelay (`k=2`) was the top reported message-passing model on:
  - Wisconsin: `76.47 ± 5.19%`
  - Texas: `73.24 ± 5.98%`
  - Cornell: `62.43 ± 7.09%`
- Relative to plain GCN, reported gains were:
  - Wisconsin: `+24.31%`
  - Texas: `+16.21%`
  - Cornell: `+22.97%`
- Relative to standard one-delay Half-Hop, reported gains were:
  - Wisconsin: `+14.71%`
  - Texas: `+10.27%`
  - Cornell: `+10.00%`
  - Chameleon: `+5.00%`

**What was gained:**

- Evidence that dilation depth matters and that one inserted node is not necessarily the best path-delay design.
- A fairer comparison between physical path dilation and scalar ego-weighting.
- A concrete computational trade-off: more delays improve effective slowing but expand the graph substantially.

### Plan K — AdaHop adaptive path dilation

**Goal:** Obtain some of MultiDelay's benefit while reducing graph expansion by allocating delay budgets selectively.

**Completed work:**

- AdaHop architecture.
- Budgeted adaptive selection.
- Soft and discrete variants.
- Benchmark runner and report in [`experiments/results/adahop/ADAHOP_BENCHMARK_REPORT.md`](./experiments/results/adahop/ADAHOP_BENCHMARK_REPORT.md).

**Reported result:**

- At depth `L=4`, uniform MultiDelay was strongest on Wisconsin, Texas, and Cornell.
- On Chameleon, uniform MultiDelay expanded the graph to 74,379 nodes, described as `+3,166%` bloat.
- AdaHop at budget `B=0.25` reduced added-node footprint by approximately `73%` versus MultiDelay.
- AdaHop at `B=0.50` reduced expansion by approximately `50%`.
- The lower-budget variants generally recovered part, but not all, of the full MultiDelay accuracy.

**What was gained:**

- A concrete accuracy-versus-memory trade-off.
- A scalable direction for graphs where uniform multi-delay is too expensive.

### Plan L — Adaptive MultiDelay on ModelNet

**Goal:** Test uniform, random, and geometry/curvature-driven delay schedules on 3D object graphs.

**Completed work:**

- Adaptive point-cloud graph construction.
- Multi-delay insertion.
- Curvature and other delay modes.
- ModelNet10 optimization for Kaggle Free Tier.
- Fair protocol with EdgeConv, sparse-inverse delay, augmentation, and multi-seed logging.
- Disk-space and download-race fixes.

**Reported ModelNet10 result:**

| Mode | Test accuracy |
|---|---:|
| No delay baseline | `80.73%` |
| Random delays | `79.96%` |
| Uniform Half-Hop | `76.76%` |
| Curvature/AdaHop | `76.21%` |

**What was gained:**

- A clear domain-specific negative result.
- Evidence that graph dilation can starve the receptive field in moderate-depth point-cloud classification.
- A plausible mechanism for why curvature-based slowing can harm diagnostic geometric regions.
- Better Kaggle reliability through caching, sequential preparation, GPU assignment fixes, and low-memory operation.

## 6. Engineering and Infrastructure Gains

### 6.1 Experiment execution

The repository includes runners for:

- Individual supervised experiments.
- Full reproduction sweeps.
- Ablations.
- BGRL and GRACE.
- ModelNet mesh and point-cloud tasks.
- ShapeNet-Part segmentation.
- Two-GPU parallel execution.
- Kaggle-specific execution.

Existing scripts use resumable result files and progress logging in several experiment families.

### 6.2 Performance and reliability

The commit history records:

- Precomputing segmentation graphs in memory, reported as a 60x speedup.
- `--prepare-only` data preparation.
- Sequential dataset preparation to avoid download races.
- Stale-cache cleanup and a single base dataset cache to reduce Kaggle disk usage.
- Device assignment fixes for one-model-per-GPU execution.
- Pinning `typing_extensions==4.11.0` in Kaggle runners.
- Removing a module-shadowing conflict involving standard-library `ssl`.
- Reduced-worker/IPC safety for ModelNet.

**Gain:** The experiments became more practical on constrained Kaggle environments and less vulnerable to race conditions or disk exhaustion.

### 6.3 Testing

Tests cover:

- Half-Hop core behavior.
- GCN, GraphSAGE, and GAT.
- Half-Hop model wrappers.
- Dataset loading and splits.
- Training utilities.
- DropEdge.
- BGRL.
- Texas behavior.
- Point-cloud kNN construction.
- Boundary statistics.
- Selective Half-Hop.
- Adaptive ModelNet graph invariants.

The adaptive ModelNet tests specifically check:

- Slow-node count and masks.
- Interpolation values.
- Zero-delay edges.
- Edge index type.
- Feature construction without normals.

**Gain:** Core transformations and several experiment-specific invariants have automated regression coverage.

### 6.4 Documentation and reporting

The project contains:

- User-facing setup and commands in [`README.md`](./README.md).
- Source and paper alignment notes in [`SOURCE_NOTES.md`](./SOURCE_NOTES.md).
- Reproduction caveats in [`REPRODUCTION_NOTES.md`](./REPRODUCTION_NOTES.md).
- Extension design in [`extension_plan.md`](./extension_plan.md).
- Corrected research synthesis in [`EXTENSION_REPORT.md`](./EXTENSION_REPORT.md).
- A LaTeX final report in [`report/halfhop_final_report_exact.tex`](./report/halfhop_final_report_exact.tex).
- Per-experiment result reports and raw result files.

**Gain:** The repository preserves not only code and final numbers, but also rationale, controls, failed hypotheses, and protocol caveats.

## 7. Main Results at a Glance

| Area | Current status | Main gain |
|---|---|---|
| Core Half-Hop implementation | Complete | Reusable paper-aligned transform |
| GCN/SAGE/GAT backbones | Complete | Matched baseline and HH comparisons |
| Heterophilic reproduction | Implemented and benchmarked | Paper trends and protocol differences documented |
| Connectivity/init ablations | Complete | Shows which HH design choices matter |
| SSL prototypes | Implemented | Extends evaluation beyond supervised learning |
| Alpha/p sensitivity | Complete, protocol caveat | Shows small alpha/high coverage under one fixed protocol |
| DropEdge combination | Complete, mixed result | No consistent supervised synergy |
| Geometry Stage 0 | Passed modestly | Geometry predicts boundary edges above chance |
| Geometry Stage 1 | Reported | Positive on 3/4 categories, failure on Table |
| Selective HH Option A | Reported negative | Ego preservation explains much of the benefit |
| MultiDelay/depth study | Reported positive | More delay can help at greater depth |
| AdaHop | Implemented and benchmarked | Reduces graph expansion at partial accuracy |
| ModelNet10 adaptive delays | Complete, negative | Delays hurt this point-cloud setting |
| Tests | Broad coverage | Regression protection for core and extension logic |
| Kaggle execution | Hardened | Fewer race, device, memory, and disk failures |

## 8. Remaining Work and Open Questions

The repository contains substantial completed work, but the following items remain open or should be treated cautiously:

1. **Validate sensitivity selections correctly.** Re-select `alpha` and `p` using validation accuracy only, then evaluate test accuracy once.
2. **Make protocol comparisons consistent.** The reports note that the same nominal configuration can produce different values under different training scripts, depths, seeds, or selection rules.
3. **Run broader DropEdge sweeps.** Only a limited drop-rate setting was used for the reported combination study.
4. **Report per-arm uncertainty consistently.** Some depth-study tables do not include standard deviations for every arm.
5. **Separate ego preservation from physical dilation.** Continue matched comparisons against tunable self-loops, Lazy Walk, APPNP, GraphSAGE root weighting, and MLP controls.
6. **Reduce AdaHop expansion further.** The budgeted variants reduce memory but do not always recover full MultiDelay accuracy.
7. **Investigate the Table geometry failure.** High normal-angle edges may remain within the same semantic component; geometry alone is not a universal boundary detector.
8. **Clarify the final scientific claim.** The strongest defensible claim is conditional: Half-Hop-like path dilation can help under particular heterophilic/depth regimes, but its benefit is not universal and may be explained by simpler ego-preserving mechanisms.
9. **Keep generated artifacts manageable.** Archives, cached datasets, virtual environments, and generated outputs should be separated from source when preparing a clean distribution.
10. **Run the full validation suite before release.** This documentation-only change does not alter runtime code, so tests were not rerun as part of creating this report.

## 9. Chronological Milestones from Git History

The major development sequence was:

1. **2026-09-22 to 2026-09-24:** Initial implementation, training/logging fixes, BGRL fixes, official BGRL runner, and early Half-Hop experiments.
2. **2026-10-04:** Table 2/ablation reporting, ModelNet10 mesh graph classification, Amazon-ratings experiment, Kaggle race fixes, and LaTeX reporting.
3. **2026-10-05:** Extension planning, segmentation proxy work, reliable data hosting, edge-budget selection, and tests.
4. **2026-11-02 to 2026-11-06:** Extension branch consolidation, corrected extension report, Stage 0 geometry evaluation, full package/config/test additions, segmentation caching, multi-GPU runners, and Stage 1 category results.
5. **2026-11-07:** Option A selective Half-Hop and Option B/3D extension work, followed by the 3,960-run Option A report.
6. **2026-11-09:** AdaHop, Phase 1 controls, reproduction diagnostics, all-backbone reproduction, Kaggle dependency/runtime fixes, and fair-comparison preparation.
7. **2026-11-10:** Equal-budget comparison, depth-matched dilation, MultiDelay, AdaHop reporting, adaptive ModelNet work, ModelNet10 optimization, fair protocol, and Kaggle disk/download fixes.

## 10. Files to Read First

For a quick understanding of the project, read in this order:

1. [`README.md`](./README.md) — setup, commands, repository map, and experiment entry points.
2. [`halfhop/halfhop.py`](./halfhop/halfhop.py) — core transformation and selection semantics.
3. [`SOURCE_NOTES.md`](./SOURCE_NOTES.md) — mathematical definition, paper alignment, architectures, and reported paper values.
4. [`REPRODUCTION_NOTES.md`](./REPRODUCTION_NOTES.md) — reproduction caveats.
5. [`extension_plan.md`](./extension_plan.md) — hypotheses, gates, controls, and why the research plan changed.
6. [`EXTENSION_REPORT.md`](./EXTENSION_REPORT.md) — corrected synthesis of the extension findings.
7. [`experiments/results/fair/STEP2_REPORT.md`](./experiments/results/fair/STEP2_REPORT.md) — Lazy Walk and ego-preservation comparison.
8. [`experiments/results/adahop/ADAHOP_BENCHMARK_REPORT.md`](./experiments/results/adahop/ADAHOP_BENCHMARK_REPORT.md) — adaptive dilation and expansion trade-offs.
9. [`experiments/results/geometry_proxy/stage0_report.md`](./experiments/results/geometry_proxy/stage0_report.md) — Stage 0 geometry evidence.
10. [`experiments/results/modelnet10_report.md`](./experiments/results/modelnet10_report.md) — ModelNet10 adaptive-delay result and failure analysis.

## 11. Final Assessment

The repository is beyond a prototype: it contains a functioning research codebase, multiple controlled experiment families, automated tests, reproducibility notes, result artifacts, and a corrected scientific narrative.

The primary gain so far is not a single universally positive accuracy number. It is the progressively sharper explanation of **when** slowing message passing helps:

- It can help heterophilic graphs by reducing harmful neighbor mixing.
- Its apparent benefit can often be reproduced by simpler ego-preserving operators.
- More delay can help deep message passing, but costs graph memory.
- Selective or geometry-aware delay is promising but domain-dependent.
- The same operation can hurt point-cloud classification when it prevents global shape information from propagating.

That conditional conclusion is currently better supported by the repository than a broad claim that Half-Hop is always superior.
