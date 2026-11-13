# SOURCE_NOTES.md
# Half-Hop: Foundational Source Analysis and Literature Notes

## 1. Citation and Provenance

```bibtex
@inproceedings{azabou2023half,
  title     = {Half-Hop: A graph upsampling approach for slowing down message passing},
  author    = {Azabou, Mehdi and Ganesh, Venkataramana and Thakoor, Shantanu and Lin, Chi-Heng
               and Sathidevi, Lakshmi and Liu, Ran and Valko, Michal and Veli{\v{c}}kovi{\'c}, Petar and Dyer, Eva L},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research (PMLR)},
  volume    = {202},
  pages     = {1342--1360},
  year      = {2023},
  publisher = {PMLR}
}
```

- **Official Repository:** [https://github.com/nerdslab/halfhop](https://github.com/nerdslab/halfhop)  
- **Published Proceedings (PMLR 202):** [https://proceedings.mlr.press/v202/azabou23a.html](https://proceedings.mlr.press/v202/azabou23a.html)
- **OpenReview Forum:** [https://openreview.net/forum?id=lXczFIwQkv](https://openreview.net/forum?id=lXczFIwQkv)
- **Project Context:** Foundation reference for 2D reproduction and 3D geometric extension.

---

## 2. Core Problem: Over-Smoothing and Over-Squashing in Heterophilic Graphs

In standard Graph Neural Networks (GNNs), message passing iteratively updates node representations by aggregating features across $1$-hop spatial neighborhoods:
$$\mathbf{h}_v^{(l+1)} = \sigma \left( \mathbf{W}^{(l)} \cdot \text{AGG} \left( \left\{ \mathbf{h}_u^{(l)} : u \in \tilde{\mathcal{N}}(v) \right\} \right) \right)$$

On **heterophilic graphs**—where adjacent nodes predominantly belong to different semantic classes ($h \ll 0.5$)—standard message passing exhibits severe pathologies:
1. **Feature Bleeding / Premature Over-Smoothing:** High-frequency information and distinct local class identities are prematurely washed out into homogeneous cluster representations in early layers.
2. **Over-Squashing:** Information from exponentially growing neighborhoods is compressed into fixed-size node feature vectors.
3. **Trade-off Dilemma:** Shallow networks ($L=2$) avoid over-smoothing but fail to capture multi-hop structural context, while deep networks ($L \ge 4$) capture distant context but destroy discriminatory node features.

---

## 3. Mathematical Formulation of Half-Hop

Half-Hop resolves this dilemma by **upsampling the graph topology**—subdividing edges to insert virtual "slow nodes" that mechanically halve the speed of message propagation.

### 3.1 Single Edge Transformation (Equation 1)
For an original directed edge $e_{ij} = (v_i \to v_j)$ with $v_i \neq v_j$:
1. A virtual slow node $\nu_k$ is added to the node set:
   $$\mathcal{V}' = \mathcal{V} \cup \{\nu_k\}$$
2. The original direct edge $e_{ij}$ is replaced by a triplet of directed edges:
   $$\mathcal{E}' = (\mathcal{E} \setminus \{e_{ij}\}) \cup \{e_{i \to k}, e_{j \to k}, e_{k \to j}\}$$

```
Original Edge:      v_i ───────────────> v_j

Half-Hop Rewiring:  v_i ──────> ν_k <───── v_j
                                 │
                                 └──────> v_j
                    (Compact notation: v_i → ν_k ↔ v_j)
```

### 3.2 Information Flow Mechanics
- Both source node $v_i$ and target node $v_j$ send messages to the slow node $\nu_k$.
- The slow node $\nu_k$ passes information **strictly toward the target** $v_j$, preserving the causal directionality of the original edge.
- At layer $l=1$, target node $v_j$ receives only its own self-message and feedback from $\nu_k$, while $v_i$'s information reaches $\nu_k$ but has **not yet arrived at $v_j$**.
- At layer $l=2$, the message from $v_i$ is transmitted through $\nu_k$ into $v_j$. 
- Result: Propagation from $v_i$ to $v_j$ requires **2 GNN layers** instead of 1, effectively realizing a **0.5-hop spatial rate of expansion per layer**.

### 3.3 Slow Node Feature Initialization
The slow node $\nu_k$ bridging source $v_i$ and target $v_j$ is initialized via convex interpolation:
$$\mathbf{x}_k = \alpha \, \mathbf{x}_i + (1 - \alpha) \, \mathbf{x}_j, \quad \alpha \in [0, 1]$$
- $\alpha = 0$: Slow node begins with pure target features ($\mathbf{x}_j$).
- $\alpha = 1$: Slow node begins with pure source features ($\mathbf{x}_i$).
- $\alpha = 0.5$: Balanced symmetric initialization (standard default).

### 3.4 Stochastic Node-Level Half-Hop ($p < 1$)
Rather than edge-level Bernoulli sampling, Half-Hop executes **node-level sampling**:
1. For each node $v_j \in \mathcal{V}$, sample an independent Bernoulli variable $b_j \sim \text{Bernoulli}(p)$.
2. If $b_j = 1$, **all incoming edges** pointing into $v_j$ are transformed into half-hop triplets.
3. If $b_j = 0$, all incoming edges of $v_j$ remain unaltered direct $1$-hop connections.

*Rationale:* Node-level sampling prevents asymmetric degree distortion where a target node receives a chaotic mixture of direct and delayed inputs from identical hops.

---

## 4. Connectivity Ablation Variants (Appendix B.1)

The paper investigates alternative edge connectivity configurations for slow node $\nu_k$:

| Variant | Edge Set Added | Effective Topology | Characteristic |
| :--- | :--- | :--- | :--- |
| **Proposed (Default)** | $\{e_{i \to k}, e_{j \to k}, e_{k \to j}\}$ | $v_i \to \nu_k \leftrightarrow v_j$ | Target feedbacks into $\nu_k$; preserves directed flow. Best performance. |
| **Simple Subdivision** | $\{e_{i \to k}, e_{k \to j}\}$ | $v_i \to \nu_k \to v_j$ | Pure 2-hop pipeline without target feedback. |
| **Undirected / Bidirectional** | $\{e_{i \to k}, e_{k \to i}, e_{j \to k}, e_{k \to j}\}$ | $v_i \leftrightarrow \nu_k \leftrightarrow v_j$ | Introduces reverse paths; causes premature information leakage into $v_i$. |
| **Target Feedback Only** | $\{e_{j \to k}, e_{k \to j}\}$ | $\nu_k \leftrightarrow v_j$ | Disconnects source; serves as memory buffer for target only. |

---

## 5. Self-Loops and Prediction Masking

### 5.1 Self-Loop Treatment
Self-loops $e_{ii} = (v_i \to v_i)$ are **explicitly excluded** from the Half-Hop edge subdivision:
- If self-loops were subdivided, a node would experience delayed access to its own state.
- Standard self-loops are retained or appended after the Half-Hop graph rewiring.

### 5.2 Decoupled Prediction Head
Virtual slow nodes exist exclusively to mediate spatial message flow:
- They are **never included** in the prediction head, cross-entropy loss computation, or accuracy calculation.
- An explicit boolean mask $\mathbf{m}_{\text{slow}} \in \{0, 1\}^{|\mathcal{V}'|}$ tracks node provenance:
  $$\mathbf{m}_{\text{slow}}[v] = \begin{cases} 1 & \text{if } v \in \mathcal{V}_{\text{slow}} \\ 0 & \text{if } v \in \mathcal{V}_{\text{orig}} \end{cases}$$
- Final classification logits are filtered via:
  $$\hat{\mathbf{Y}} = \text{GNN}(\mathbf{X}', \mathcal{E}')[\neg \mathbf{m}_{\text{slow}}]$$

---

## 6. Official Published Benchmark Datasets

The published evaluation covers six canonical heterophilic graphs from Pei et al. (2020) with 10 standard splits:

| Dataset | Nodes ($|\mathcal{V}|$) | Edges ($|\mathcal{E}|$) | Features ($D$) | Classes ($C$) | Heterophily ($h$) | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Texas** | 183 | 309 | 1,703 | 5 | 0.11 | WebKB university webpage network |
| **Wisconsin** | 251 | 509 | 1,703 | 5 | 0.21 | WebKB university webpage network |
| **Cornell** | 183 | 295 | 1,703 | 5 | 0.05 | WebKB university webpage network |
| **Chameleon** | 2,277 | 36,101 | 2,325 | 5 | 0.23 | Wikipedia topic page network |
| **Squirrel** | 5,201 | 217,073 | 2,089 | 5 | 0.22 | Wikipedia topic page network |
| **Actor** | 7,600 | 33,544 | 931 | 5 | 0.22 | Film actor co-occurrence network |

*Edge homophily metric:* $h = \frac{|\{(u, v) \in \mathcal{E} : y_u = y_v\}|}{|\mathcal{E}|}$

---

## 7. Published Hyperparameters (ICML 2023 Table 5)

Optimal hyperparameters reported in the paper appendix for Half-Hop models:

| Dataset | Model | Learning Rate | Weight Decay | Depth | Hidden Dim | Dropout | $\alpha$ | $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Texas** | HH-GCN | 0.0291 | 0.0096 | 2 | 64 | 0.8058 | 0.0043 | 0.9526 |
| **Texas** | HH-SAGE | 0.0170 | 0.0053 | 2 | 64 | 0.1967 | 0.9397 | 0.7140 |
| **Texas** | HH-GAT | 0.0328 | 0.0066 | 2 | 32 | 0.1288 | 0.0902 | 0.9841 |
| **Wisconsin** | HH-GCN | 0.0105 | 0.0002 | 3 | 128 | 0.6612 | 0.9937 | 0.7140 |
| **Wisconsin** | HH-SAGE | 0.0202 | 0.0042 | 3 | 64 | 0.3462 | 0.0100 | 0.6177 |
| **Wisconsin** | HH-GAT | 0.0539 | 0.0068 | 3 | 16 | 0.2141 | 0.0026 | 0.9797 |
| **Cornell** | HH-GCN | 0.0505 | 0.0055 | 2 | 32 | 0.4123 | 0.0145 | 0.9660 |
| **Cornell** | HH-SAGE | 0.0697 | 0.0018 | 2 | 64 | 0.0697 | 0.8807 | 0.5660 |
| **Cornell** | HH-GAT | 0.0572 | 0.0070 | 2 | 64 | 0.0572 | 0.0710 | 0.9979 |
| **Chameleon** | HH-GCN | 0.0318 | 0.0057 | 2 | 128 | 0.8040 | 0.0510 | 0.9986 |
| **Chameleon** | HH-SAGE | 0.0225 | 0.0001 | 2 | 32 | 0.7175 | 0.9834 | 0.6226 |
| **Chameleon** | HH-GAT | 0.0012 | 0.0008 | 3 | 64 | 0.0439 | 0.9766 | 0.9386 |
| **Squirrel** | HH-GCN | 0.0053 | 0.0001 | 3 | 128 | 0.2455 | 0.0145 | 0.8257 |
| **Squirrel** | HH-SAGE | 0.0296 | 0.0001 | 2 | 128 | 0.8668 | 0.9474 | 0.5198 |
| **Squirrel** | HH-GAT | 0.0027 | 0.0001 | 3 | 64 | 0.5131 | 0.9277 | 0.1549 |
| **Actor** | HH-GCN | 0.0313 | 0.0087 | 3 | 64 | 0.5511 | 0.0369 | 0.5466 |
| **Actor** | HH-SAGE | 0.0133 | 0.0090 | 3 | 32 | 0.3737 | 0.0116 | 0.8368 |
| **Actor** | HH-GAT | 0.0009 | 0.0001 | 3 | 128 | 0.8708 | 0.0549 | 0.9594 |

---

## 8. Published Benchmark Results (ICML 2023 Table 2)

Mean classification test accuracy ($\% \pm \text{std}$) across 10 fixed splits as reported in PMLR 202:

| Dataset | GCN (Base)$^\dagger$ | HH-GCN | GraphSAGE (Base)$^\dagger$ | HH-GraphSAGE | GAT (Base)$^\dagger$ | HH-GAT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Texas** | $55.14 \pm 5.16$ | $\mathbf{71.89 \pm 3.46}$ | $82.43 \pm 6.14$ | $\mathbf{85.95 \pm 6.42}$ | $52.16 \pm 6.63$ | $\mathbf{80.54 \pm 4.80}$ |
| **Wisconsin** | $51.76 \pm 3.06$ | $\mathbf{79.80 \pm 4.30}$ | $81.18 \pm 5.56$ | $\mathbf{85.88 \pm 3.99}$ | $49.41 \pm 4.09$ | $\mathbf{83.53 \pm 3.84}$ |
| **Cornell** | $60.54 \pm 5.30$ | $\mathbf{63.24 \pm 5.43}$ | $75.95 \pm 5.01$ | $74.60 \pm 6.06$ | $61.89 \pm 5.05$ | $\mathbf{72.70 \pm 4.26}$ |
| **Chameleon** | $38.44 \pm 1.92$ | $\mathbf{60.24 \pm 1.93}$ | $58.73 \pm 1.68$ | $\mathbf{62.98 \pm 3.35}$ | $48.36 \pm 1.58$ | $\mathbf{61.12 \pm 1.83}$ |
| **Squirrel** | $31.52 \pm 0.71$ | $\mathbf{47.19 \pm 1.21}$ | $41.61 \pm 0.74$ | $\mathbf{45.25 \pm 1.52}$ | $36.77 \pm 1.68$ | $\mathbf{46.35 \pm 1.86}$ |
| **Actor** | $27.32 \pm 1.10$ | $\mathbf{35.12 \pm 1.06}$ | $34.23 \pm 0.99$ | $\mathbf{36.82 \pm 0.77}$ | $27.44 \pm 0.89$ | $\mathbf{36.70 \pm 0.92}$ |

$^\dagger$*Baselines cited in paper from Yan et al. (2021) "Two sides of the same coin: Heterophily and oversmoothing in GNNs".*

---

## 9. Paper vs. Official Repository Discrepancies

Through line-by-line inspection of `nerdslab/halfhop`, several subtle operational discrepancies were identified:

1. **Interpolation Notation:**  
   The paper text states $\mathbf{x}_k = (1 - \alpha)\mathbf{x}_j + \alpha \mathbf{x}_i$, whereas the code computes `alpha * x[src] + (1 - alpha) * x[dst]`. These are mathematically identical under index exchange, but required careful verification during reproduction.
2. **Boundary Condition ($p = 1$):**  
   In the official repository, when $p = 1$, the retained edge index is set to `None`. In standard PyTorch Geometric tensor concatenation, this caused unhandled `TypeError` exceptions unless sanitized with `None`-safe edge filtering.
3. **In-place Operations:**  
   The official code leveraged PyTorch in-place tensor mutations (`mul_`, `add_`) on shared feature matrices, which interfered with autograd backward passes when computing multi-layer gradient hooks. Safe out-of-place equivalents were substituted.

---

## 10. Direct Architectural Lineage to Project Deliverables
- **Reproduction:** Validated in [IMPLEMENTED.md](file:///d:/ai_course/Project/halfhop-project/IMPLEMENTED.md) across all 6 heterophilic benchmarks.
- **3D Extension:** Extended in [EXTENSION.md](file:///d:/ai_course/Project/halfhop-project/EXTENSION.md) to dual-gated geometric crease throttling on ShapeNet-Part.
- **Manuscript:** Synthesized in [report/ieee_reproduction_and_extension.pdf](file:///d:/ai_course/Project/halfhop-project/report/ieee_reproduction_and_extension.pdf).
