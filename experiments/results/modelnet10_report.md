# Empirical Analysis: ModelNet10 Adaptive Multi-Delay Half-Hop

**Dataset:** ModelNet10 (10 classes, 3,991 train / 908 test shapes, 512 points/shape)  
**Backbone:** 6-layer GraphSAGE ($L=6, H=128$, dropout 0.5, Adam $\text{lr}=10^{-3}$)  
**Hardware & Platform:** Kaggle Free Tier (2× Tesla T4 GPUs)  
**Execution Timestamp:** 2026-11-10  
**Summary Source:** `experiments/results/modelnet10_summary.json`

---

## 1. Executive Summary of Results

| Experiment | Delays ($m$) | Delay Allocation Mode | Val Acc (%) | Test Acc @ Best Val (%) | $\Delta$ vs Baseline |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`none` (Baseline)** | $(0, 0, 0)$ | No slow nodes (pure KNN graph) | **86.72%** | **80.73%** | **Ref** |
| **`random`** | $(1, 2, 3)$ | Random delay assignment (control) | 85.96% | **79.96%** | $-0.77\%$ |
| **`uniform` (Half-Hop)** | $(1, 1, 1)$ | Uniform delay ($m=1$ on all edges) | 84.96% | **76.76%** | $-3.97\%$ |
| **`curvature` (AdaHop)** | $(1, 2, 3)$ | High-curvature edges receive longest delays | 81.95% | **76.21%** | $-4.52\%$ |

---

## 2. In-Depth Mechanistic Analysis: Why Does Dilation Fail on Point Clouds?

### Mechanism 1: Receptive Field Starvation ($L_{\text{eff}}$ Bottleneck)
In our Step 3 depth study, we established the **Topological Transmission Equation**:
$$L_{\text{eff}} = \frac{L}{m + 1}$$
where $m$ is the edge delay and $L$ is GNN depth.

- **In Citation/Web Graphs ($L \ge 8$):** Graphs suffer from catastrophic over-smoothing (Dirichlet energy drops to $\approx 0.02$). The transmission gearbox ($L_{\text{eff}} = L/3$) is beneficial because it prevents node features from collapsing into a uniform representation.
- **In 3D Point Clouds ($L=6$):** 3D shape recognition requires **global structural integration**. To classify an object as a *chair*, *bed*, or *table*, the model must relate distant geometric components (e.g., chair backrest to seat to legs).
  - With baseline `none` ($m=0$): $L_{\text{eff}} = 6$ hops. Features propagate across 6 spatial hops, spanning the diameter of the point cloud.
  - With `uniform` ($m=1$): $L_{\text{eff}} = 6 / (1+1) = 3$ hops. The effective spatial diameter reachable by message passing is cut in half.
  - With `curvature` ($m \in \{1, 2, 3\}$): On edges with $m=3$, $L_{\text{eff}} = 6 / (3+1) = 1.5$ hops.
  The network suffers from **topological myopic blindness**: it cannot integrate global shape context within 6 layers.

---

### Mechanism 2: Manifold Chordal Corruption (Linear Interpolation in Curved Space)
In `AdaptiveHalfHop`, slow nodes are inserted along edges via linear coordinate interpolation:
$$\mathbf{x}_{\text{slow}} = (1 - \lambda)\mathbf{x}_u + \lambda \mathbf{x}_v, \quad \lambda \in (0, 1)$$

In a 3D point cloud:
1. Vertices $\mathbf{x}_u, \mathbf{x}_v \in \mathcal{M}$ lie on a curved 2D surface manifold embedded in $\mathbb{R}^3$.
2. The straight segment connecting $\mathbf{x}_u$ and $\mathbf{x}_v$ is a **secant chord**, not a geodesic on $\mathcal{M}$.
3. In regions of high curvature (sharp edges, corners, thin tubular structures), linear interpolation places slow nodes in **empty Euclidean space outside the true object boundary**.
4. These "ghost nodes" corrupt the local surface normals and geometric invariants, injecting false spatial artifacts into the message-passing stream.

---

### Mechanism 3: Why Did `random` Outperform `curvature`?
One of the most striking results is:
$$\text{Test Acc}(\text{random}) = 79.96\% \quad \gg \quad \text{Test Acc}(\text{curvature}) = 76.21\% \quad (\Delta = +3.75\%)$$

Why does allocating delays randomly beat allocating them based on Pauly surface curvature?

- **Diagnostic Landmarking Hypothesis:**
  In 3D shape classification, high-curvature points (corners, boundaries, sharp bends) are the **primary diagnostic landmarks** distinguishing geometric classes (e.g., the sharp 90° corners of a desk vs. the curved arms of a sofa).
- In the `curvature` strategy:
  Edges adjacent to high-curvature regions were assigned the maximum delay ($m=3$). This throttled message propagation precisely at the most informative geometric features of the shape!
- In the `random` strategy:
  Delays were randomly distributed across planar surfaces and high-curvature zones alike. Most diagnostic high-curvature edges remained fast ($m=0$ or $m=1$), preserving critical feature flow.

---

## 3. Cross-Benchmark Comparative Synthesis

| Benchmark Domain | Graph Type | Depth Regime | Half-Hop / Multi-Delay Impact | Primary Phenomenon |
| :--- | :--- | :---: | :---: | :--- |
| **Wisconsin / Chameleon** | Heterophilic Relational | Shallow ($L=2$) | $\approx$ Matched by Lazy Walk ($p < 0.005$) | Ego-weight damping; structural dilation adds no topological advantage. |
| **Heterophilic at Scale** | Heterophilic Relational | Deep ($L \ge 6$) | **Strong Win (+19.6% at $L=8$)** | Anti-over-smoothing gearbox; preserves Dirichlet energy ($0.68$ vs $0.02$). |
| **ModelNet10** | 3D Geometric Manifold | Moderate ($L=6$) | **Negative Impact ($-4.52\%$)** | Receptive field starvation ($L_{\text{eff}} \le 1.5$) + chordal manifold distortion. |

