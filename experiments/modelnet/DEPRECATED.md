# DEPRECATED — ModelNet10 Classification

> [!WARNING]
> This module is **superseded**. The project now focuses exclusively on
> **Geometry-Aware Half-Hop for 3D Part Segmentation** (ShapeNet Part).
> See `experiments/shapenet_partseg/` for the current pipeline.

## Why deprecated

Random / uniform Half-Hop applied to 3D point-cloud classification:

1. **Geometry distortion** — slow nodes inserted on smooth interior surfaces
   break local 3D shape features.
2. **Receptive-field starvation** — half-hopping interior edges delays message
   propagation inside a part, starving nodes of their local context.
3. **Wrong task** — whole-object classification cannot reveal whether boundary
   slow-node placement improves part boundary sharpness.

## What to use instead

| Old | New |
|---|---|
| `run_fair.py` / `run_adaptive.py` | `experiments/shapenet_partseg/run_partseg.py` |
| `AdaptiveHalfHop` (all-edge curvature) | `halfhop/boundary_halfhop.py` (boundary-only) |
| ModelNet10 accuracy | ShapeNet Part mIoU (per-category + instance) |

## Code preserved for reference

The geometry utilities in `adaptive_halfhop.py` (`knn_edge_index`,
`surface_variation`, `assign_delays`, `insert_slow_chains`) were reused in
`halfhop/geometry_boundary.py` and `halfhop/boundary_halfhop.py`.

Do not delete this directory — it is retained for git history and test coverage.
