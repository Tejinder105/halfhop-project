"""Geometry-Aware Boundary Half-Hop for 3D Point Cloud Part Segmentation."""

from .halfhop import HalfHop
from .boundary_halfhop import BoundaryHalfHop
from .selective_halfhop import half_hop, select_edges, topk_mask
from .geometry_boundary import (
    inter_part_boundary_edges,
    boundary_edge_mask,
    compute_normals,
    surface_variation,
    normal_disagreement,
    geometry_boundary_score,
)
from .reproducibility import set_seed

__all__ = [
    "HalfHop",
    "BoundaryHalfHop",
    "half_hop",
    "select_edges",
    "topk_mask",
    "inter_part_boundary_edges",
    "boundary_edge_mask",
    "compute_normals",
    "surface_variation",
    "normal_disagreement",
    "geometry_boundary_score",
    "set_seed",
]
