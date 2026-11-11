import torch
from halfhop.selective_halfhop import half_hop
from halfhop.geometry_boundary import inter_part_boundary_edges, boundary_edge_mask


def _knn(pos, k):
    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float('inf'))
    nb = dist.topk(k, largest=False, dim=1).indices
    dst = torch.arange(pos.size(0), device=pos.device).repeat_interleave(k)
    return torch.stack([nb.reshape(-1), dst], dim=0).long()


class BoundaryHalfHop:
    """Inserts Half-Hop slow nodes strictly on inter-part boundary edges.

    Training (data.y present, use_labels=True): boundary = edges crossing GT part labels.
    Inference (data.y absent or use_labels=False): boundary = top geometry_boundary_score edges.
    Interior edges are never slowed.
    """

    def __init__(self, knn=16, alpha=0.4, use_labels=True, curv_quantile=0.85):
        self.knn = knn
        self.alpha = alpha
        self.use_labels = use_labels
        self.curv_quantile = curv_quantile

    def __call__(self, data):
        data = data.clone()
        pos = data.pos.float() if data.pos is not None else None

        # Resolve edge_index
        edge_index = getattr(data, 'base_edge_index', None)
        if edge_index is None:
            edge_index = getattr(data, 'edge_index', None)
        if edge_index is None:
            if pos is None:
                return data
            edge_index = _knn(pos, self.knn)

        # Resolve features x
        x = getattr(data, 'x', None)
        if x is None:
            normal = getattr(data, 'normal', None)
            x = torch.cat([pos, normal.float()], dim=1) if normal is not None else pos
        x = x.float()

        # Determine boundary mask
        y = getattr(data, 'y', None)
        if self.use_labels and y is not None:
            bnd_mask = inter_part_boundary_edges(edge_index, y.view(-1))
        else:
            normal = getattr(data, 'normal', None)
            if normal is None and x is not None and x.size(1) >= 6:
                normal = x[:, 3:6]
            bnd_mask = boundary_edge_mask(
                pos, edge_index,
                normal.float() if normal is not None else None,
                self.knn, self.curv_quantile
            )

        if not bnd_mask.any():
            # No boundary edges — return original graph with features set
            data.x = x
            data.edge_index = edge_index
            data.slow_node_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            data.num_nodes = x.size(0)
            return data

        # Apply half_hop only on boundary edges
        x_new, ei_new, slow_mask = half_hop(x, edge_index, bnd_mask, self.alpha)
        data.x = x_new
        data.edge_index = ei_new
        data.slow_node_mask = slow_mask
        data.num_nodes = x_new.size(0)
        return data
