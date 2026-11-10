import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_ch, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU()
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        h = torch.cat([x[dst], x[src] - x[dst]], dim=-1)
        msg = self.mlp(h)
        agg = torch.zeros(x.size(0), msg.size(-1), device=x.device, dtype=msg.dtype)
        idx = dst.unsqueeze(1).expand_as(msg)
        agg.scatter_reduce_(0, idx, msg, reduce='amax', include_self=True)
        return agg


class EdgeConvSegModel(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, depth=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(EdgeConvLayer(in_channels, hidden))
        for _ in range(depth - 1):
            self.layers.append(EdgeConvLayer(hidden, hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden * depth, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        feats = []
        for layer in self.layers:
            x = layer(x, edge_index)
            feats.append(x)
        out = self.head(torch.cat(feats, dim=-1))
        mask = getattr(data, 'slow_node_mask', None)
        if mask is not None:
            out = out[~mask]
        return out
