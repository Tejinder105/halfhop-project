from .halfhop import HalfHop
from .gcn import GCN
from .graphsage import GraphSAGE
from .gat import GAT
from .hh_gcn import HHGCN
from .hh_graphsage import HHGraphSAGE
from .hh_gat import HHGAT
from .datasets import load_dataset
from .splits import get_split, make_random_splits
from .training import run_training_loop
from .evaluation import format_results
from .reproducibility import set_seed

# Phase 3: AdaHop components
from .edge_alpha import EdgeAlpha
from .soft_halfhop import SoftHalfHop, SoftGate
from .stateful_halfhop import StatefulHalfHop
from .multidelay_halfhop import MultiDelayHalfHop
from .adahop import (
    AdaHopGCN,
    AdaHopSAGE,
    AdaHopGAT,
    SoftAdaHopGCN,
    StatefulHHGCN,
    MultiDelayGCN,
)

__all__ = [
    # Original
    'HalfHop',
    'GCN', 'GraphSAGE', 'GAT',
    'HHGCN', 'HHGraphSAGE', 'HHGAT',
    'load_dataset',
    'get_split', 'make_random_splits',
    'run_training_loop',
    'format_results',
    'set_seed',
    # AdaHop Phase 3
    'EdgeAlpha',
    'SoftHalfHop', 'SoftGate',
    'StatefulHalfHop',
    'MultiDelayHalfHop',
    'AdaHopGCN', 'AdaHopSAGE', 'AdaHopGAT',
    'SoftAdaHopGCN',
    'StatefulHHGCN',
    'MultiDelayGCN',
]
