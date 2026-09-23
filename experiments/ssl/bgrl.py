"""
BGRL with Half-Hop augmentation.

Based on:
    Thakoor et al., "Large-Scale Representation Learning on Graphs
    via Bootstrapping", ICLR 2022.

    Azabou et al., "Half-Hop: A graph upsampling approach for
    slowing down message passing", ICML 2023.

Paper-specific Amazon Computers settings:
    Half-Hop p:
        view 1 = 0.75
        view 2 = 0.75

    Half-Hop alpha:
        view 1 = 0.50
        view 2 = 0.50

    Feature masking:
        view 1 = 0.20
        view 2 = 0.10

    Edge masking:
        view 1 = 0.50
        view 2 = 0.40

SSL linear evaluation:
    train = 10%
    validation = 10%
    test = 80%
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tqdm import trange

from halfhop.datasets import load_dataset
from halfhop.halfhop import HalfHop
from halfhop.reproducibility import set_seed


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    os.environ.get(
        "HALFHOP_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu"
    )
)


# ============================================================
# GCN Encoder
# ============================================================

class GCNEncoder(nn.Module):
    """
    2-layer GCN encoder used by BGRL.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 512,
        out_channels: int = 256,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_channels
        )

        self.prelu1 = nn.PReLU(
            hidden_channels,
            init=0.25
        )

        self.conv2 = GCNConv(
            hidden_channels,
            out_channels
        )

    def forward(
        self,
        x,
        edge_index
    ):
        x = self.conv1(
            x,
            edge_index
        )

        x = self.bn1(x)

        x = self.prelu1(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# Predictor
# ============================================================

class Predictor(nn.Module):
    """
    BGRL predictor MLP.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 512,
        out_channels: int = 256,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            in_channels,
            hidden_channels
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_channels
        )

        self.prelu1 = nn.PReLU(
            hidden_channels,
            init=0.25
        )

        self.fc2 = nn.Linear(
            hidden_channels,
            out_channels
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.bn1(x)

        x = self.prelu1(x)

        x = self.fc2(x)

        return x


# ============================================================
# Half-Hop augmentation
# ============================================================

def halfhop_augment(
    data: Data,
    alpha: float,
    p: float,
):
    """
    Apply Half-Hop.

    Returns a graph containing:
        - original nodes
        - slow nodes
        - half-hopped edges
    """

    hh = HalfHop(
        alpha=alpha,
        p=p,
        inplace=False
    )

    return hh(data)


# ============================================================
# Feature masking
# ============================================================

def feature_mask(
    data: Data,
    drop_rate: float,
):
    """
    Randomly mask node features.

    The graph structure is unchanged.
    """

    data = copy.deepcopy(data)

    if drop_rate <= 0:
        return data

    keep_probability = 1.0 - drop_rate

    mask = (
        torch.rand_like(
            data.x
        ) < keep_probability
    )

    data.x = data.x * mask

    return data


# ============================================================
# Edge masking
# ============================================================

def edge_dropout(
    data: Data,
    drop_rate: float,
):
    """
    Randomly remove edges.
    """

    data = copy.deepcopy(data)

    if drop_rate <= 0:
        return data

    num_edges = data.edge_index.size(1)

    keep_probability = 1.0 - drop_rate

    mask = (
        torch.rand(
            num_edges,
            device=data.edge_index.device
        )
        < keep_probability
    )

    data.edge_index = data.edge_index[:, mask]

    return data


# ============================================================
# Complete BGRL view construction
# ============================================================

def make_view(
    data: Data,
    hh_alpha: float,
    hh_p: float,
    feature_drop: float,
    edge_drop: float,
):
    """
    Construct one BGRL graph view.

    Order:
        1. Half-Hop
        2. Feature masking
        3. Edge masking

    Returns:
        augmented graph
        mask identifying original nodes
    """

    # --------------------------------------------------------
    # Half-Hop
    # --------------------------------------------------------

    aug = halfhop_augment(
        data,
        alpha=hh_alpha,
        p=hh_p
    )

    # --------------------------------------------------------
    # Identify original nodes
    # --------------------------------------------------------

    if hasattr(
        aug,
        "slow_node_mask"
    ):
        original_mask = ~aug.slow_node_mask
    else:
        # Fallback if the HalfHop object does not expose
        # slow_node_mask.
        original_mask = torch.ones(
            aug.num_nodes,
            dtype=torch.bool,
            device=aug.x.device
        )

    # --------------------------------------------------------
    # Feature masking
    # --------------------------------------------------------

    aug = feature_mask(
        aug,
        drop_rate=feature_drop
    )

    # --------------------------------------------------------
    # Edge masking
    # --------------------------------------------------------

    aug = edge_dropout(
        aug,
        drop_rate=edge_drop
    )

    return aug, original_mask


# ============================================================
# BGRL
# ============================================================

class BGRL(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        online_device,
        target_device,
    ):
        super().__init__()

        self.online_device = online_device
        self.target_device = target_device

        # ----------------------------------------------------
        # Online network
        # ----------------------------------------------------

        self.online_encoder = encoder.to(
            online_device
        )

        self.predictor = predictor.to(
            online_device
        )

        # ----------------------------------------------------
        # Target network
        # ----------------------------------------------------

        self.target_encoder = copy.deepcopy(
            encoder
        ).to(
            target_device
        )

        for param in self.target_encoder.parameters():
            param.requires_grad = False

    # --------------------------------------------------------
    # EMA target update
    # --------------------------------------------------------

    @torch.no_grad()
    def update_target(
        self,
        tau=0.99
    ):

        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):

            target_param.data.mul_(tau)

            target_param.data.add_(
                online_param.data.to(
                    self.target_device
                ),
                alpha=1.0 - tau
            )

    # --------------------------------------------------------
    # Training step
    # --------------------------------------------------------

    def train_step(
        self,
        x1,
        edge1,
        mask1,
        x2,
        edge2,
        mask2,
        use_amp=True,
    ):

        device_type = (
            "cuda"
            if self.online_device.type == "cuda"
            else "cpu"
        )

        amp_enabled = (
            use_amp
            and device_type == "cuda"
        )

        # ====================================================
        # TARGET NETWORK
        # ====================================================

        with torch.no_grad():

            with torch.amp.autocast(
                device_type=device_type,
                enabled=amp_enabled
            ):

                # Target view 1
                t1 = self.target_encoder(
                    x1.to(self.target_device),
                    edge1.to(self.target_device)
                )

                # Keep ONLY original nodes
                t1 = t1[
                    mask1.to(self.target_device)
                ]

                # Target view 2
                t2 = self.target_encoder(
                    x2.to(self.target_device),
                    edge2.to(self.target_device)
                )

                # Keep ONLY original nodes
                t2 = t2[
                    mask2.to(self.target_device)
                ]

                t1 = t1.to(
                    self.online_device
                )

                t2 = t2.to(
                    self.online_device
                )

        # ====================================================
        # ONLINE VIEW 1
        # ====================================================

        with torch.amp.autocast(
            device_type=device_type,
            enabled=amp_enabled
        ):

            z1 = self.online_encoder(
                x1.to(self.online_device),
                edge1.to(self.online_device)
            )

            z1 = z1[
                mask1.to(self.online_device)
            ]

            q1 = self.predictor(z1)

            loss1 = _cosine_loss(
                q1,
                t2
            )

        loss1.backward()

        loss1_value = loss1.item()

        del z1
        del q1
        del loss1

        # ====================================================
        # ONLINE VIEW 2
        # ====================================================

        with torch.amp.autocast(
            device_type=device_type,
            enabled=amp_enabled
        ):

            z2 = self.online_encoder(
                x2.to(self.online_device),
                edge2.to(self.online_device)
            )

            z2 = z2[
                mask2.to(self.online_device)
            ]

            q2 = self.predictor(z2)

            loss2 = _cosine_loss(
                q2,
                t1
            )

        loss2.backward()

        loss2_value = loss2.item()

        del z2
        del q2
        del loss2
        del t1
        del t2

        return loss1_value + loss2_value

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    @torch.no_grad()
    def get_embeddings(
        self,
        x,
        edge_index
    ):

        self.online_encoder.eval()

        return self.online_encoder(
            x.to(self.online_device),
            edge_index.to(self.online_device)
        )


# ============================================================
# BGRL loss
# ============================================================

def _cosine_loss(
    p,
    z
):

    p = F.normalize(
        p,
        dim=-1
    )

    z = F.normalize(
        z,
        dim=-1
    )

    return -(
        p * z
    ).sum(
        dim=-1
    ).mean()


# ============================================================
# Linear evaluation
# ============================================================

def create_ssl_split(
    num_nodes,
    seed
):
    """
    Half-Hop paper SSL protocol:

        10% train
        10% validation
        80% test
    """

    rng = np.random.default_rng(
        seed
    )

    indices = rng.permutation(
        num_nodes
    )

    n_train = int(
        0.10 * num_nodes
    )

    n_val = int(
        0.10 * num_nodes
    )

    train_idx = indices[
        :n_train
    ]

    val_idx = indices[
        n_train:n_train + n_val
    ]

    test_idx = indices[
        n_train + n_val:
    ]

    train_mask = np.zeros(
        num_nodes,
        dtype=bool
    )

    val_mask = np.zeros(
        num_nodes,
        dtype=bool
    )

    test_mask = np.zeros(
        num_nodes,
        dtype=bool
    )

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return (
        train_mask,
        val_mask,
        test_mask
    )


def linear_eval(
    embeddings,
    labels,
    train_mask,
    val_mask,
    test_mask,
):
    """
    Logistic regression linear evaluation.
    """

    clf = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "lr",
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                C=1.0
            )
        )
    ])

    clf.fit(
        embeddings[train_mask],
        labels[train_mask]
    )

    val_acc = clf.score(
        embeddings[val_mask],
        labels[val_mask]
    )

    test_acc = clf.score(
        embeddings[test_mask],
        labels[test_mask]
    )

    return val_acc, test_acc


# ============================================================
# Main experiment
# ============================================================

def run_bgrl(
    dataset_name="amazon-computers",

    hidden=512,

    out_channels=256,

    epochs=1000,

    lr=1e-5,

    tau=0.99,

    # --------------------------------------------------------
    # Half-Hop
    # --------------------------------------------------------

    hh_p1=0.75,
    hh_p2=0.75,

    hh_alpha1=0.50,
    hh_alpha2=0.50,

    # --------------------------------------------------------
    # Feature masking
    # Amazon Computers values from Table 6
    # --------------------------------------------------------

    feature_drop1=0.20,
    feature_drop2=0.10,

    # --------------------------------------------------------
    # Edge masking
    # Amazon Computers values from Table 6
    # --------------------------------------------------------

    edge_drop1=0.50,
    edge_drop2=0.40,

    seed=42,

    use_amp=True,
):

    os.environ[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = "expandable_segments:True"

    set_seed(seed)

    # ========================================================
    # Dataset
    # ========================================================

    dataset, data = load_dataset(
        dataset_name
    )

    in_channels = dataset.num_features
    num_classes = dataset.num_classes

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        f"Dataset:       {dataset_name}"
    )

    print(
        f"Nodes:         {data.num_nodes}"
    )

    print(
        f"Features:      {in_channels}"
    )

    print(
        f"Classes:       {num_classes}"
    )

    print("=" * 70)

    # ========================================================
    # GPU configuration
    # ========================================================

    num_gpus = torch.cuda.device_count()

    if num_gpus >= 2:

        online_device = torch.device(
            "cuda:0"
        )

        target_device = torch.device(
            "cuda:1"
        )

        print(
            f"Detected {num_gpus} GPUs."
        )

        print(
            "Online network:",
            torch.cuda.get_device_name(0)
        )

        print(
            "Target network:",
            torch.cuda.get_device_name(1)
        )

    elif torch.cuda.is_available():

        online_device = torch.device(
            "cuda:0"
        )

        target_device = torch.device(
            "cuda:0"
        )

        print(
            "Using single GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        online_device = torch.device(
            "cpu"
        )

        target_device = torch.device(
            "cpu"
        )

        print(
            "Using CPU."
        )

    # ========================================================
    # Model
    # ========================================================

    encoder = GCNEncoder(
        in_channels=in_channels,
        hidden_channels=hidden,
        out_channels=out_channels
    )

    predictor = Predictor(
        in_channels=out_channels,
        hidden_channels=hidden,
        out_channels=out_channels
    )

    model = BGRL(
        encoder=encoder,
        predictor=predictor,
        online_device=online_device,
        target_device=target_device
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        list(
            model.online_encoder.parameters()
        )
        +
        list(
            model.predictor.parameters()
        ),

        lr=lr,

        weight_decay=1e-5
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    # ========================================================
    # Print experiment configuration
    # ========================================================

    print()
    print("=" * 70)
    print("HALF-HOP BGRL CONFIGURATION")
    print("=" * 70)

    print(
        f"Epochs:              {epochs}"
    )

    print(
        f"Learning rate:       {lr}"
    )

    print(
        f"EMA tau:             {tau}"
    )

    print()

    print("VIEW 1")

    print(
        f"  Half-Hop p:        {hh_p1}"
    )

    print(
        f"  Half-Hop alpha:    {hh_alpha1}"
    )

    print(
        f"  Feature drop:      {feature_drop1}"
    )

    print(
        f"  Edge drop:         {edge_drop1}"
    )

    print()

    print("VIEW 2")

    print(
        f"  Half-Hop p:        {hh_p2}"
    )

    print(
        f"  Half-Hop alpha:    {hh_alpha2}"
    )

    print(
        f"  Feature drop:      {feature_drop2}"
    )

    print(
        f"  Edge drop:         {edge_drop2}"
    )

    print("=" * 70)

    # ========================================================
    # Training
    # ========================================================

    model.train()

    progress = trange(
        1,
        epochs + 1,
        desc=f"HH-BGRL {dataset_name}",
        unit="epoch"
    )

    for epoch in progress:

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # View 1
        # ----------------------------------------------------

        aug1, mask1 = make_view(
            data=data,

            hh_alpha=hh_alpha1,
            hh_p=hh_p1,

            feature_drop=feature_drop1,
            edge_drop=edge_drop1
        )

        # ----------------------------------------------------
        # View 2
        # ----------------------------------------------------

        aug2, mask2 = make_view(
            data=data,

            hh_alpha=hh_alpha2,
            hh_p=hh_p2,

            feature_drop=feature_drop2,
            edge_drop=edge_drop2
        )

        # ----------------------------------------------------
        # BGRL step
        # ----------------------------------------------------

        loss_value = model.train_step(
            x1=aug1.x,
            edge1=aug1.edge_index,
            mask1=mask1,

            x2=aug2.x,
            edge2=aug2.edge_index,
            mask2=mask2,

            use_amp=use_amp
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # EMA target update
        # ----------------------------------------------------

        model.update_target(
            tau=tau
        )

        # ----------------------------------------------------
        # LR scheduler
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if epoch == 1 or epoch % 100 == 0:

            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        del aug1
        del aug2

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # Embedding extraction
    # ========================================================

    print()
    print(
        "Extracting final embeddings..."
    )

    model.eval()

    with torch.no_grad():

        embeddings = model.get_embeddings(
            data.x,
            data.edge_index
        )

    embeddings = (
        embeddings
        .cpu()
        .numpy()
    )

    labels = (
        data.y
        .cpu()
        .numpy()
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # ========================================================
    # SSL evaluation split
    # ========================================================

    train_mask, val_mask, test_mask = create_ssl_split(
        num_nodes=data.num_nodes,
        seed=seed
    )

    print()
    print("=" * 70)
    print("LINEAR EVALUATION")
    print("=" * 70)

    print(
        f"Train nodes:       {train_mask.sum()} "
        f"({train_mask.mean():.1%})"
    )

    print(
        f"Validation nodes:  {val_mask.sum()} "
        f"({val_mask.mean():.1%})"
    )

    print(
        f"Test nodes:        {test_mask.sum()} "
        f"({test_mask.mean():.1%})"
    )

    # ========================================================
    # Linear probe
    # ========================================================

    val_acc, test_acc = linear_eval(
        embeddings=embeddings,
        labels=labels,

        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    # ========================================================
    # Final result
    # ========================================================

    print()
    print("=" * 70)
    print(
        f"FINAL SSL RESULTS: HH-BGRL on {dataset_name}"
    )
    print("=" * 70)

    print(
        f"Val Accuracy:   {val_acc:.4f} "
        f"({val_acc:.2%})"
    )

    print(
        f"Test Accuracy:  {test_acc:.4f} "
        f"({test_acc:.2%})"
    )

    print("=" * 70)

    return {
        "val_acc": val_acc,
        "test_acc": test_acc
    }


# ============================================================
# Command line interface
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="BGRL + Half-Hop"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="amazon-computers"
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=512
    )

    parser.add_argument(
        "--out_channels",
        type=int,
        default=256
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.99
    )

    parser.add_argument(
        "--hh_p1",
        type=float,
        default=0.75
    )

    parser.add_argument(
        "--hh_p2",
        type=float,
        default=0.75
    )

    parser.add_argument(
        "--hh_alpha1",
        type=float,
        default=0.50
    )

    parser.add_argument(
        "--hh_alpha2",
        type=float,
        default=0.50
    )

    parser.add_argument(
        "--feature_drop1",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--feature_drop2",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--edge_drop1",
        type=float,
        default=0.50
    )

    parser.add_argument(
        "--edge_drop2",
        type=float,
        default=0.40
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--no_amp",
        action="store_true"
    )

    args = parser.parse_args()

    run_bgrl(
        dataset_name=args.dataset,

        hidden=args.hidden,

        out_channels=args.out_channels,

        epochs=args.epochs,

        lr=args.lr,

        tau=args.tau,

        hh_p1=args.hh_p1,
        hh_p2=args.hh_p2,

        hh_alpha1=args.hh_alpha1,
        hh_alpha2=args.hh_alpha2,

        feature_drop1=args.feature_drop1,
        feature_drop2=args.feature_drop2,

        edge_drop1=args.edge_drop1,
        edge_drop2=args.edge_drop2,

        seed=args.seed,

        use_amp=not args.no_amp
    )

