"""Official BGRL configs plus Half-Hop Table 6 augmentation values.

BGRL architecture / optimizer / schedule: Thakoor et al. repo `config/*.cfg`.
Half-Hop p/alpha and the FeatDrop/EdgeDrop rates for HH-BGRL: Half-Hop Table 6.
"""

from __future__ import annotations

from copy import deepcopy

DATASET_ALIASES = {
    "computers": "amazon-computers",
    "amazon-computers": "amazon-computers",
    "amazon_computers": "amazon-computers",
    "photos": "amazon-photos",
    "amazon-photos": "amazon-photos",
    "amazon_photos": "amazon-photos",
    "amazon-photo": "amazon-photos",
    "cs": "coauthor-cs",
    "coauthor-cs": "coauthor-cs",
    "coauthor_cs": "coauthor-cs",
    "physics": "coauthor-physics",
    "coauthor-physics": "coauthor-physics",
    "coauthor_physics": "coauthor-physics",
    "wiki-cs": "wiki-cs",
    "wikics": "wiki-cs",
    "wiki_cs": "wiki-cs",
}

# Official nerdslab/bgrl config/*.cfg
OFFICIAL_BGRL = {
    "amazon-computers": {
        "graph_encoder_layer": [256, 128],
        "predictor_hidden_size": 512,
        "epochs": 10000,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "mm": 0.99,
        "lr_warmup_epochs": 1000,
        "drop_edge_p_1": 0.5,
        "drop_feat_p_1": 0.2,
        "drop_edge_p_2": 0.4,
        "drop_feat_p_2": 0.1,
        "eval_epochs": 250,
    },
    "amazon-photos": {
        "graph_encoder_layer": [256, 128],
        "predictor_hidden_size": 512,
        "epochs": 10000,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "mm": 0.99,
        "lr_warmup_epochs": 1000,
        "drop_edge_p_1": 0.4,
        "drop_feat_p_1": 0.1,
        "drop_edge_p_2": 0.1,
        "drop_feat_p_2": 0.2,
        "eval_epochs": 250,
    },
    "coauthor-cs": {
        "graph_encoder_layer": [512, 256],
        "predictor_hidden_size": 512,
        "epochs": 10000,
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "mm": 0.99,
        "lr_warmup_epochs": 1000,
        "drop_edge_p_1": 0.3,
        "drop_feat_p_1": 0.3,
        "drop_edge_p_2": 0.2,
        "drop_feat_p_2": 0.4,
        "eval_epochs": 250,
    },
    "coauthor-physics": {
        "graph_encoder_layer": [256, 128],
        "predictor_hidden_size": 512,
        "epochs": 10000,
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "mm": 0.99,
        "lr_warmup_epochs": 1000,
        "drop_edge_p_1": 0.4,
        "drop_feat_p_1": 0.1,
        "drop_edge_p_2": 0.1,
        "drop_feat_p_2": 0.4,
        "eval_epochs": 250,
    },
    "wiki-cs": {
        "graph_encoder_layer": [512, 256],
        "predictor_hidden_size": 512,
        "epochs": 10000,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "mm": 0.99,
        "lr_warmup_epochs": 1000,
        "drop_edge_p_1": 0.2,
        "drop_feat_p_1": 0.2,
        "drop_edge_p_2": 0.3,
        "drop_feat_p_2": 0.1,
        "eval_epochs": 250,
    },
}

# Half-Hop paper Table 6 (HH-BGRL / HH-GRACE).
TABLE6_HALFHOP = {
    "amazon-computers": {
        "p_hh_1": 0.75,
        "p_hh_2": 0.75,
        "alpha_1": 0.50,
        "alpha_2": 0.50,
        "drop_feat_p_1": 0.20,
        "drop_feat_p_2": 0.10,
        "drop_edge_p_1": 0.50,
        "drop_edge_p_2": 0.40,
    },
    "amazon-photos": {
        "p_hh_1": 0.75,
        "p_hh_2": 0.75,
        "alpha_1": 0.50,
        "alpha_2": 0.50,
        "drop_feat_p_1": 0.10,
        "drop_feat_p_2": 0.20,
        "drop_edge_p_1": 0.40,
        "drop_edge_p_2": 0.10,
    },
    "coauthor-cs": {
        "p_hh_1": 0.75,
        "p_hh_2": 0.75,
        "alpha_1": 0.50,
        "alpha_2": 0.50,
        "drop_feat_p_1": 0.30,
        "drop_feat_p_2": 0.40,
        "drop_edge_p_1": 0.30,
        "drop_edge_p_2": 0.20,
    },
    "coauthor-physics": {
        "p_hh_1": 0.75,
        "p_hh_2": 0.75,
        "alpha_1": 0.50,
        "alpha_2": 0.50,
        "drop_feat_p_1": 0.10,
        "drop_feat_p_2": 0.40,
        "drop_edge_p_1": 0.40,
        "drop_edge_p_2": 0.10,
    },
    "wiki-cs": {
        "p_hh_1": 0.75,
        "p_hh_2": 0.75,
        "alpha_1": 0.50,
        "alpha_2": 0.50,
        "drop_feat_p_1": 0.20,
        "drop_feat_p_2": 0.10,
        "drop_edge_p_1": 0.20,
        "drop_edge_p_2": 0.30,
    },
}

AUGMENTATIONS = ("none", "feat_edge", "hh", "feat_edge_hh")

RESULT_DIR_NAMES = {
    "amazon-computers": "computers",
    "amazon-photos": "photos",
    "coauthor-cs": "cs",
    "coauthor-physics": "physics",
    "wiki-cs": "wiki-cs",
}

METHOD_NAMES = {
    "none": "BGRL",
    "feat_edge": "BGRL + FeatDrop + EdgeDrop",
    "hh": "BGRL + Half-Hop",
    "feat_edge_hh": "BGRL + FeatDrop + EdgeDrop + Half-Hop",
}


def canonical_dataset(name: str) -> str:
    key = name.lower().replace("_", "-")
    if key not in DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Valid names: {sorted(set(DATASET_ALIASES.values()))}"
        )
    return DATASET_ALIASES[key]


def get_run_config(dataset: str, augmentation: str) -> dict:
    dataset = canonical_dataset(dataset)
    if augmentation not in AUGMENTATIONS:
        raise ValueError(
            f"Unknown augmentation '{augmentation}'. "
            f"Valid: {list(AUGMENTATIONS)}"
        )

    cfg = deepcopy(OFFICIAL_BGRL[dataset])
    hh = TABLE6_HALFHOP[dataset]
    use_feat_edge = augmentation in ("feat_edge", "feat_edge_hh")
    use_hh = augmentation in ("hh", "feat_edge_hh")

    if use_feat_edge:
        cfg["drop_edge_p_1"] = hh["drop_edge_p_1"]
        cfg["drop_edge_p_2"] = hh["drop_edge_p_2"]
        cfg["drop_feat_p_1"] = hh["drop_feat_p_1"]
        cfg["drop_feat_p_2"] = hh["drop_feat_p_2"]
    else:
        cfg["drop_edge_p_1"] = 0.0
        cfg["drop_edge_p_2"] = 0.0
        cfg["drop_feat_p_1"] = 0.0
        cfg["drop_feat_p_2"] = 0.0

    if use_hh:
        cfg["p_hh_1"] = hh["p_hh_1"]
        cfg["p_hh_2"] = hh["p_hh_2"]
        cfg["alpha_1"] = hh["alpha_1"]
        cfg["alpha_2"] = hh["alpha_2"]
    else:
        cfg["p_hh_1"] = 0.0
        cfg["p_hh_2"] = 0.0
        cfg["alpha_1"] = hh["alpha_1"]
        cfg["alpha_2"] = hh["alpha_2"]

    cfg["dataset"] = dataset
    cfg["augmentation"] = augmentation
    cfg["use_halfhop"] = use_hh
    cfg["use_feat_edge"] = use_feat_edge
    cfg["method"] = METHOD_NAMES[augmentation]
    cfg["result_dataset"] = RESULT_DIR_NAMES[dataset]
    return cfg
