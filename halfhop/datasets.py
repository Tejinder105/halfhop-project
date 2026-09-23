"""Dataset loaders for all 11 datasets used in the Half-Hop paper.

Heterophilic (10 pre-set splits from Pei et al. 2020):
    Texas, Wisconsin, Actor, Squirrel, Chameleon, Cornell

Homophilic (20 random splits, 60:20:20 or WikiCS pre-set masks):
    Amazon Photos, Amazon Computers, Coauthor CS, Coauthor Physics, WikiCS
"""

from torch_geometric.datasets import (
    WebKB,
    Actor,
    WikipediaNetwork,
    Amazon,
    Coauthor,
    WikiCS,
)


# ---------------------------------------------------------------------------
# Heterophilic datasets
# ---------------------------------------------------------------------------

def load_texas(root: str = "data") -> tuple:
    """Load the Texas dataset (WebKB).

    Returns:
        Tuple[Dataset, Data]: The dataset object and the graph data object.
    """
    dataset = WebKB(root=root, name="Texas")
    return dataset, dataset[0]


def load_wisconsin(root: str = "data") -> tuple:
    """Load the Wisconsin dataset (WebKB)."""
    dataset = WebKB(root=root, name="Wisconsin")
    return dataset, dataset[0]


def load_cornell(root: str = "data") -> tuple:
    """Load the Cornell dataset (WebKB)."""
    dataset = WebKB(root=root, name="Cornell")
    return dataset, dataset[0]


def load_actor(root: str = "data") -> tuple:
    """Load the Actor (Film) dataset.

    Note: The paper refers to this as 'Film'. PyG uses the name 'Actor'.
    """
    dataset = Actor(root=f"{root}/Actor")
    return dataset, dataset[0]


def load_squirrel(root: str = "data") -> tuple:
    """Load the Squirrel dataset (WikipediaNetwork)."""
    dataset = WikipediaNetwork(root=root, name="squirrel")
    return dataset, dataset[0]


def load_chameleon(root: str = "data") -> tuple:
    """Load the Chameleon dataset (WikipediaNetwork)."""
    dataset = WikipediaNetwork(root=root, name="chameleon")
    return dataset, dataset[0]


# ---------------------------------------------------------------------------
# Homophilic datasets
# ---------------------------------------------------------------------------

def load_amazon_photos(root: str = "data") -> tuple:
    """Load the Amazon Photos dataset."""
    dataset = Amazon(root=root, name="photo")
    return dataset, dataset[0]


def load_amazon_computers(root: str = "data") -> tuple:
    """Load the Amazon Computers dataset."""
    dataset = Amazon(root=root, name="computers")
    return dataset, dataset[0]


def load_coauthor_cs(root: str = "data") -> tuple:
    """Load the Coauthor CS dataset."""
    dataset = Coauthor(root=root, name="cs")
    return dataset, dataset[0]


def load_coauthor_physics(root: str = "data") -> tuple:
    """Load the Coauthor Physics dataset."""
    dataset = Coauthor(root=root, name="physics")
    return dataset, dataset[0]


def load_wikics(root: str = "data") -> tuple:
    """Load the WikiCS dataset.

    WikiCS provides 20 pre-set train/val masks. Test nodes are fixed.
    """
    dataset = WikiCS(root=f"{root}/WikiCS")
    return dataset, dataset[0]


# ---------------------------------------------------------------------------
# Unified loader by name
# ---------------------------------------------------------------------------

HETEROPHILIC_DATASETS = [
    "texas", "wisconsin", "cornell", "actor", "squirrel", "chameleon"
]
HOMOPHILIC_DATASETS = [
    "amazon-photos", "amazon-computers",
    "coauthor-cs", "coauthor-physics",
    "wikics",
]
ALL_DATASETS = HETEROPHILIC_DATASETS + HOMOPHILIC_DATASETS

_LOADERS = {
    "texas": load_texas,
    "wisconsin": load_wisconsin,
    "cornell": load_cornell,
    "actor": load_actor,
    "squirrel": load_squirrel,
    "chameleon": load_chameleon,
    "amazon-photos": load_amazon_photos,
    "amazon-computers": load_amazon_computers,
    "coauthor-cs": load_coauthor_cs,
    "coauthor-physics": load_coauthor_physics,
    "wikics": load_wikics,
}


def load_dataset(name: str, root: str = "data") -> tuple:
    """Unified dataset loader by name.

    Args:
        name (str): Dataset name (case-insensitive). One of:
            ``'texas'``, ``'wisconsin'``, ``'cornell'``,
            ``'actor'``, ``'squirrel'``, ``'chameleon'``,
            ``'amazon-photos'``, ``'amazon-computers'``,
            ``'coauthor-cs'``, ``'coauthor-physics'``,
            ``'wikics'``.
        root (str): Root directory for dataset storage. Default: ``'data'``.

    Returns:
        Tuple[Dataset, Data]: The dataset object and graph data.

    Raises:
        ValueError: If ``name`` is not recognized.
    """
    name = name.lower().replace("_", "-")
    aliases = {
        "amazon-photo": "amazon-photos",
        "film": "actor",
    }
    name = aliases.get(name, name)
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Valid names: {sorted(_LOADERS.keys())}"
        )
    return _LOADERS[name](root=root)