from halfhop.datasets import load_texas
from halfhop.splits import get_split


def test_texas():

    dataset, data = load_texas()

    assert data.num_nodes > 0
    assert data.num_edges > 0
    assert data.x is not None
    assert data.y is not None

    train_mask, val_mask, test_mask = get_split(
        data,
        split_id=0,
    )

    assert train_mask.shape[0] == data.num_nodes
    assert val_mask.shape[0] == data.num_nodes
    assert test_mask.shape[0] == data.num_nodes

    print("Nodes:", data.num_nodes)
    print("Edges:", data.num_edges)
    print("Features:", dataset.num_features)
    print("Classes:", dataset.num_classes)

    print("Train:", train_mask.sum().item())
    print("Validation:", val_mask.sum().item())
    print("Test:", test_mask.sum().item())