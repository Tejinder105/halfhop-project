import torch
import torch.nn.functional as F

from halfhop.datasets import load_texas
from halfhop.gcn import GCN
from halfhop.splits import get_split


def train(model, data, train_mask, optimizer):

    model.train()

    optimizer.zero_grad()

    out = model(
        data.x,
        data.edge_index,
    )

    loss = F.cross_entropy(
        out[train_mask],
        data.y[train_mask],
    )

    loss.backward()

    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, train_mask, val_mask, test_mask):

    model.eval()

    out = model(
        data.x,
        data.edge_index,
    )

    pred = out.argmax(dim=-1)

    train_acc = (
        pred[train_mask] == data.y[train_mask]
    ).float().mean().item()

    val_acc = (
        pred[val_mask] == data.y[val_mask]
    ).float().mean().item()

    test_acc = (
        pred[test_mask] == data.y[test_mask]
    ).float().mean().item()

    return train_acc, val_acc, test_acc


def run_split(split_id):

    dataset, data = load_texas()

    train_mask, val_mask, test_mask = get_split(
        data,
        split_id,
    )

    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        dropout=0.5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4,
    )

    best_val = 0.0
    best_test = 0.0

    for epoch in range(200):

        loss = train(
            model,
            data,
            train_mask,
            optimizer,
        )

        _, val_acc, test_acc = evaluate(
            model,
            data,
            train_mask,
            val_mask,
            test_mask,
        )

        if val_acc > best_val:

            best_val = val_acc
            best_test = test_acc

        if epoch % 20 == 0:

            print(
                f"Split {split_id:02d} | "
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | "
                f"Val {val_acc:.4f} | "
                f"Test {test_acc:.4f}"
            )

    return best_val, best_test


def run_split(split_id):

    dataset, data = load_texas()

    train_mask, val_mask, test_mask = get_split(
        data,
        split_id,
    )

    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        dropout=0.5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4,
    )

    best_val = 0.0
    best_test = 0.0

    for epoch in range(200):

        loss = train(
            model,
            data,
            train_mask,
            optimizer,
        )

        _, val_acc, test_acc = evaluate(
            model,
            data,
            train_mask,
            val_mask,
            test_mask,
        )

        if val_acc > best_val:

            best_val = val_acc
            best_test = test_acc

        if epoch % 20 == 0:

            print(
                f"Split {split_id:02d} | "
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | "
                f"Val {val_acc:.4f} | "
                f"Test {test_acc:.4f}"
            )

    return best_val, best_test


def main():

    test_results = []

    for split_id in range(10):

        print()
        print("=" * 60)
        print(f"Running split {split_id}")
        print("=" * 60)

        best_val, best_test = run_split(
            split_id
        )

        print(
            f"Best validation: {best_val:.4f}"
        )

        print(
            f"Test at best validation: {best_test:.4f}"
        )

        test_results.append(best_test)

    results = torch.tensor(test_results)

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        "Test accuracies:",
        results.tolist()
    )

    print(
        "Mean:",
        results.mean().item()
    )

    print(
        "Std:",
        results.std(unbiased=True).item()
    )


if __name__ == "__main__":
    main()

