import torch
import torch.nn.functional as F

from halfhop.datasets import load_texas
from halfhop.hh_gcn import HHGCN
from halfhop.splits import get_split


def train(model, data, train_mask, optimizer):

    model.train()

    optimizer.zero_grad()

    out = model(data)

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

    out = model(data)

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

    model = HHGCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        alpha=0.5,
        p=1.0,
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


if __name__ == "__main__":

    val, test = run_split(0)

    print()
    print("Best validation:", val)
    print("Test:", test)

