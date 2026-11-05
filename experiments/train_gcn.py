import torch
import torch.nn.functional as F


def train(model, data, optimizer):

    model.train()

    optimizer.zero_grad()

    out = model(
        data.x,
        data.edge_index
    )

    loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()

    optimizer.step()

    return loss.item()



@torch.no_grad()
def evaluate(model, data):

    model.eval()

    out = model(
        data.x,
        data.edge_index
    )

    pred = out.argmax(dim=-1)

    train_acc = (
        pred[data.train_mask]
        == data.y[data.train_mask]
    ).float().mean().item()

    val_acc = (
        pred[data.val_mask]
        == data.y[data.val_mask]
    ).float().mean().item()

    test_acc = (
        pred[data.test_mask]
        == data.y[data.test_mask]
    ).float().mean().item()

    return train_acc, val_acc, test_acc