"""Training and evaluation utilities for supervised experiments.

Implements the standard supervised learning protocol:
1. Train on training nodes.
2. Select the best model (epoch) based on validation accuracy.
3. Evaluate test accuracy using the selected best model.
"""

import copy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# Automatically use GPU if available, otherwise fall back to CPU.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _move_data(data: Data, device: torch.device) -> Data:
    """Move all tensor attributes of a Data object to device."""
    data = copy.copy(data)  # shallow copy to avoid mutating original
    for key in data.keys():
        val = getattr(data, key, None)
        if isinstance(val, torch.Tensor):
            setattr(data, key, val.to(device))
    return data


def _forward_pass(model: torch.nn.Module, data: Data) -> torch.Tensor:
    if hasattr(model, 'halfhop'):
        return model(data)
    else:
        return model(data.x, data.edge_index)

def train_step(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    train_mask: torch.Tensor,
) -> float:
    """Perform a single training step.

    Args:
        model (torch.nn.Module): The GNN or HH-GNN model.
        data (Data): The graph data object.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_mask (torch.Tensor): Boolean mask of shape [N] for training nodes.

    Returns:
        float: Training loss for this step.
    """
    model.train()
    optimizer.zero_grad()
    
    out = _forward_pass(model, data)
    
    # Check if this is a classification task (1D target)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> tuple:
    """Evaluate the model on train, val, and test splits.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (Data): The graph data object.
        train_mask (torch.Tensor): Boolean mask for train nodes.
        val_mask (torch.Tensor): Boolean mask for val nodes.
        test_mask (torch.Tensor): Boolean mask for test nodes.

    Returns:
        tuple: (train_acc, val_acc, test_acc)
    """
    model.eval()
    
    out = _forward_pass(model, data)
    pred = out.argmax(dim=-1)
    
    # Calculate accuracy
    train_acc = int((pred[train_mask] == data.y[train_mask]).sum()) / int(train_mask.sum())
    val_acc = int((pred[val_mask] == data.y[val_mask]).sum()) / int(val_mask.sum())
    test_acc = int((pred[test_mask] == data.y[test_mask]).sum()) / int(test_mask.sum())
    
    return train_acc, val_acc, test_acc


def run_training_loop(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    epochs: int = 200,
    verbose: bool = False,
    device: torch.device = DEVICE,
) -> tuple:
    """Run a full training loop and return best validation results.

    Implements the standard evaluation protocol: select the epoch with
    the highest validation accuracy, and report test accuracy from that epoch.

    Args:
        model (torch.nn.Module): The model to train.
        data (Data): The graph data object.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_mask (torch.Tensor): Train nodes mask.
        val_mask (torch.Tensor): Validation nodes mask.
        test_mask (torch.Tensor): Test nodes mask.
        epochs (int): Number of training epochs. Default: 200.
        verbose (bool): Whether to print progress. Default: False.

    Returns:
        tuple: (best_val_acc, best_test_acc)
    """
    # Move model and data to the target device
    model = model.to(device)
    data = _move_data(data, device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        loss = train_step(model, data, optimizer, train_mask)
        train_acc, val_acc, test_acc = evaluate(
            model, data, train_mask, val_mask, test_mask
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            
        if verbose and epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )
            
    return best_val_acc, best_test_acc
