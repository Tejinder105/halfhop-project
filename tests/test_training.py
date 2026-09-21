import torch
import pytest
from torch_geometric.data import Data

from halfhop.gcn import GCN
from halfhop.training import train_step, evaluate, run_training_loop

@pytest.fixture
def basic_data():
    x = torch.randn(5, 4)
    y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3],
        [1, 0, 2, 1, 4],
    ], dtype=torch.long)
    
    train_mask = torch.tensor([True, True, False, False, False])
    val_mask = torch.tensor([False, False, True, False, False])
    test_mask = torch.tensor([False, False, False, True, True])
    
    return Data(x=x, y=y, edge_index=edge_index, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

def test_train_step(basic_data):
    model = GCN(in_channels=4, hidden_channels=8, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    loss = train_step(model, basic_data, optimizer, basic_data.train_mask)
    assert isinstance(loss, float)
    assert loss > 0

def test_evaluate(basic_data):
    model = GCN(in_channels=4, hidden_channels=8, out_channels=2)
    train_acc, val_acc, test_acc = evaluate(
        model, basic_data, 
        basic_data.train_mask, basic_data.val_mask, basic_data.test_mask
    )
    
    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= val_acc <= 1.0
    assert 0.0 <= test_acc <= 1.0

def test_run_training_loop(basic_data):
    model = GCN(in_channels=4, hidden_channels=8, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    best_val, best_test = run_training_loop(
        model, basic_data, optimizer,
        basic_data.train_mask, basic_data.val_mask, basic_data.test_mask,
        epochs=5, verbose=False
    )
    
    assert 0.0 <= best_val <= 1.0
    assert 0.0 <= best_test <= 1.0
