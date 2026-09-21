import torch
import pytest
from torch_geometric.data import Data

from halfhop.splits import get_split, make_random_splits, get_wikics_split, num_splits_for

def test_get_split_2d():
    train_mask = torch.ones(5, 10, dtype=torch.bool)
    val_mask = torch.zeros(5, 10, dtype=torch.bool)
    test_mask = torch.zeros(5, 10, dtype=torch.bool)
    
    data = Data(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    assert num_splits_for(data) == 10
    
    tr, v, te = get_split(data, split_id=5)
    assert tr.shape == (5,)
    assert v.shape == (5,)
    assert te.shape == (5,)

def test_get_split_1d():
    train_mask = torch.ones(5, dtype=torch.bool)
    val_mask = torch.zeros(5, dtype=torch.bool)
    test_mask = torch.zeros(5, dtype=torch.bool)
    
    data = Data(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    assert num_splits_for(data) == 1
    
    tr, v, te = get_split(data, split_id=0)
    assert tr.shape == (5,)

def test_make_random_splits():
    train, val, test = make_random_splits(num_nodes=100, train_ratio=0.6, val_ratio=0.2, num_splits=5)
    assert train.shape == (100, 5)
    assert val.shape == (100, 5)
    assert test.shape == (100, 5)
    
    # Check disjoint
    for i in range(5):
        assert torch.all(~(train[:, i] & val[:, i]))
        assert torch.all(~(train[:, i] & test[:, i]))
        assert torch.all(~(val[:, i] & test[:, i]))
        
        # Check totals
        assert train[:, i].sum() == 60
        assert val[:, i].sum() == 20
        assert test[:, i].sum() == 20

def test_get_wikics_split():
    train_mask = torch.ones(5, 20, dtype=torch.bool)
    val_mask = torch.zeros(5, dtype=torch.bool)
    test_mask = torch.zeros(5, dtype=torch.bool)
    
    data = Data(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    tr, v, te = get_wikics_split(data, split_id=5)
    assert tr.shape == (5,)
    assert v.shape == (5,)
    assert te.shape == (5,)
