import pytest
from halfhop.datasets import load_dataset, ALL_DATASETS

def test_load_dataset_invalid():
    with pytest.raises(ValueError):
        load_dataset("invalid_dataset_name")

def test_dataset_names():
    assert len(ALL_DATASETS) == 11
    assert "texas" in ALL_DATASETS
    assert "wikics" in ALL_DATASETS
