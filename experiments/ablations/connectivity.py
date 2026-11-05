import argparse
import torch

from halfhop.datasets import load_dataset
from halfhop.splits import get_split
from halfhop.training import run_training_loop
from halfhop.reproducibility import set_seed
from halfhop.hh_gcn import HHGCN
from experiments.supervised.run import load_config


def run_connectivity_ablation(dataset_name="texas", epochs=200, seed=42):
    set_seed(seed)
    dataset, data = load_dataset(dataset_name)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    config = load_config(dataset_name, "hh-gcn")
    lr = config.get("lr", 0.01)
    wd = config.get("wd", 5e-4)
    hidden = config.get("hidden", 64)
    depth = config.get("depth", 2)
    dropout = config.get("dropout", 0.5)
    alpha = config.get("alpha", 0.5)
    p = config.get("p", 1.0)

    connectivities = ["proposed", "hh1", "hh2"]
    results_summary = {c: [] for c in connectivities}

    for split_id in range(10):
        train_mask, val_mask, test_mask = get_split(data, split_id)
        
        for conn in connectivities:
            # We must pass the connectivity parameter to HH_GCN
            # Let's assume HH_GCN supports a `connectivity` parameter as per Phase 1
            model = HHGCN(
                num_features, hidden, num_classes, 
                dropout=dropout, depth=depth, alpha=alpha, p=p, 
                connectivity=conn
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            
            _, best_test = run_training_loop(
                model=model, data=data, optimizer=optimizer,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                epochs=epochs, verbose=False
            )
            results_summary[conn].append(best_test)

    print("\n" + "="*60)
    print(f"CONNECTIVITY ABLATION RESULTS on {dataset_name.upper()}")
    print("="*60)
    for conn in connectivities:
        res = torch.tensor(results_summary[conn])
        print(f"{conn:>10}: {res.mean().item():.2%} ± {res.std(unbiased=True).item():.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run connectivity ablation")
    parser.add_argument("--dataset", type=str, default="texas", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    run_connectivity_ablation(args.dataset, args.epochs, args.seed)
