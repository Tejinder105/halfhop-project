import argparse
import os
import yaml
import torch
import numpy as np

from halfhop.datasets import load_dataset, HETEROPHILIC_DATASETS, HOMOPHILIC_DATASETS
from halfhop.splits import get_split, generate_homophilic_splits
from halfhop.training import run_training_loop
from halfhop.reproducibility import set_seed

from halfhop.gcn import GCN
from halfhop.graphsage import GraphSAGE
from halfhop.gat import GAT
from halfhop.hh_gcn import HHGCN
from halfhop.hh_graphsage import HHGraphSAGE
from halfhop.hh_gat import HHGAT
from halfhop.adahop import (
    AdaHopGCN,
    AdaHopSAGE,
    AdaHopGAT,
    SoftAdaHopGCN,
    StatefulHHGCN,
    MultiDelayGCN,
)


def get_model(model_name, num_features, num_classes, config):
    model_name = model_name.lower()
    
    # Common hyperparameters
    hidden = config.get("hidden", 64)
    depth = config.get("depth", 2)
    dropout = config.get("dropout", 0.5)
    
    # Half-Hop specific hyperparameters
    alpha = config.get("alpha", 0.5)
    p = config.get("p", 1.0)
    
    if model_name == "gcn":
        return GCN(num_features, hidden, num_classes, dropout=dropout, depth=depth)
    elif model_name == "sage" or model_name == "graphsage":
        return GraphSAGE(num_features, hidden, num_classes, dropout=dropout, depth=depth)
    elif model_name == "gat":
        return GAT(num_features, hidden, num_classes, dropout=dropout, depth=depth)
    elif model_name == "hh-gcn":
        return HHGCN(num_features, hidden, num_classes, dropout=dropout, depth=depth, alpha=alpha, p=p)
    elif model_name == "hh-sage" or model_name == "hh-graphsage":
        return HHGraphSAGE(num_features, hidden, num_classes, dropout=dropout, depth=depth, alpha=alpha, p=p)
    elif model_name == "hh-gat":
        return HHGAT(num_features, hidden, num_classes, dropout=dropout, depth=depth, alpha=alpha, p=p)
    elif model_name == "adahop-gcn":
        alpha_hidden = config.get("alpha_hidden", 32)
        entropy_reg = config.get("entropy_reg", 0.0)
        p = config.get("p", 1.0)
        return AdaHopGCN(num_features, hidden, num_classes,
                         alpha_hidden=alpha_hidden, entropy_reg=entropy_reg,
                         p=p, dropout=dropout, depth=depth)
    elif model_name == "adahop-sage":
        alpha_hidden = config.get("alpha_hidden", 32)
        entropy_reg = config.get("entropy_reg", 0.0)
        p = config.get("p", 1.0)
        return AdaHopSAGE(num_features, hidden, num_classes,
                          alpha_hidden=alpha_hidden, entropy_reg=entropy_reg,
                          p=p, dropout=dropout, depth=depth)
    elif model_name == "adahop-gat":
        alpha_hidden = config.get("alpha_hidden", 32)
        entropy_reg = config.get("entropy_reg", 0.0)
        p = config.get("p", 1.0)
        return AdaHopGAT(num_features, hidden, num_classes,
                         alpha_hidden=alpha_hidden, entropy_reg=entropy_reg,
                         p=p, dropout=dropout, depth=depth)
    elif model_name == "soft-adahop-gcn":
        alpha = config.get("alpha", 0.5)
        gate_hidden = config.get("gate_hidden", 32)
        gamma_init = config.get("gamma_init", None)
        return SoftAdaHopGCN(num_features, hidden, num_classes,
                             alpha=alpha, gate_hidden=gate_hidden,
                             gamma_init=gamma_init, dropout=dropout, depth=depth)
    elif model_name == "stateful-hh-gcn":
        alpha = config.get("alpha", 0.5)
        p = config.get("p", 1.0)
        return StatefulHHGCN(num_features, hidden, num_classes,
                             alpha=alpha, p=p, dropout=dropout, depth=depth)
    elif model_name in ("multidelay-gcn", "multi-delay-gcn"):
        k = config.get("k", 2)
        alpha = config.get("alpha", 0.5)
        p = config.get("p", 1.0)
        return MultiDelayGCN(num_features, hidden, num_classes,
                             k=k, alpha=alpha, p=p, dropout=dropout, depth=depth)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_config(dataset, model_name):
    config_path = f"configs/{dataset.lower()}.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            if model_name.lower() in cfg:
                return cfg[model_name.lower()]
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run supervised experiments")
    parser.add_argument("--dataset", type=str, default="texas", help="Dataset name")
    parser.add_argument("--model", type=str, default="hh-gcn", help="Model name (e.g. gcn, sage, gat, hh-gcn, hh-sage, hh-gat)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print epoch details")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda:0, cuda:1, cpu)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device) if args.device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device: {device}")

    dataset_name = args.dataset.lower().replace("_", "-")
    if dataset_name in ("amazon-rating", "amazonratings"):
        dataset_name = "amazon-ratings"

    dataset, data = load_dataset(dataset_name)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    
    config = load_config(dataset_name, args.model)
    print(f"Loaded config for {args.model} on {dataset_name}: {config}")
    
    lr = config.get("lr", 0.01)
    wd = config.get("wd", 5e-4)

    if dataset_name in HETEROPHILIC_DATASETS:
        # Texas…Cornell and Amazon-ratings: 10 fixed masks on data.
        num_splits = data.train_mask.size(1) if data.train_mask.dim() == 2 else 10
    elif dataset_name in HOMOPHILIC_DATASETS:
        num_splits = 20
        if dataset_name != "wikics":
            data = generate_homophilic_splits(data, num_splits=num_splits)
    else:
        num_splits = 1

    test_results = []
    
    for split_id in range(num_splits):
        print(f"Running split {split_id}/{num_splits-1}")
        
        train_mask, val_mask, test_mask = get_split(data, split_id)
        
        model = get_model(args.model, num_features, num_classes, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
        best_val, best_test = run_training_loop(
            model=model,
            data=data,
            optimizer=optimizer,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            epochs=args.epochs,
            verbose=args.verbose,
            device=device,
        )
        
        print(f"Split {split_id} - Best Val: {best_val:.4f}, Test at Best Val: {best_test:.4f}")
        test_results.append(best_test)
        
    results = torch.tensor(test_results)
    print("\n" + "="*60)
    print(f"FINAL RESULTS: {args.model.upper()} on {dataset_name}")
    print("="*60)
    print(f"Test accuracies: {results.tolist()}")
    print(f"Mean: {results.mean().item():.4f}")
    print(f"Std:  {results.std(unbiased=True).item():.4f}")

if __name__ == "__main__":
    main()
