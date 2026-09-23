"""Train official BGRL, optionally with Half-Hop as a view augmentation.

Copied training mechanics from nerdslab/bgrl `train_transductive.py`:
    AdamW, CosineDecayScheduler warmup+cosine LR, cosine EMA momentum,
    bootstrap loss ``2 - mean(cos(q1,y2)) - mean(cos(q2,y1))``,
    independent target-encoder reset, original-graph linear eval.

Half-Hop paper Table 3: at test time the original graph is used.
Slow nodes are training-only augmentation nodes.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from tqdm import tqdm

from .compat import prepare_official_imports
from .configs import get_run_config
from .eval import fit_logistic_regression
from .views import build_view_transform

prepare_official_imports()

from bgrl.bgrl import BGRL, compute_representations  # noqa: E402
from bgrl.data import get_dataset, get_wiki_cs  # noqa: E402
from bgrl.models import GCN  # noqa: E402
from bgrl.predictors import MLP_Predictor  # noqa: E402
from bgrl.scheduler import CosineDecayScheduler  # noqa: E402
from bgrl.utils import set_random_seeds  # noqa: E402


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_flag)


def _load_data(dataset_name: str, dataset_dir: str, device: torch.device):
    if dataset_name == "wiki-cs":
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(dataset_dir)
        data = dataset[0].to(device)
        return dataset, data, (train_masks, val_masks, test_masks)
    dataset = get_dataset(dataset_dir, dataset_name)
    data = dataset[0].to(device)
    return dataset, data, None


def _masked_forward(model, online_data, target_data, online_mask, target_mask):
    online_y = model.online_encoder(online_data)
    online_q = model.predictor(online_y[online_mask])
    with torch.no_grad():
        target_y = model.target_encoder(target_data)[target_mask].detach()
    return online_q, target_y


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_experiment(
    dataset: str = "computers",
    augmentation: str = "feat_edge",
    seed: int = 0,
    data_seed: int | None = None,
    epochs: int | None = None,
    device: str = "auto",
    output_dir: str | None = None,
    dataset_dir: str = "data",
    eval_epochs: int | None = None,
    eval_jobs: int = 5,
    low_memory: bool = False,
    debug_counts: bool = True,
    save_embeddings: bool = True,
) -> dict:
    cfg = get_run_config(dataset, augmentation)
    if epochs is not None:
        cfg["epochs"] = epochs
    if eval_epochs is not None:
        cfg["eval_epochs"] = eval_epochs
    if data_seed is None:
        data_seed = seed

    device_obj = resolve_device(device)
    set_random_seeds(seed)

    out = Path(output_dir) if output_dir else (
        Path("results") / "bgrl" / cfg["result_dataset"] / cfg["augmentation"] / f"seed_{seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger("halfhop_bgrl")
    log.info("Device: %s", device_obj)
    if device_obj.type == "cuda":
        log.info(
            "GPU: %s  (visible %d). Official BGRL is single-GPU; "
            "use CUDA_VISIBLE_DEVICES to pin a second Kaggle T4 to another job.",
            torch.cuda.get_device_name(device_obj),
            torch.cuda.device_count(),
        )

    dataset_obj, data, wiki_splits = _load_data(
        cfg["dataset"], dataset_dir, device_obj
    )
    log.info("Dataset %s: %s", cfg["dataset"], data)

    transform_1 = build_view_transform(
        p_hh=cfg["p_hh_1"],
        alpha=cfg["alpha_1"],
        drop_edge_p=cfg["drop_edge_p_1"],
        drop_feat_p=cfg["drop_feat_p_1"],
        log_counts=debug_counts,
        name="view1",
    )
    transform_2 = build_view_transform(
        p_hh=cfg["p_hh_2"],
        alpha=cfg["alpha_2"],
        drop_edge_p=cfg["drop_edge_p_2"],
        drop_feat_p=cfg["drop_feat_p_2"],
        log_counts=debug_counts,
        name="view2",
    )

    input_size = data.x.size(1)
    representation_size = cfg["graph_encoder_layer"][-1]
    encoder = GCN([input_size] + cfg["graph_encoder_layer"], batchnorm=True)
    predictor = MLP_Predictor(
        representation_size,
        representation_size,
        hidden_size=cfg["predictor_hidden_size"],
    )
    model = BGRL(encoder, predictor).to(device_obj)

    optimizer = AdamW(
        model.trainable_parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    lr_scheduler = CosineDecayScheduler(
        cfg["lr"], cfg["lr_warmup_epochs"], cfg["epochs"]
    )
    mm_scheduler = CosineDecayScheduler(1 - cfg["mm"], 0, cfg["epochs"])

    run_config = {
        **cfg,
        "seed": seed,
        "data_seed": data_seed,
        "device": str(device_obj),
        "low_memory": low_memory,
        "dataset_dir": dataset_dir,
        "output_dir": str(out),
        "embedding_dimension": representation_size,
        "augmentation_order": "deepcopy → Half-Hop → EdgeDrop → FeatDrop",
        "linear_eval": "official BGRL liblinear + C grid, 20%/80% then ShuffleSplit 50/50",
        "test_time_graph": "original graph (Half-Hop Table 3)",
    }
    _write_json(out / "config.json", run_config)

    last_loss = None

    def train_step(step: int) -> float:
        model.train()
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        mm = 1 - mm_scheduler.get(step)
        optimizer.zero_grad()

        view1, mask1 = transform_1(data)
        view2, mask2 = transform_2(data)

        if low_memory:
            q1, y2 = _masked_forward(model, view1, view2, mask1, mask2)
            loss1 = 1 - cosine_similarity(q1, y2.detach(), dim=-1).mean()
            loss1.backward()
            loss1_value = float(loss1.detach())
            del q1, y2, loss1
            q2, y1 = _masked_forward(model, view2, view1, mask2, mask1)
            loss2 = 1 - cosine_similarity(q2, y1.detach(), dim=-1).mean()
            loss2.backward()
            loss = loss1_value + float(loss2.detach())
            del q2, y1, loss2, view1, view2
        else:
            q1, y2 = _masked_forward(model, view1, view2, mask1, mask2)
            q2, y1 = _masked_forward(model, view2, view1, mask2, mask1)
            loss_t = (
                2
                - cosine_similarity(q1, y2.detach(), dim=-1).mean()
                - cosine_similarity(q2, y1.detach(), dim=-1).mean()
            )
            loss_t.backward()
            loss = float(loss_t.detach())
            del q1, y2, q2, y1, loss_t, view1, view2

        optimizer.step()
        model.update_target_network(mm)
        return loss

    def evaluate() -> tuple[float, float | None, torch.Tensor, torch.Tensor]:
        tmp_encoder = deepcopy(model.online_encoder).eval()
        representations, labels = compute_representations(
            tmp_encoder, dataset_obj, device_obj
        )
        if wiki_splits is not None:
            from bgrl.logistic_regression_eval import (
                fit_logistic_regression_preset_splits,
            )

            train_masks, val_masks, test_masks = wiki_splits
            scores = fit_logistic_regression_preset_splits(
                representations.cpu().numpy(),
                labels.cpu().numpy(),
                train_masks,
                val_masks,
                test_masks,
            )
            return float(sum(scores) / len(scores)), None, representations, labels

        scores, best_val = fit_logistic_regression(
            representations.cpu().numpy(),
            labels.cpu().numpy(),
            data_random_seed=data_seed,
            repeat=1,
            n_jobs=eval_jobs,
        )
        return float(scores[0]), best_val, representations, labels

    progress = tqdm(range(1, cfg["epochs"] + 1), desc=f"BGRL {cfg['dataset']} {cfg['augmentation']}")
    for epoch in progress:
        last_loss = train_step(epoch - 1)
        if epoch == 1 or epoch % 100 == 0:
            progress.set_postfix(loss=f"{last_loss:.4f}")
            log.info("epoch %d loss %.6f", epoch, last_loss)
        if cfg["eval_epochs"] > 0 and epoch % cfg["eval_epochs"] == 0:
            acc, val_acc, _, _ = evaluate()
            log.info("epoch %d eval test_acc=%.4f val_cv=%.4f", epoch, acc, val_acc or -1)
            torch.save(
                {"model": model.online_encoder.state_dict(), "epoch": epoch},
                out / "checkpoint.pt",
            )

    test_acc, val_acc, representations, labels = evaluate()
    torch.save(
        {"model": model.online_encoder.state_dict(), "epoch": cfg["epochs"]},
        out / "checkpoint.pt",
    )
    if save_embeddings:
        torch.save(
            {"embeddings": representations.cpu(), "labels": labels.cpu()},
            out / "embeddings.pt",
        )

    metrics = {
        "dataset": cfg["dataset"],
        "method": cfg["method"],
        "augmentation": cfg["augmentation"],
        "seed": seed,
        "data_seed": data_seed,
        "epochs": cfg["epochs"],
        "best_val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "embedding_dimension": representation_size,
        "final_training_loss": last_loss,
        "device": str(device_obj),
    }
    _write_json(out / "metrics.json", metrics)
    log.info("FINAL %s", json.dumps(metrics))
    return metrics
