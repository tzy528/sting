import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import NeighborLoader

from model import StabilityGraphAD
from utils import load_data, seed_everything, split_mask


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    data_dir: str = "../datasets"
    dataset_names: Tuple[str, ...] = (
        "Amazon",
        "YelpChi",
        "Questions",
        "Weibo",
        "AmazonFull",
        "YelpChiFull",
    )
    seeds: Tuple[int, ...] = (42, 100, 2023, 2026, 999)

    hidden_dim: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 15

    train_batch_size: int = 512
    eval_batch_size: int = 1024
    num_neighbors: Tuple[int, ...] = (25, 10)
    num_workers: int = 0

    num_samples: int = 16
    drop_rate: float = 0.3

    # Final anomaly score: score = w_conf * s_conf + w_ins * s_ins.
    score_conf_weight: float = 0.60
    score_ins_weight: float = 0.40

    # Loss weights. Entropy is removed. DEV is used as a loss, not final score.
    w_stable: float = 0.05
    w_dev: float = 0.02
    w_sep: float = 0.01
    w_std: float = 0.005

    # Pseudo-genuine nodes are selected by low response-neighborhood discrepancy.
    pseudo_genuine_ratio: float = 0.15
    min_pseudo_genuine: int = 10

    # Keep the same split for all seeds by default, only changing model randomness.
    # Set split_seed=None if every seed should also generate a new split.
    split_seed: int | None = 42

    output_dir: str = "./sting_modular_results"
    device: str = "cuda:1"


# ============================================================
# IO helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Loaders
# ============================================================
def build_loaders(pyg_data, train_mask, cfg: Config):
    train_loader = NeighborLoader(
        pyg_data,
        num_neighbors=list(cfg.num_neighbors),
        batch_size=cfg.train_batch_size,
        input_nodes=train_mask,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    eval_loader = NeighborLoader(
        pyg_data,
        num_neighbors=list(cfg.num_neighbors),
        batch_size=cfg.eval_batch_size,
        input_nodes=None,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, eval_loader


# ============================================================
# Training
# ============================================================
def train_epoch(model, loader, optimizer, labels, device, cfg: Config):
    model.train()
    total_loss = 0.0
    total_nodes = 0
    pg_correct = 0
    pg_total = 0

    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.batch_size
        if batch_size == 0:
            continue

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch_size)

        instability = out["instability"]
        local_dev = out["local_dev"]
        response = out["response"]
        response_discrepancy = out["response_discrepancy"]

        # Low response-neighborhood discrepancy nodes are treated as pseudo-genuine nodes.
        pseudo_thresh = torch.quantile(response_discrepancy.detach(), cfg.pseudo_genuine_ratio)
        pseudo_mask = response_discrepancy <= pseudo_thresh

        if pseudo_mask.sum() < cfg.min_pseudo_genuine:
            pseudo_mask = response_discrepancy <= response_discrepancy.median()

        # PGAcc uses labels only for analysis/logging, not for optimization.
        target_nid = batch.n_id[:batch_size]
        selected_labels = labels[target_nid[pseudo_mask.detach().cpu()]].cpu()
        if selected_labels.numel() > 0:
            pg_correct += int((selected_labels == 0).sum().item())
            pg_total += int(selected_labels.numel())

        loss_stable = instability[pseudo_mask].mean()
        loss_dev = local_dev[pseudo_mask].mean()

        # Auxiliary losses are kept configurable. Set w_sep=0 and w_std=0 to disable them.
        high_thresh = torch.quantile(instability.detach(), 0.8)
        low_thresh = torch.quantile(instability.detach(), 0.2)
        high_mask = instability >= high_thresh
        low_mask = instability <= low_thresh
        if high_mask.sum() > 5 and low_mask.sum() > 5:
            loss_sep = F.relu(1.0 - instability[high_mask].mean() + instability[low_mask].mean())
        else:
            loss_sep = torch.tensor(0.0, device=device)

        loss_std = torch.mean(F.relu(1.0 - response.std(dim=0)))

        loss = (
            cfg.w_stable * loss_stable
            + cfg.w_dev * loss_dev
            + cfg.w_sep * loss_sep
            + cfg.w_std * loss_std
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_nodes += batch_size

    pg_acc = pg_correct / pg_total if pg_total > 0 else 0.0
    return total_loss / max(total_nodes, 1), pg_acc, pg_total


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_large_graph(model, loader, mask, labels, device):
    model.eval()
    num_nodes = labels.size(0)
    full_score = torch.zeros(num_nodes)
    full_seen = torch.zeros(num_nodes, dtype=torch.bool)

    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.batch_size
        if batch_size == 0:
            continue

        out = model(batch.x, batch.edge_index, batch_size)
        raw_score = out["score"].detach().cpu()

        target_nid = batch.n_id[:batch_size].cpu()
        full_score[target_nid] = raw_score
        full_seen[target_nid] = True

    if not full_seen.all():
        print(f"Warning: {(~full_seen).sum().item()} nodes were not evaluated.")

    med = full_score[full_seen].median()
    mad = (full_score[full_seen] - med).abs().median() + 1e-6
    normalized_score = (full_score - med) / mad

    s = normalized_score[mask].numpy()
    y = labels[mask].cpu().numpy()
    return roc_auc_score(y, s), average_precision_score(y, s)


# ============================================================
# Experiment Runner
# ============================================================
def run_one_seed(seed: int, pyg_data, cfg: Config, device):
    seed_everything(seed)
    split_seed = seed if cfg.split_seed is None else cfg.split_seed
    train_mask, val_mask, test_mask = split_mask(pyg_data.y, split_seed)
    train_loader, eval_loader = build_loaders(pyg_data, train_mask, cfg)

    model = StabilityGraphAD(
        in_dim=pyg_data.x.size(1),
        hid_dim=cfg.hidden_dim,
        num_samples=cfg.num_samples,
        drop_rate=cfg.drop_rate,
        score_conf_weight=cfg.score_conf_weight,
        score_ins_weight=cfg.score_ins_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_auc = 0.0
    best_test_auc = 0.0
    best_test_ap = 0.0
    best_pg_acc = 0.0
    best_pg_num = 0
    best_epoch = -1
    patience_count = 0
    seed_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        loss, pg_acc, pg_num = train_epoch(model, train_loader, optimizer, pyg_data.y, device, cfg)
        val_auc, _ = evaluate_large_graph(model, eval_loader, val_mask, pyg_data.y, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc, best_test_ap = evaluate_large_graph(model, eval_loader, test_mask, pyg_data.y, device)
            best_pg_acc = pg_acc
            best_pg_num = pg_num
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | Loss {loss:.4f} | ValAUC {val_auc:.4f} | "
                f"PGAcc {pg_acc:.4f} | PGNum {pg_num}"
            )

        if patience_count > cfg.patience:
            break

    seed_time = time.perf_counter() - seed_start
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_auc": best_test_auc,
        "test_ap": best_test_ap,
        "pg_acc": best_pg_acc,
        "pg_num": best_pg_num,
        "time_sec": seed_time,
    }


def run_one_dataset(dataset_name: str, cfg: Config, device):
    print(f"\n================ Dataset: {dataset_name} ================")
    pyg_data = load_data(cfg.data_dir, dataset_name)

    rows = []
    for seed in cfg.seeds:
        print(f"\n========== Seed {seed} ==========")
        result = run_one_seed(seed, pyg_data, cfg, device)
        result["dataset"] = dataset_name
        rows.append(result)
        print(
            f"[Seed {seed}] AUROC {result['test_auc']:.4f} | "
            f"AUPRC {result['test_ap']:.4f} | PGAcc {result['pg_acc']:.4f} | "
            f"Time {result['time_sec']:.2f}s"
        )

    aucs = np.array([r["test_auc"] for r in rows], dtype=float)
    aps = np.array([r["test_ap"] for r in rows], dtype=float)
    pg_accs = np.array([r["pg_acc"] for r in rows], dtype=float)

    summary = {
        "dataset": dataset_name,
        "n_seeds": len(rows),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=0)),
        "ap_mean": float(aps.mean()),
        "ap_std": float(aps.std(ddof=0)),
        "pg_acc_mean": float(pg_accs.mean()),
        "pg_acc_std": float(pg_accs.std(ddof=0)),
    }

    write_csv(os.path.join(cfg.output_dir, f"{dataset_name}_runs.csv"), rows)
    return rows, summary


# ============================================================
# Main
# ============================================================
def main():
    cfg = Config()
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    ensure_dir(cfg.output_dir)
    print(f"Device: {device}")
    print(f"Datasets: {cfg.dataset_names}")
    print(f"Seeds: {cfg.seeds}")
    print(f"Pseudo-genuine ratio: {cfg.pseudo_genuine_ratio}")

    all_rows = []
    summary_rows = []
    total_start = time.perf_counter()

    for dataset_name in cfg.dataset_names:
        try:
            rows, summary = run_one_dataset(dataset_name, cfg, device)
            all_rows.extend(rows)
            summary_rows.append(summary)
        except Exception as exc:
            print(f"[ERROR] Dataset {dataset_name} failed: {repr(exc)}")
            summary_rows.append({
                "dataset": dataset_name,
                "n_seeds": 0,
                "auc_mean": "ERROR",
                "auc_std": "ERROR",
                "ap_mean": "ERROR",
                "ap_std": "ERROR",
                "pg_acc_mean": "ERROR",
                "pg_acc_std": repr(exc),
            })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(os.path.join(cfg.output_dir, "all_dataset_runs.csv"), all_rows)
    write_csv(os.path.join(cfg.output_dir, "all_dataset_summary.csv"), summary_rows)

    total_time = time.perf_counter() - total_start
    print("\n================ FINAL SUMMARY ================")
    for s in summary_rows:
        print(s)
    print(f"Total time: {total_time:.2f}s")
    print(f"Saved results to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
