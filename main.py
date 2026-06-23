import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import NeighborLoader

from model import StabilityGraphAD
from utils import load_data, seed_everything, split_mask


# ============================================================
# Hyperparameters
# ============================================================
@dataclass
class Config:
    dataset_name: str = "Amazon"
    data_dir: str = "../datasets"
    seeds: tuple = (42, 100, 2023, 2026, 999)

    hidden_dim: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 15

    train_batch_size: int = 512
    eval_batch_size: int = 1024
    num_neighbors: tuple = (25, 10)
    num_workers: int = 0

    # 与原代码保持一致：五个 seed 使用同一份数据划分，只改变模型随机性。
    # 如果想让每个 seed 也重新划分数据，把 split_seed 改成 None。
    split_seed: int | None = 42


# ============================================================
# Train Layer (适配 Mini-batch Mini-Graph)
# ============================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.batch_size
        if batch_size == 0:
            continue

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch_size)

        instability = out["instability"]
        entropy = out["entropy"]
        response = out["response"]
        discrepancy = out["discrepancy"]

        # 低 discrepancy 节点 = 更可能正常
        low_thresh = torch.quantile(discrepancy.detach(), 0.3)
        normal_mask = discrepancy <= low_thresh

        # 防止极端 batch 没有节点
        if normal_mask.sum() < 10:
            normal_mask = discrepancy <= discrepancy.median()

        # 只稳定 pseudo-normal nodes
        loss_stable = instability[normal_mask].mean()

        high_thresh = torch.quantile(instability.detach(), 0.8)
        low_thresh_inst = torch.quantile(instability.detach(), 0.2)

        high_mask = instability >= high_thresh
        low_mask_inst = instability <= low_thresh_inst

        if high_mask.sum() > 5 and low_mask_inst.sum() > 5:
            loss_separation = F.relu(
                1.0
                - instability[high_mask].mean()
                + instability[low_mask_inst].mean()
            )
        else:
            loss_separation = torch.tensor(0.0, device=device)

        # 正常节点应该 response 更确定
        loss_entropy = entropy[normal_mask].mean()
        std_loss = torch.mean(F.relu(1.0 - response.std(dim=0)))

        loss = (
            0.05 * loss_stable
            + 0.001 * loss_entropy
            + 0.01 * loss_separation
            + 0.005 * std_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_nodes += batch_size

    return total_loss / max(total_nodes, 1)


# ============================================================
# Evaluate Layer (流式推理，避免大图全图 OOM)
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

    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    return auc, ap


def build_loaders(pyg_data, train_mask, cfg):
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


def run_one_seed(seed, pyg_data, cfg, device):
    seed_everything(seed)

    split_seed = seed if cfg.split_seed is None else cfg.split_seed
    train_mask, val_mask, test_mask = split_mask(pyg_data.y, split_seed)

    train_loader, eval_loader = build_loaders(pyg_data, train_mask, cfg)

    model = StabilityGraphAD(
        in_dim=pyg_data.x.size(1),
        hid_dim=cfg.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_auc = 0.0
    best_test_auc = 0.0
    best_test_ap = 0.0
    patience_count = 0

    seed_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_auc, _ = evaluate_large_graph(model, eval_loader, val_mask, pyg_data.y, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc, best_test_ap = evaluate_large_graph(
                model,
                eval_loader,
                test_mask,
                pyg_data.y,
                device,
            )
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Train Loss {loss:.4f} | ValAUC {val_auc:.4f}")

        if patience_count > cfg.patience:
            break

    seed_time = time.perf_counter() - seed_start
    return best_test_auc, best_test_ap, seed_time


# ============================================================
# Main 入口
# ============================================================
def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Streaming Loader initialized for: {cfg.dataset_name} ...")
    pyg_data = load_data(cfg.data_dir, cfg.dataset_name)

    aucs = []
    aps = []
    seed_times = []

    # 从训练开始计时：不包含数据读取，只统计 5 个 seed 的训练/验证/测试过程。
    train_start = time.perf_counter()

    for seed in cfg.seeds:
        print(f"\n================ Seed {seed} ================")
        test_auc, test_ap, seed_time = run_one_seed(seed, pyg_data, cfg, device)

        print(
            f"[Seed {seed}] AUROC {test_auc:.4f} | "
            f"AUPRC {test_ap:.4f} | Time {seed_time:.2f}s"
        )
        aucs.append(test_auc)
        aps.append(test_ap)
        seed_times.append(seed_time)
# ============================================================
# test time consume

    # seed=42
    # print(f"\n================ Seed {seed} ================")
    # test_auc, test_ap, seed_time = run_one_seed(seed, pyg_data, cfg, device)
    #
    # print(
    #     f"[Seed {seed}] AUROC {test_auc:.4f} | "
    #     f"AUPRC {test_ap:.4f} | Time {seed_time:.2f}s"
    # )
    # aucs.append(test_auc)
    # aps.append(test_ap)
    # seed_times.append(seed_time)
# ============================================================

    total_train_time = time.perf_counter() - train_start

    print("\n================ FINAL RESULTS ================")
    print(f"AUROC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"AUPRC: {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"Total Train/Eval Time: {total_train_time:.2f}s")
    print(f"Avg Time per Seed: {np.mean(seed_times):.2f}s")


if __name__ == "__main__":
    main()
