import os
import random

import dgl
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected


# ============================================================
# Seed
# ============================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data Loader
# ============================================================
def load_data(data_dir, dataset_name, anomaly_std=None, anomaly_alpha=None):
    name = dataset_name.lower()

    # ============================================================
    # 1. Questions: .npz format
    # 文件路径: ../datasets/questions.npz
    # ============================================================
    if name in ["questions", "question"]:
        path = os.path.join(data_dir, "questions.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")

        raw = np.load(path, allow_pickle=True)
        print("Questions npz keys:", raw.files)

        features = raw["node_features"]
        labels = raw["node_labels"].flatten()
        edges = raw["edges"]

        if sp.issparse(features):
            features = features.toarray()

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        edge_index = torch.from_numpy(edges).long()
        if edge_index.size(0) != 2:
            edge_index = edge_index.t().contiguous()

        edge_index = to_undirected(edge_index, num_nodes=features.shape[0])
        labels = np.array([1 if l > 0 else 0 for l in labels], dtype=np.int64)

        print(
            f"Loaded Questions: nodes={features.shape[0]} "
            f"edges={edge_index.shape[1]} anomaly={(labels == 1).sum()}"
        )

    # ============================================================
    # 2. T-Finance: DGL graph format
    # 文件路径: ../datasets/tfinance
    # ============================================================
    elif name in ["tfinance", "t-finance", "t_finance"]:
        path_candidates = [
            os.path.join(data_dir, "tfinance"),
            os.path.join(data_dir, "T-Finance"),
            os.path.join(data_dir, "t-finance"),
        ]

        path = None
        for p in path_candidates:
            if os.path.exists(p):
                path = p
                break

        if path is None:
            raise FileNotFoundError(
                f"T-Finance dataset not found. Tried: {path_candidates}"
            )

        graph_list, _ = load_graphs(path)
        graph = graph_list[0]

        features = graph.ndata["feature"]

        raw_label = graph.ndata["label"]
        if raw_label.dim() > 1:
            labels = raw_label.argmax(1)
        else:
            labels = raw_label

        # 可选：保留 anomaly_std 操作，默认不用
        if anomaly_std is not None:
            feat = features.cpu().numpy() if torch.is_tensor(features) else features
            label_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

            anomaly_id = np.where(label_np == 1)[0]
            feat = (feat - np.average(feat, axis=0)) / (np.std(feat, axis=0) + 1e-12)
            feat[anomaly_id] = anomaly_std * feat[anomaly_id]

            features = torch.tensor(feat, dtype=torch.float32)

        # 可选：保留 anomaly_alpha 操作，默认不用
        if anomaly_alpha is not None:
            feat = features.cpu().numpy() if torch.is_tensor(features) else features
            label_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

            anomaly_id = list(np.where(label_np == 1)[0])
            normal_id = list(np.where(label_np == 0)[0])

            diff = int(anomaly_alpha * len(label_np) - len(anomaly_id))

            if diff > 0 and len(normal_id) >= diff and len(anomaly_id) > 0:
                new_id = random.sample(normal_id, diff)
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    feat[idx] = feat[aid]
                    label_np[idx] = 1

            features = torch.tensor(feat, dtype=torch.float32)
            labels = torch.tensor(label_np, dtype=torch.long)

        if torch.is_tensor(features):
            features = features.cpu().numpy()

        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        graph = dgl.to_bidirected(graph)
        graph = graph.remove_self_loop()

        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0).long()
        edge_index = to_undirected(edge_index, num_nodes=features.shape[0])

        labels = np.array([1 if l > 0 else 0 for l in labels], dtype=np.int64)

        print(
            f"Loaded T-Finance: nodes={features.shape[0]} "
            f"edges={edge_index.shape[1]} anomaly={(labels == 1).sum()}"
        )

    # ============================================================
    # 3. AmazonFull / YelpChiFull: DGL graph format
    # ============================================================
    elif dataset_name in ["AmazonFull", "YelpChiFull"]:
        path = os.path.join(data_dir, dataset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")

        graph_list = load_graphs(path)[0]
        graph = graph_list[0]

        features = graph.ndata["feature"]
        labels = graph.ndata["label"].cpu().numpy()

        graph = dgl.to_bidirected(graph)
        graph = graph.remove_self_loop()

        nx_graph = dgl.to_networkx(graph)
        adj = nx.to_scipy_sparse_array(nx_graph)

        if sp.issparse(features):
            features = features.toarray()

        if torch.is_tensor(features):
            features = features.cpu().numpy()

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        edge_index, _ = from_scipy_sparse_matrix(adj)
        edge_index = to_undirected(edge_index)

        labels = np.array([1 if l > 0 else 0 for l in labels], dtype=np.int64)
        num_pos = (labels == 1).sum()

        print(
            f"Loaded {dataset_name}: nodes={features.shape[0]} "
            f"edges={edge_index.shape[1]} fake={num_pos}"
        )

    # ============================================================
    # 4. Other datasets: .mat format
    # ============================================================
    else:
        path = os.path.join(data_dir, dataset_name + ".mat")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")

        mat = sio.loadmat(path)

        adj_key = [k for k in ["Network", "net", "homo", "A"] if k in mat][0]
        feat_key = [k for k in ["Attributes", "features", "attr", "X"] if k in mat][0]
        label_key = [k for k in ["Label", "label", "labels"] if k in mat][0]

        adj = sp.coo_matrix(mat[adj_key])
        feat = mat[feat_key]

        if sp.issparse(feat):
            feat = feat.toarray()

        labels = mat[label_key].flatten()
        labels = np.array([1 if l > 0 else 0 for l in labels], dtype=np.int64)

        scaler = StandardScaler()
        features = scaler.fit_transform(feat)

        edge_index, _ = from_scipy_sparse_matrix(adj)
        edge_index = to_undirected(edge_index)

        print(
            f"Loaded mat: nodes={features.shape[0]} "
            f"edges={edge_index.shape[1]} fake={(labels == 1).sum()}"
        )

    pyg_data = Data(
        x=torch.FloatTensor(features),
        edge_index=edge_index.long(),
        y=torch.LongTensor(labels),
    )

    return pyg_data


# ============================================================
# Split
# ============================================================
def split_mask(labels, seed=42):
    idx = np.arange(len(labels))
    y = labels.cpu().numpy()

    train_val, test = train_test_split(
        idx,
        test_size=0.3,
        stratify=y,
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.1,
        stratify=y[train_val],
        random_state=seed,
    )

    def build(index):
        mask = torch.zeros(len(labels), dtype=torch.bool)
        mask[index] = True
        return mask

    return build(train), build(val), build(test)
