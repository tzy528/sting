import os
import random
from typing import Iterable, Optional

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
# Data Loading Helpers
# ============================================================
def _find_existing_path(candidates: Iterable[str]) -> Optional[str]:
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _extract_npz_array(raw, keys):
    for key in keys:
        if key in raw:
            return raw[key]
    raise KeyError(f"Cannot find any of {keys}; available keys: {list(raw.keys())}")


def _standardize_features(features):
    if sp.issparse(features):
        features = features.toarray()
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    return StandardScaler().fit_transform(features)


def _binarize_labels(labels):
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels = np.asarray(labels).flatten()
    return np.array([1 if label > 0 else 0 for label in labels], dtype=np.int64)


def _load_questions(data_dir):
    candidates = [
        os.path.join(data_dir, "questions.npz"),
        os.path.join(data_dir, "Questions.npz"),
        os.path.join(data_dir, "questions.npy"),
        os.path.join(data_dir, "Questions.npy"),
    ]
    path = _find_existing_path(candidates)
    if path is None:
        raise FileNotFoundError(f"Questions dataset not found. Tried: {candidates}")

    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.lib.npyio.NpzFile):
        print("Questions keys:", raw.files)
        features = _extract_npz_array(raw, ["node_features", "features", "feature", "X", "x"])
        labels = _extract_npz_array(raw, ["node_labels", "labels", "label", "y"])
        edges = _extract_npz_array(raw, ["edges", "edge_index"])
    else:
        raw = raw.item() if isinstance(raw, np.ndarray) and raw.shape == () else raw
        print("Questions dict keys:", list(raw.keys()))
        features = _extract_npz_array(raw, ["node_features", "features", "feature", "X", "x"])
        labels = _extract_npz_array(raw, ["node_labels", "labels", "label", "y"])
        edges = _extract_npz_array(raw, ["edges", "edge_index"])

    features = _standardize_features(features)
    labels = _binarize_labels(labels)

    edge_index = torch.from_numpy(edges).long()
    if edge_index.size(0) != 2:
        edge_index = edge_index.t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=features.shape[0])

    print(
        f"Loaded Questions: nodes={features.shape[0]} "
        f"edges={edge_index.shape[1]} anomaly={(labels == 1).sum()}"
    )
    return features, edge_index, labels


def _load_dgl_graph(data_dir, dataset_name):
    candidates = [
        os.path.join(data_dir, dataset_name),
        os.path.join(data_dir, dataset_name.lower()),
        os.path.join(data_dir, dataset_name.upper()),
        os.path.join(data_dir, dataset_name.replace("Full", "full")),
    ]
    path = _find_existing_path(candidates)
    if path is None:
        raise FileNotFoundError(f"DGL dataset not found. Tried: {candidates}")

    graph_list = load_graphs(path)[0]
    graph = graph_list[0]

    features = graph.ndata["feature"]
    labels = graph.ndata["label"]
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    graph = dgl.to_bidirected(graph)
    graph = graph.remove_self_loop()

    nx_graph = dgl.to_networkx(graph)
    adj = nx.to_scipy_sparse_array(nx_graph)

    features = _standardize_features(features)
    labels = _binarize_labels(labels)
    edge_index, _ = from_scipy_sparse_matrix(adj)
    edge_index = to_undirected(edge_index, num_nodes=features.shape[0])

    print(
        f"Loaded {dataset_name}: nodes={features.shape[0]} "
        f"edges={edge_index.shape[1]} anomaly={(labels == 1).sum()}"
    )
    return features, edge_index, labels


def _load_mat_graph(data_dir, dataset_name):
    candidates = [
        os.path.join(data_dir, dataset_name + ".mat"),
        os.path.join(data_dir, dataset_name.lower() + ".mat"),
        os.path.join(data_dir, dataset_name.upper() + ".mat"),
    ]
    path = _find_existing_path(candidates)
    if path is None:
        raise FileNotFoundError(f"MAT dataset not found. Tried: {candidates}")

    mat = sio.loadmat(path)
    adj_key = [k for k in ["Network", "net", "homo", "A"] if k in mat][0]
    feat_key = [k for k in ["Attributes", "features", "attr", "X"] if k in mat][0]
    label_key = [k for k in ["Label", "label", "labels", "y"] if k in mat][0]

    adj = sp.coo_matrix(mat[adj_key])
    features = _standardize_features(mat[feat_key])
    labels = _binarize_labels(mat[label_key])

    edge_index, _ = from_scipy_sparse_matrix(adj)
    edge_index = to_undirected(edge_index, num_nodes=features.shape[0])

    print(
        f"Loaded {dataset_name}: nodes={features.shape[0]} "
        f"edges={edge_index.shape[1]} anomaly={(labels == 1).sum()}"
    )
    return features, edge_index, labels


# ============================================================
# Public Data Loader
# ============================================================
def load_data(data_dir: str, dataset_name: str) -> Data:
    name = dataset_name.lower()

    if name in ["questions", "question"]:
        features, edge_index, labels = _load_questions(data_dir)
    elif name in ["amazonfull", "yelpchifull"]:
        features, edge_index, labels = _load_dgl_graph(data_dir, dataset_name)
    else:
        features, edge_index, labels = _load_mat_graph(data_dir, dataset_name)

    return Data(
        x=torch.FloatTensor(features),
        edge_index=edge_index.long(),
        y=torch.LongTensor(labels),
    )


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
