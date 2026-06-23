import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.utils import add_self_loops, degree


# ============================================================
# Graph Normalize (适配子图：传入当前子图的边和节点总数)
# ============================================================
def normalize_graph(edge_index, num_nodes):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes).clamp(min=1)
    deg_inv = deg.pow(-0.5)
    norm = deg_inv[row] * deg_inv[col]
    return edge_index, norm


# ============================================================
# Spectral Response Extractor (大图修正版)
# ============================================================
class SpectralResponse(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
        self.energy_proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index
        h = self.encoder(x)

        low = (h[row] + h[col]) * 0.5
        band = torch.abs(h[row] - h[col])

        low_e = self.energy_proj(low)
        band_e = self.energy_proj(band)

        low_e = F.normalize(low_e, p=2, dim=1)
        band_e = F.normalize(band_e, p=2, dim=1)

        spectral_conflict = 1.0 - (low_e * band_e).sum(dim=1)

        node_conflict = scatter_mean(
            spectral_conflict,
            row,
            dim=0,
            dim_size=num_nodes,
        )
        return h, node_conflict


# ============================================================
# Response Sampler (大图修正版)
# ============================================================
class ResponseSampler(nn.Module):
    def __init__(self, hid_dim, num_samples=16, drop_rate=0.3):
        super().__init__()
        self.num_samples = num_samples
        self.drop_rate = drop_rate
        self.lin = nn.Linear(hid_dim, hid_dim)

    def propagate_once(self, h, edge_index, norm, num_nodes):
        row, col = edge_index
        edge_mask = torch.rand(row.size(0), device=h.device) > self.drop_rate

        row_drop = row[edge_mask]
        col_drop = col[edge_mask]
        norm_drop = norm[edge_mask]

        msg = h[col_drop] * norm_drop.unsqueeze(1)

        out = scatter_mean(
            msg,
            row_drop,
            dim=0,
            dim_size=num_nodes,
        )
        out = F.relu(self.lin(out))
        return out

    def forward(self, h, edge_index, norm, num_nodes):
        responses = []
        for _ in range(self.num_samples):
            out = self.propagate_once(h, edge_index, norm, num_nodes)
            responses.append(out)

        responses = torch.stack(responses, dim=0)
        mean_response = responses.mean(dim=0)

        instability = responses.var(dim=0).mean(dim=1)
        entropy = -(
            F.softmax(responses, dim=-1)
            * F.log_softmax(responses + 1e-6, dim=-1)
        ).sum(dim=-1).mean(dim=0)

        return mean_response, instability, entropy


# ============================================================
# Relative Local Instability & response-neighborhood discrepancy (大图修正版)
# ============================================================
def compute_relative_instability(instability, edge_index, num_nodes):
    row, col = edge_index
    neigh_mean = scatter_mean(instability[col], row, dim=0, dim_size=num_nodes)
    relative_dev = torch.abs(instability - neigh_mean)
    return relative_dev


def compute_response_discrepancy(response, edge_index, num_nodes):
    row, col = edge_index
    response = F.normalize(response, p=2, dim=1)
    sim = (response[row] * response[col]).sum(dim=1)
    discrepancy = scatter_mean(1.0 - sim, row, dim=0, dim_size=num_nodes)
    return discrepancy


# ============================================================
# Full Model (适配大图局部子图流)
# ============================================================
class StabilityGraphAD(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.spectral = SpectralResponse(in_dim, hid_dim)
        self.sampler = ResponseSampler(hid_dim)

    def forward(self, x, edge_index, batch_size):
        """
        batch_size: 当前 batch 对应的小批量中核心目标节点的数量。
        PyG 采样机制会把目标节点排在最前。
        """
        num_nodes = x.size(0)

        # 在子图内部动态计算归一化系数 norm，防止大图切碎后的越界与不匹配。
        edge_index, norm = normalize_graph(edge_index, num_nodes)

        h, spectral_conflict = self.spectral(x, edge_index, num_nodes)
        response, instability, entropy = self.sampler(h, edge_index, norm, num_nodes)
        local_dev = compute_relative_instability(instability, edge_index, num_nodes)
        discrepancy = compute_response_discrepancy(response, edge_index, num_nodes)

        score = (
            0.60 * spectral_conflict
            + 0.15 * instability
            + 0.25 * local_dev
        )

        # 前向传播计算涉及采样子图邻域，但训练/评估只返回当前 batch 的目标节点输出。
        return {
            "score": score[:batch_size],
            "spectral": spectral_conflict[:batch_size],
            "instability": instability[:batch_size],
            "local_dev": local_dev[:batch_size],
            "discrepancy": discrepancy[:batch_size],
            "entropy": entropy[:batch_size],
            "response": response[:batch_size],
        }
