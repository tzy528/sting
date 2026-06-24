import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.utils import add_self_loops, degree


# ============================================================
# Graph Normalize (for sampled subgraphs)
# ============================================================
def normalize_graph(edge_index, num_nodes):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes).clamp(min=1)
    deg_inv = deg.pow(-0.5)
    norm = deg_inv[row] * deg_inv[col]
    return edge_index, norm


# ============================================================
# Spectral / Structural Response Conflict Extractor
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

        smooth_response = (h[row] + h[col]) * 0.5
        variation_response = torch.abs(h[row] - h[col])

        smooth_e = F.normalize(self.energy_proj(smooth_response), p=2, dim=1)
        variation_e = F.normalize(self.energy_proj(variation_response), p=2, dim=1)

        edge_conflict = 1.0 - (smooth_e * variation_e).sum(dim=1)
        node_conflict = scatter_mean(edge_conflict, row, dim=0, dim_size=num_nodes)
        return h, node_conflict


# ============================================================
# Perturbation-Induced Structural Response Sampler
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
        out = scatter_mean(msg, row_drop, dim=0, dim_size=num_nodes)
        out = F.relu(self.lin(out))
        return out

    def forward(self, h, edge_index, norm, num_nodes):
        responses = []
        for _ in range(self.num_samples):
            responses.append(self.propagate_once(h, edge_index, norm, num_nodes))

        responses = torch.stack(responses, dim=0)
        mean_response = responses.mean(dim=0)
        instability = responses.var(dim=0).mean(dim=1)
        return mean_response, instability


# ============================================================
# Auxiliary Response Statistics
# ============================================================
def compute_relative_instability(instability, edge_index, num_nodes):
    row, col = edge_index
    neigh_mean = scatter_mean(instability[col], row, dim=0, dim_size=num_nodes)
    return torch.abs(instability - neigh_mean)


def compute_response_neighborhood_discrepancy(response, edge_index, num_nodes):
    row, col = edge_index
    response = F.normalize(response, p=2, dim=1)
    sim = (response[row] * response[col]).sum(dim=1)
    discrepancy = scatter_mean(1.0 - sim, row, dim=0, dim_size=num_nodes)
    return discrepancy


# ============================================================
# Full Model
# ============================================================
class StabilityGraphAD(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_samples=16,
        drop_rate=0.3,
        score_conf_weight=0.60,
        score_ins_weight=0.40,
    ):
        super().__init__()
        self.spectral = SpectralResponse(in_dim, hid_dim)
        self.sampler = ResponseSampler(hid_dim, num_samples=num_samples, drop_rate=drop_rate)
        self.score_conf_weight = score_conf_weight
        self.score_ins_weight = score_ins_weight

    def forward(self, x, edge_index, batch_size):
        """
        batch_size: number of target nodes in the sampled mini-batch.
        PyG NeighborLoader places target nodes at the beginning of the sampled subgraph.
        """
        num_nodes = x.size(0)
        edge_index, norm = normalize_graph(edge_index, num_nodes)

        h, spectral_conflict = self.spectral(x, edge_index, num_nodes)
        response, instability = self.sampler(h, edge_index, norm, num_nodes)
        local_dev = compute_relative_instability(instability, edge_index, num_nodes)
        response_discrepancy = compute_response_neighborhood_discrepancy(
            response, edge_index, num_nodes
        )

        # Final anomaly score only uses two detection signals:
        # 1) structural response conflict; 2) perturbation-induced instability.
        score = (
            self.score_conf_weight * spectral_conflict
            + self.score_ins_weight * instability
        )

        return {
            "score": score[:batch_size],
            "spectral": spectral_conflict[:batch_size],
            "instability": instability[:batch_size],
            "local_dev": local_dev[:batch_size],
            "response_discrepancy": response_discrepancy[:batch_size],
            "response": response[:batch_size],
        }
