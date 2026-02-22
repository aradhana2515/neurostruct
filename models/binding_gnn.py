"""
binding_gnn.py
==============
Graph Attention Network (GAT) for binding affinity prediction at
neurotransmitter receptor binding sites.

Architecture overview:
  Input graph: residue contact graph
    Nodes: per-residue features (one-hot AA + biochemical props + optional ESM embedding)
    Edges: Cα–Cα contacts < 8Å, weighted by distance

  Network:
    1. Input projection  : Linear(node_dim → hidden_dim)
    2. GAT layers × N    : GATv2Conv with multi-head attention
                           Edge distances used as edge features
    3. Residue dropout   : stochastic depth for regularization
    4. Global pooling    : Attention-weighted global mean (virtual node readout)
    5. MLP head          : hidden_dim → 64 → 1  (ΔG regression)

The GAT attention weights serve dual purpose:
  a) They improve prediction by weighting important residues
  b) They can be visualized to identify binding hotspots

Usage:
  from models.binding_gnn import BindingGNN

  model = BindingGNN(node_dim=25, hidden_dim=256, num_layers=4, heads=4)
  out = model(data)        # returns (batch_size,) predicted ΔG values
  attn = model.get_attention_weights(data)  # residue importance scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    global_add_pool,
)
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualGATLayer(nn.Module):
    """
    Single GAT layer with:
      - GATv2Conv (improved attention that depends on both source & target)
      - LayerNorm
      - Residual skip connection
      - Optional edge feature injection (Cα distances)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = heads
        self.out_dim = out_dim

        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
            share_weights=False,
        )

        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(out_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        return_attention: bool = False,
    ):
        if return_attention:
            out, attn = self.conv(x, edge_index, edge_attr=edge_attr,
                                  return_attention_weights=True)
        else:
            out = self.conv(x, edge_index, edge_attr=edge_attr)
            attn = None

        out = self.act(out)
        out = self.drop(out)
        out = self.norm(out + self.skip(x))

        return (out, attn) if return_attention else out


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over nodes → single graph embedding.
    Score each node with a small MLP, softmax over graph, weighted sum.
    Uses manual per-graph softmax for full PyG version compatibility.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
          pooled  : (B, hidden_dim) — one vector per graph
          weights : (N,)            — per-node attention scores
        """
        gate_scores = self.gate(x).squeeze(-1)      # (N,)

        # Manual per-graph softmax — compatible with all PyG versions
        weights = torch.zeros_like(gate_scores)
        for g in batch.unique():
            mask = (batch == g)
            weights[mask] = torch.softmax(gate_scores[mask], dim=0)

        pooled = global_add_pool(weights.unsqueeze(-1) * x, batch)  # (B, hidden_dim)
        return pooled, weights


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class BindingGNN(nn.Module):
    """
    Full Graph Attention Network for binding affinity regression.

    Parameters
    ----------
    node_dim    : Dimensionality of input node features
                  (25 for biochemical only; 1280+ when ESM embeddings appended)
    hidden_dim  : Internal representation size
    num_layers  : Number of GAT layers
    heads       : Number of attention heads per GAT layer
    dropout     : Dropout probability
    edge_dim    : Dimensionality of edge features (1 = distance)
    """

    def __init__(
        self,
        node_dim: int = 25,
        hidden_dim: int = 256,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.15,
        edge_dim: int = 1,
    ):
        super().__init__()
        self.node_dim   = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Edge feature projection
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self._edge_dim_internal = 16

        # GAT layers
        self.gat_layers = nn.ModuleList([
            ResidualGATLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=self._edge_dim_internal,
            )
            for _ in range(num_layers)
        ])

        # Stochastic depth
        self.layer_drop_probs = [
            i * 0.05 / max(num_layers - 1, 1) for i in range(num_layers)
        ]

        # Readout
        self.pool = AttentionPooling(hidden_dim)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> Tensor:
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch      = getattr(data, "batch",
                             torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        x = self.input_proj(x)
        e = self.edge_proj(edge_attr)

        for layer, drop_prob in zip(self.gat_layers, self.layer_drop_probs):
            if self.training and torch.rand(1).item() < drop_prob:
                continue
            x = layer(x, edge_index, edge_attr=e)

        graph_emb, _ = self.pool(x, batch)
        out = self.head(graph_emb).squeeze(-1)
        return out

    def get_attention_weights(self, data: Data) -> Tensor:
        """
        Returns per-residue importance scores (N,).
        Higher = more important for binding affinity prediction.
        """
        self.eval()
        with torch.no_grad():
            x          = data.x
            edge_index = data.edge_index
            edge_attr  = data.edge_attr
            batch      = getattr(data, "batch",
                                 torch.zeros(x.size(0), dtype=torch.long))

            x = self.input_proj(x)
            e = self.edge_proj(edge_attr)
            for layer in self.gat_layers:
                x = layer(x, edge_index, edge_attr=e)

            _, weights = self.pool(x, batch)

        return weights  # (N,)

    def get_layer_attention(
        self, data: Data, layer_idx: int = -1
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract raw GAT attention coefficients from a specific layer.

        Returns
        -------
        edge_index   : (2, E)
        attn_weights : (E, heads)
        """
        if layer_idx == -1:
            layer_idx = self.num_layers - 1

        self.eval()
        with torch.no_grad():
            x          = data.x
            edge_index = data.edge_index
            edge_attr  = data.edge_attr

            x = self.input_proj(x)
            e = self.edge_proj(edge_attr)

            for i, layer in enumerate(self.gat_layers):
                if i < layer_idx:
                    x = layer(x, edge_index, edge_attr=e)
                else:
                    x, (ei, alpha) = layer(
                        x, edge_index, edge_attr=e, return_attention=True
                    )
                    break

        return ei, alpha


# ---------------------------------------------------------------------------
# Baseline: simple MLP on mean-pooled features
# ---------------------------------------------------------------------------

class BaselineMLP(nn.Module):
    """
    Ablation baseline: mean-pool node features → MLP regression.
    No graph structure used — shows the value added by the GAT.
    """

    def __init__(self, node_dim: int = 25, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data: Data) -> Tensor:
        batch = getattr(data, "batch",
                        torch.zeros(data.x.size(0), dtype=torch.long))
        pooled = global_mean_pool(data.x, batch)
        return self.net(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running BindingGNN sanity check...")

    N, E, F = 30, 120, 25
    data = Data(
        x=torch.randn(N, F),
        edge_index=torch.randint(0, N, (2, E)),
        edge_attr=torch.rand(E, 1) * 8.0,
        y=torch.tensor([-10.5]),
    )

    model = BindingGNN(node_dim=F, hidden_dim=128, num_layers=3, heads=4)
    pred = model(data)
    attn = model.get_attention_weights(data)

    print(f"  Predicted ΔG: {pred.item():.3f} kcal/mol")
    print(f"  Attention weights shape: {attn.shape}")
    print(f"  Attention sum: {attn.sum().item():.4f}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("  ✓ BindingGNN OK")

    baseline = BaselineMLP(node_dim=F, hidden_dim=128)
    pred_b = baseline(data)
    print(f"\n  Baseline MLP ΔG: {pred_b.item():.3f}")
    print(f"  Baseline params: {sum(p.numel() for p in baseline.parameters()):,}")
    print("  ✓ BaselineMLP OK")
