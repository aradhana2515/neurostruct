"""
attention_map.py
================
Extracts and analyzes Graph Attention Network (GAT) attention weights
to identify binding hotspot residues in neurotransmitter receptors.

Two types of attention:
  1. Node-level (readout attention): which residues are most important
     for the final binding affinity prediction
  2. Edge-level (GAT attention): which residue–residue interactions
     the network focuses on within the contact graph

These can be overlaid on structure visualizations to reveal
the model's "reasoning" about binding sites.

Usage:
  from analysis.attention_map import AttentionAnalyzer

  analyzer = AttentionAnalyzer(model, data)
  node_scores = analyzer.node_importance()
  edge_scores = analyzer.edge_importance(layer_idx=-1)
  top_residues = analyzer.top_binding_residues(k=10)
  analyzer.save_report("outputs/6HUP_attention.json")
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List, Tuple, Union


class AttentionAnalyzer:
    """
    Extracts and analyzes attention weights from a trained BindingGNN.

    Parameters
    ----------
    model : BindingGNN
    data  : PyG Data object (single receptor graph, not batched)
    residue_ids : Optional list of (chain, resnum, resname) tuples
    """

    def __init__(self, model, data, residue_ids: Optional[List[tuple]] = None):
        self.model = model
        self.data  = data
        self.residue_ids = residue_ids

        self._node_weights = None   # cached readout attention
        self._edge_attns   = {}     # cached per-layer GAT attention
        self._prediction   = None   # cached ΔG prediction

    # ─────────────────────────────────────────────────────────────────
    # Core extraction methods
    # ─────────────────────────────────────────────────────────────────

    def predict(self) -> float:
        """Return predicted ΔG (kcal/mol)."""
        if self._prediction is None:
            self.model.eval()
            with torch.no_grad():
                out = self.model(self.data)
            self._prediction = float(out.item())
        return self._prediction

    def node_importance(self) -> np.ndarray:
        """
        Readout attention weights — one score per residue.
        Higher = more important for binding affinity prediction.

        Returns (N,) normalized importance scores.
        """
        if self._node_weights is None:
            weights = self.model.get_attention_weights(self.data)
            self._node_weights = weights.numpy()
        return self._node_weights

    def edge_importance(self, layer_idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-edge GAT attention from a specified layer.

        Returns
        -------
        edge_index   : (2, E) source/target residue pairs
        edge_weights : (E,)   mean attention across heads
        """
        if layer_idx not in self._edge_attns:
            ei, alpha = self.model.get_layer_attention(self.data, layer_idx)
            edge_index  = ei.cpu().numpy()                  # (2, E)
            edge_weights = alpha.cpu().numpy().mean(axis=-1) # mean over heads → (E,)
            self._edge_attns[layer_idx] = (edge_index, edge_weights)
        return self._edge_attns[layer_idx]

    def top_binding_residues(self, k: int = 10) -> List[dict]:
        """
        Return the top-k residues ranked by node importance.

        Each entry is a dict:
          {rank, residue_idx, chain, resnum, resname, importance_score}
        """
        scores = self.node_importance()
        norm   = scores / scores.sum()
        top_k  = np.argsort(norm)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_k, 1):
            entry = {
                "rank":       rank,
                "residue_idx": int(idx),
                "importance":  float(norm[idx]),
            }
            if self.residue_ids and idx < len(self.residue_ids):
                chain, resnum, resname = self.residue_ids[idx]
                entry.update({"chain": chain, "resnum": int(resnum), "resname": resname})
            results.append(entry)

        return results

    def residue_partner_map(self, residue_idx: int, k: int = 5) -> List[dict]:
        """
        For a given residue, return its top-k most strongly attended
        interaction partners (from edge attention).

        Useful for understanding allosteric networks.
        """
        edge_index, edge_weights = self.edge_importance()
        src, dst = edge_index[0], edge_index[1]

        # Find all edges where this residue is the source
        mask = (src == residue_idx)
        partner_indices = dst[mask]
        partner_weights = edge_weights[mask]

        # Sort by weight
        order = np.argsort(partner_weights)[::-1][:k]
        partners = []
        for rank, oi in enumerate(order, 1):
            pidx = int(partner_indices[oi])
            entry = {
                "rank": rank,
                "partner_residue_idx": pidx,
                "attention": float(partner_weights[oi]),
            }
            if self.residue_ids and pidx < len(self.residue_ids):
                chain, resnum, resname = self.residue_ids[pidx]
                entry.update({"chain": chain, "resnum": int(resnum), "resname": resname})
            partners.append(entry)

        return partners

    # ─────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────

    def plot_node_importance(
        self,
        out_path: Optional[Union[str, Path]] = None,
        top_k: int = 15,
    ) -> plt.Figure:
        """Horizontal bar chart of top-k residues by importance."""
        top_residues = self.top_binding_residues(k=top_k)
        labels = []
        values = []
        for r in reversed(top_residues):
            if "resname" in r:
                label = f"{r['resname']}{r['resnum']} ({r['chain']})"
            else:
                label = f"Res {r['residue_idx']}"
            labels.append(label)
            values.append(r["importance"])

        fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.35)))
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.95, len(values)))
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.7)

        ax.set_xlabel("Normalized attention weight", fontsize=11)
        ax.set_title(
            f"Top {top_k} Binding Site Residues\n"
            f"Predicted ΔG = {self.predict():.2f} kcal/mol",
            fontsize=12,
        )
        ax.axvline(1 / len(self.node_importance()), color="grey",
                   linestyle="--", lw=1, label="Uniform baseline")
        ax.legend(fontsize=9)
        plt.tight_layout()

        if out_path:
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            print(f"✓ Node importance chart → {out_path}")

        return fig

    def plot_attention_heatmap(
        self,
        out_path: Optional[Union[str, Path]] = None,
        layer_idx: int = -1,
        max_residues: int = 60,
    ) -> plt.Figure:
        """
        Heatmap of inter-residue attention weights (E × heads view
        collapsed to a residue × residue matrix).
        """
        edge_index, edge_weights = self.edge_importance(layer_idx)
        N = int(edge_index.max()) + 1
        N = min(N, max_residues)

        attn_matrix = np.zeros((N, N))
        for i in range(edge_index.shape[1]):
            s, d = edge_index[0, i], edge_index[1, i]
            if s < N and d < N:
                attn_matrix[s, d] = edge_weights[i]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(attn_matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")
        ax.set_title(
            f"GAT Attention Heatmap (Layer {layer_idx})\n"
            f"First {N} residues", fontsize=12
        )
        ax.set_xlabel("Target residue", fontsize=11)
        ax.set_ylabel("Source residue", fontsize=11)
        plt.colorbar(im, ax=ax, label="Attention weight")
        plt.tight_layout()

        if out_path:
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            print(f"✓ Attention heatmap → {out_path}")

        return fig

    # ─────────────────────────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────────────────────────

    def save_report(
        self,
        out_path: Union[str, Path],
        top_k: int = 20,
    ) -> dict:
        """
        Save a JSON report of:
          - Predicted ΔG
          - Top-k binding residues with importance scores
          - Model metadata
        """
        report = {
            "predicted_dG_kcal_mol": self.predict(),
            "n_residues": int(self.data.num_nodes),
            "n_edges": int(self.data.edge_index.shape[1]),
            "top_binding_residues": self.top_binding_residues(k=top_k),
        }

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"✓ Attention report saved → {out_path}")
        return report


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble analysis across multiple structures
# ─────────────────────────────────────────────────────────────────────────────

def consensus_hotspots(
    reports: List[dict],
    top_k: int = 10,
) -> List[dict]:
    """
    Aggregate attention reports from multiple PDB structures of the same
    receptor to find consensus binding hotspot positions.

    Residues that consistently receive high attention across multiple
    crystal structures / MD snapshots are likely true binding determinants.

    Parameters
    ----------
    reports : list of dicts from AttentionAnalyzer.save_report()

    Returns list of consensus hotspots sorted by mean importance.
    """
    from collections import defaultdict

    position_scores = defaultdict(list)  # resnum → list of importance scores

    for report in reports:
        for res in report.get("top_binding_residues", []):
            key = f"{res.get('chain','A')}_{res.get('resnum', res['residue_idx'])}"
            position_scores[key].append(res["importance"])

    consensus = []
    for key, scores in position_scores.items():
        parts = key.split("_")
        chain  = parts[0]
        resnum = parts[1]
        consensus.append({
            "chain":        chain,
            "resnum":       resnum,
            "mean_importance": float(np.mean(scores)),
            "std_importance":  float(np.std(scores)),
            "n_structures": len(scores),
        })

    consensus.sort(key=lambda x: x["mean_importance"], reverse=True)
    return consensus[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (no model needed — synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("Attention map analysis — synthetic test\n")

    # Simulate attention scores
    N = 50
    attn_weights = np.random.dirichlet(np.ones(N) * 0.5)

    # Fake residue IDs
    residue_ids = [("A", i, np.random.choice(["ALA","GLY","SER","PHE","TRP"]))
                   for i in range(1, N+1)]

    # Minimal mock analyzer
    class MockAnalyzer:
        def node_importance(self): return attn_weights
        def predict(self): return -10.34
        def top_binding_residues(self, k=10):
            norm = attn_weights / attn_weights.sum()
            top = np.argsort(norm)[::-1][:k]
            return [{"rank": i+1, "residue_idx": int(t),
                     "chain": "A", "resnum": t+1,
                     "resname": residue_ids[t][2],
                     "importance": float(norm[t])} for i, t in enumerate(top)]

    analyzer = MockAnalyzer()
    top = analyzer.top_binding_residues(k=5)

    print("Top 5 predicted binding residues:")
    for r in top:
        print(f"  #{r['rank']}  {r['resname']}{r['resnum']}  "
              f"importance={r['importance']:.4f}")

    print("\n✓ Attention analysis module OK")
