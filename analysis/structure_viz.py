"""
structure_viz.py
================
Visualization utilities for NeuroStruct:
  - PyMOL session generation with binding site colored by attention weights
  - MDAnalysis RMSF B-factor visualization
  - Contact map heatmaps
  - Receptor structure comparison utilities

Usage:
  from analysis.structure_viz import (
      color_by_attention,
      plot_contact_map,
      plot_rmsf_profile,
  )

  color_by_attention("data/raw/structures/6HUP.pdb", attn_weights, "outputs/6HUP_hotspots.pse")
  plot_rmsf_profile(rmsf_array, pdb_id="6HUP", out_path="outputs/6HUP_rmsf.png")
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Union, List

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Color palettes
# ─────────────────────────────────────────────────────────────────────────────

CMAP_ATTENTION = plt.cm.YlOrRd     # yellow → red for high attention
CMAP_RMSF      = plt.cm.cool       # cool blue for flexibility
CMAP_CONTACT   = plt.cm.viridis


def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr)


# ─────────────────────────────────────────────────────────────────────────────
# PyMOL session generation
# ─────────────────────────────────────────────────────────────────────────────

def color_by_attention(
    pdb_path: Union[str, Path],
    attention_weights: np.ndarray,
    out_path: Union[str, Path],
    top_k: int = 10,
) -> None:
    """
    Generate a PyMOL .pml script that colors each residue by its
    GAT attention weight (importance for binding affinity prediction).

    Colors: blue (low importance) → red (high importance)
    Top-k residues are labeled.

    Parameters
    ----------
    pdb_path         : path to .pdb file
    attention_weights: (N_residues,) attention scores from model
    out_path         : output .pml script path
    top_k            : number of top residues to label
    """
    pdb_path = Path(pdb_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    norm_weights = _normalize(attention_weights)
    top_indices  = np.argsort(norm_weights)[-top_k:][::-1]

    # Extract residue IDs from PDB (Biopython)
    residue_ids = _get_residue_ids(pdb_path)

    lines = [
        f"# NeuroStruct — Binding hotspot visualization for {pdb_path.stem}",
        f"load {pdb_path.resolve()}, receptor",
        "hide everything, receptor",
        "show cartoon, receptor",
        "color grey80, receptor",
        "",
        "# Color binding site residues by attention weight",
    ]

    # Assign colors per residue
    for i, (chain_id, res_num, _) in enumerate(residue_ids):
        if i >= len(norm_weights):
            break
        w = float(norm_weights[i])
        # interpolate from blue (0,0,1) to red (1,0,0)
        r = w
        g = 0.0
        b = 1.0 - w
        color_name = f"res_color_{i}"
        lines.append(f"set_color {color_name}, [{r:.3f}, {g:.3f}, {b:.3f}]")
        lines.append(f"color {color_name}, receptor and chain {chain_id} and resi {res_num}")

    lines += [
        "",
        "# Show sticks for top binding residues",
    ]
    for idx in top_indices:
        if idx < len(residue_ids):
            chain_id, res_num, res_name = residue_ids[idx]
            lines.append(
                f"show sticks, receptor and chain {chain_id} and resi {res_num}"
            )
            lines.append(
                f"label receptor and chain {chain_id} and resi {res_num} and name CA, "
                f"'{res_name}{res_num}\\n({norm_weights[idx]:.2f})'"
            )

    lines += [
        "",
        "# Display settings",
        "set cartoon_transparency, 0.2",
        "set label_size, 14",
        "set label_color, black",
        "bg_color white",
        "set ray_shadows, 0",
        "orient receptor",
        f"# Save session: PyMOL → File → Save Session As {out_path.stem}.pse",
    ]

    pml_path = out_path.with_suffix(".pml")
    pml_path.write_text("\n".join(lines))
    print(f"✓ PyMOL script written → {pml_path}")
    print(f"  Run in PyMOL: run {pml_path}")


def _get_residue_ids(pdb_path: Path) -> List[tuple]:
    """Return list of (chain_id, res_num, res_name) for all standard residues."""
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import is_aa
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("s", str(pdb_path))
        residues = []
        for model in struct:
            for chain in model:
                for res in chain:
                    if is_aa(res, standard=True):
                        residues.append((chain.id, res.id[1], res.resname))
        return residues
    except Exception:
        # Fallback: return generic residue IDs
        return [("A", i, "UNK") for i in range(1, 201)]


# ─────────────────────────────────────────────────────────────────────────────
# Contact map heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_contact_map(
    pdb_path: Union[str, Path],
    threshold: float = 8.0,
    out_path: Optional[Union[str, Path]] = None,
    attention_weights: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plot Cα–Cα distance matrix and contact map for a receptor structure.
    Optionally overlay attention weights on the diagonal.
    """
    pdb_path = Path(pdb_path)

    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import is_aa
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("s", str(pdb_path))
        coords = []
        for model in struct:
            for chain in model:
                for res in chain:
                    if is_aa(res, standard=True) and "CA" in res:
                        coords.append(res["CA"].coord)
            break  # first model only
        coords = np.array(coords)
    except Exception as e:
        print(f"  [warn] Could not load structure: {e}. Using random coords.")
        N = 80
        coords = np.random.randn(N, 3) * 10

    N = len(coords)
    # Distance matrix
    diff = coords[:, None, :] - coords[None, :, :]         # (N, N, 3)
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1))            # (N, N)
    contact_mat = (dist_mat < threshold).astype(float)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Contact Analysis — {pdb_path.stem}", fontsize=14, fontweight="bold")

    # Distance matrix
    im0 = axes[0].imshow(dist_mat, cmap="viridis_r", aspect="auto")
    axes[0].set_title(f"Cα Distance Matrix (Å)")
    axes[0].set_xlabel("Residue index")
    axes[0].set_ylabel("Residue index")
    plt.colorbar(im0, ax=axes[0], label="Distance (Å)")

    # Contact map
    im1 = axes[1].imshow(contact_mat, cmap="Blues", aspect="auto")
    axes[1].set_title(f"Contact Map (< {threshold} Å)")
    axes[1].set_xlabel("Residue index")
    axes[1].set_ylabel("Residue index")
    plt.colorbar(im1, ax=axes[1], label="Contact")

    # Overlay attention weights on diagonal
    if attention_weights is not None:
        n = min(len(attention_weights), N)
        norm_attn = _normalize(attention_weights[:n])
        xs = np.arange(n)
        scatter = axes[1].scatter(xs, xs, c=norm_attn, cmap=CMAP_ATTENTION,
                                   s=20, zorder=5, label="Attention")
        plt.colorbar(scatter, ax=axes[1], label="Attention weight")

    plt.tight_layout()

    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"✓ Contact map saved → {out_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RMSF profile plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_rmsf_profile(
    rmsf: np.ndarray,
    pdb_id: str = "",
    attention_weights: Optional[np.ndarray] = None,
    out_path: Optional[Union[str, Path]] = None,
    secondary_structure: Optional[np.ndarray] = None,   # 0=loop,1=helix,2=sheet
) -> plt.Figure:
    """
    Plot per-residue RMSF profile with optional attention weight overlay.

    High RMSF = flexible residues (loops / hinge regions)
    High attention = important for binding prediction
    """
    N = len(rmsf)
    residues = np.arange(1, N + 1)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    fig.suptitle(f"RMSF Profile — {pdb_id}", fontsize=13, fontweight="bold")

    # RMSF line
    ax1.fill_between(residues, rmsf, alpha=0.3, color="#2196F3", label="RMSF")
    ax1.plot(residues, rmsf, color="#1565C0", lw=1.5)
    ax1.set_xlabel("Residue index")
    ax1.set_ylabel("RMSF (Å)", color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.set_xlim(1, N)
    ax1.set_ylim(bottom=0)

    # Secondary structure background (optional)
    if secondary_structure is not None:
        for i, ss in enumerate(secondary_structure):
            if ss == 1:  # helix
                ax1.axvspan(i + 0.5, i + 1.5, alpha=0.08, color="red")
            elif ss == 2:  # sheet
                ax1.axvspan(i + 0.5, i + 1.5, alpha=0.08, color="orange")

    # Attention weights overlay
    if attention_weights is not None:
        ax2 = ax1.twinx()
        n = min(len(attention_weights), N)
        norm_attn = _normalize(attention_weights[:n])
        ax2.stem(residues[:n], norm_attn,
                 linefmt="#E53935-", markerfmt="o", basefmt=" ",
                 label="Attention")
        ax2.set_ylabel("Normalized attention", color="#E53935")
        ax2.tick_params(axis="y", labelcolor="#E53935")
        ax2.set_ylim(0, 1.3)

        # Mark top-5 residues
        top5 = np.argsort(norm_attn)[-5:]
        for idx in top5:
            ax2.annotate(
                f"R{residues[idx]}",
                xy=(residues[idx], norm_attn[idx]),
                xytext=(0, 8), textcoords="offset points",
                fontsize=8, color="#B71C1C",
                ha="center",
            )

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left", fontsize=9)

    plt.tight_layout()

    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"✓ RMSF profile saved → {out_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Prediction scatter plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    receptor_labels: Optional[List[str]] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Scatter plot of predicted vs. true ΔG values, colored by receptor type.
    """
    from scipy.stats import pearsonr, linregress

    r, _ = pearsonr(y_true, y_pred)
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    slope, intercept, *_ = linregress(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))

    if receptor_labels is not None:
        unique_labels = list(set(receptor_labels))
        colors = ["#1976D2", "#E53935", "#388E3C", "#F57C00"]
        color_map = {l: colors[i % len(colors)] for i, l in enumerate(unique_labels)}
        for label in unique_labels:
            mask = np.array(receptor_labels) == label
            ax.scatter(y_true[mask], y_pred[mask],
                       c=color_map[label], label=label, alpha=0.85, s=60, edgecolors="white")
        ax.legend(fontsize=9)
    else:
        ax.scatter(y_true, y_pred, alpha=0.75, s=60, color="#1976D2", edgecolors="white")

    # Regression line
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5, label="Regression")

    # Diagonal
    ax.plot(x_line, x_line, "grey", lw=1, linestyle=":", label="Ideal")

    ax.set_xlabel("Experimental ΔG (kcal/mol)", fontsize=11)
    ax.set_ylabel("Predicted ΔG (kcal/mol)",    fontsize=11)
    ax.set_title(f"NeuroStruct Predictions\nPearson r = {r:.3f}  |  RMSE = {rmse:.3f} kcal/mol",
                 fontsize=12)

    plt.tight_layout()

    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"✓ Prediction scatter saved → {out_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import os

    os.makedirs("outputs", exist_ok=True)

    N = 80
    rmsf = np.random.exponential(0.5, N) + 0.3
    attn = np.random.dirichlet(np.ones(N))

    print("Testing RMSF profile plot...")
    fig = plot_rmsf_profile(rmsf, pdb_id="6HUP", attention_weights=attn,
                             out_path="outputs/test_rmsf.png")
    plt.close(fig)

    print("Testing prediction scatter plot...")
    y_true = np.random.uniform(-13, -7, 50)
    y_pred = y_true + np.random.normal(0, 0.8, 50)
    labels = np.random.choice(["GABA-A", "NMDA"], 50).tolist()
    fig2 = plot_predictions(y_true, y_pred, labels, out_path="outputs/test_scatter.png")
    plt.close(fig2)

    print("✓ Visualization module OK")
