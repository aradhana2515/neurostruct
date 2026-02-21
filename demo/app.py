"""
demo/app.py
Real Gradio demo: loads a trained checkpoint and runs BindingGNN on a PDB ID.

Usage:
  python demo/app.py --checkpoint checkpoints/best_model.pt
Then open: http://localhost:7860
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import requests
import torch
import gradio as gr

warnings.filterwarnings("ignore")

# Allow imports from repo root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.binding_gnn import BindingGNN
from data import preprocess as pp  # reuse your preprocessing utilities :contentReference[oaicite:8]{index=8}


def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    pdb_id = pdb_id.strip().lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb_id}.pdb"

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path


def graph_from_pdb(pdb_path: Path) -> torch_geometric.data.Data:
    """
    Build a PyG Data graph from PDB:
      - binding site residues if ligand present (fallback: all residues) :contentReference[oaicite:9]{index=9}
      - contact graph with edge distances :contentReference[oaicite:10]{index=10}
      - node features: 25-dim biochemical + one-hot AA :contentReference[oaicite:11]{index=11}
    """
    structure = pp.parse_structure(pdb_path)
    residues = pp.get_binding_site_residues(structure)  # ligand shell fallback built-in :contentReference[oaicite:12]{index=12}
    coords, edge_index, edge_dists, valid_res = pp.build_contact_graph(residues)

    node_features = np.stack([pp.residue_features(r) for r in valid_res]).astype(np.float32)

    # Build Data WITHOUT y (unknown)
    from torch_geometric.data import Data
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_dists[:, None], dtype=torch.float),
        pos=torch.tensor(coords, dtype=torch.float),
        num_nodes=len(valid_res),
    )

    # Add minimal metadata
    data.meta = {"pdb_id": pdb_path.stem, "n_residues": len(valid_res)}
    return data


def load_model(checkpoint_path: Path, node_dim: int) -> BindingGNN:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt.get("args", {})
    hidden_dim = int(args.get("hidden_dim", 256))
    num_layers = int(args.get("num_layers", 4))
    heads = int(args.get("heads", 4))
    dropout = float(args.get("dropout", 0.15))

    model = BindingGNN(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        edge_dim=1,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(pdb_id: str, checkpoint_path: str):
    if not pdb_id or len(pdb_id.strip()) < 4:
        return "⚠️ Enter a valid PDB ID (e.g., 6HUP).", None, None

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        return f"⚠️ Checkpoint not found: {ckpt_path}", None, None

    try:
        pdb_path = download_pdb(pdb_id, out_dir=Path("demo/_pdb_cache"))
    except Exception as e:
        return f"⚠️ Failed to download PDB {pdb_id}: {e}", None, None

    try:
        data = graph_from_pdb(pdb_path)
    except Exception as e:
        return f"⚠️ Failed to build graph from PDB: {e}", None, None

    model = load_model(ckpt_path, node_dim=data.x.shape[1])

    pred_dG = float(model(data).item())
    attn = model.get_attention_weights(data).cpu().numpy()  # sums ~1.0 :contentReference[oaicite:13]{index=13}

    # Top residues
    top_idx = np.argsort(attn)[::-1][:10]
    top = [{"rank": i + 1, "res_idx": int(idx + 1), "importance": float(attn[idx])} for i, idx in enumerate(top_idx)]

    summary = f"""
### Prediction
- **PDB:** `{pdb_id.upper()}`
- **Residues in graph:** `{data.num_nodes}`
- **Predicted ΔG:** `{pred_dG:.3f} kcal/mol`

### Top residues (attention pooling)
{chr(10).join([f"- #{t['rank']}: residue {t['res_idx']}  (importance={t['importance']:.4f})" for t in top])}
""".strip()

    report = json.dumps(
        {
            "pdb_id": pdb_id.upper(),
            "n_residues": int(data.num_nodes),
            "predicted_dG_kcal_per_mol": pred_dG,
            "top_residues": top,
            "attention_sum": float(attn.sum()),
        },
        indent=2,
    )

    return summary, report, str(pdb_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    with gr.Blocks(title="NeuroStruct (Real Demo)") as demo:
        gr.Markdown("# NeuroStruct — Real checkpoint demo (PDB → graph → GNN)")

        with gr.Row():
            pdb_id = gr.Textbox(value="6HUP", label="PDB ID (e.g., 6HUP)")
            ckpt = gr.Textbox(value=args.checkpoint, label="Checkpoint path")

        btn = gr.Button("Predict", variant="primary")
        out_md = gr.Markdown()
        out_json = gr.Code(language="json", label="Report")
        out_path = gr.Textbox(label="Downloaded PDB path")

        btn.click(predict, inputs=[pdb_id, ckpt], outputs=[out_md, out_json, out_path])

    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
