#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh — End-to-end NeuroStruct pipeline
# ============================================================
#
# Usage:
#   bash scripts/run_pipeline.sh [--gpu] [--skip-md] [--demo-only]
#
# Steps:
#   1. Fetch PDB structures + BindingDB data
#   2. Preprocess → contact graphs + PyG datasets
#   3. Run MD simulations → RMSF features (optional)
#   4. Train GNN model (+ baseline MLP for ablation)
#   5. Generate analysis outputs
#   6. Launch Gradio demo

set -euo pipefail

# ── Parse flags ──────────────────────────────────────────────
GPU=false
SKIP_MD=false
DEMO_ONLY=false
EPOCHS=100

for arg in "$@"; do
  case $arg in
    --gpu)       GPU=true ;;
    --skip-md)   SKIP_MD=true ;;
    --demo-only) DEMO_ONLY=true ;;
    --epochs=*)  EPOCHS="${arg#*=}" ;;
  esac
done

DEVICE=$([ "$GPU" = true ] && echo "cuda" || echo "cpu")

echo "================================================================"
echo "  NeuroStruct — End-to-End Pipeline"
echo "================================================================"
echo "  Device     : $DEVICE"
echo "  Epochs     : $EPOCHS"
echo "  Skip MD    : $SKIP_MD"
echo "  Demo only  : $DEMO_ONLY"
echo "================================================================"
echo ""

if [ "$DEMO_ONLY" = true ]; then
  echo "▶ Launching demo..."
  python demo/app.py
  exit 0
fi

mkdir -p data/raw data/processed data/trajectories checkpoints outputs

# ── Step 1: Fetch data ────────────────────────────────────────
echo ""
echo "▶ Step 1/5 — Fetching PDB structures & BindingDB data"
echo "──────────────────────────────────────────────────────"
python data/fetch_pdb.py --out_dir data/raw
echo "✓ Data fetch complete"

# ── Step 2: Preprocess ───────────────────────────────────────
echo ""
echo "▶ Step 2/5 — Preprocessing structures → PyG graphs"
echo "──────────────────────────────────────────────────────"
python data/preprocess.py \
  --pdb_dir     data/raw/structures \
  --binding_dir data/raw/binding \
  --out_dir     data/processed
echo "✓ Preprocessing complete"

# ── Step 3: MD simulations ───────────────────────────────────
if [ "$SKIP_MD" = false ]; then
  echo ""
  echo "▶ Step 3/5 — Running MD simulations (RMSF extraction)"
  echo "──────────────────────────────────────────────────────"
  echo "  Note: Full production runs take 30-120 min per structure."
  echo "  Using short runs for demonstration (5000 steps)."
  python - <<'EOF'
import sys
sys.path.insert(0, ".")
from analysis.md_simulation import compute_rmsf_all
from pathlib import Path

rmsf_dict = compute_rmsf_all(
    pdb_dir=Path("data/raw/structures"),
    output_dir=Path("data/trajectories"),
    production_steps=5_000,   # increase to 500_000 for production
    platform="CPU",
)
print(f"✓ RMSF computed for {len(rmsf_dict)} structures")
for pdb_id, rmsf in rmsf_dict.items():
    print(f"  {pdb_id}: mean RMSF = {rmsf.mean():.2f} Å")
EOF
  echo "✓ MD simulations complete"
else
  echo ""
  echo "▶ Step 3/5 — Skipping MD simulations (--skip-md flag)"
fi

# ── Step 4: Train model ──────────────────────────────────────
echo ""
echo "▶ Step 4/5 — Training BindingGNN"
echo "──────────────────────────────────────────────────────"

# Train GNN
python models/train.py \
  --data_dir   data/processed \
  --out_dir    checkpoints \
  --epochs     "$EPOCHS" \
  --hidden_dim 256 \
  --num_layers 4 \
  --heads      4 \
  --dropout    0.15 \
  --device     "$DEVICE" \
  --no_wandb

# Train baseline MLP for comparison
echo ""
echo "  Training baseline MLP for ablation comparison..."
python models/train.py \
  --data_dir   data/processed \
  --out_dir    checkpoints/baseline \
  --epochs     "$EPOCHS" \
  --hidden_dim 256 \
  --baseline \
  --device     "$DEVICE" \
  --no_wandb

echo "✓ Training complete"

# ── Step 5: Analysis outputs ─────────────────────────────────
echo ""
echo "▶ Step 5/5 — Generating analysis visualizations"
echo "──────────────────────────────────────────────────────"
python - <<'EOF'
import sys, json, numpy as np
sys.path.insert(0, ".")
from pathlib import Path
from analysis.structure_viz import plot_rmsf_profile, plot_contact_map
import matplotlib
matplotlib.use("Agg")

Path("outputs").mkdir(exist_ok=True)

# RMSF profiles
for pdb_id in ["6HUP", "4PE6"]:
    npy_path = Path(f"data/trajectories/{pdb_id}_rmsf.npy")
    if npy_path.exists():
        rmsf = np.load(npy_path)
        plot_rmsf_profile(rmsf, pdb_id=pdb_id,
                          out_path=f"outputs/{pdb_id}_rmsf.png")
    else:
        print(f"  [skip] {pdb_id} RMSF not found (run MD first)")

# Contact map for 6HUP
pdb_path = Path("data/raw/structures/6HUP.pdb")
if pdb_path.exists():
    plot_contact_map(pdb_path, out_path="outputs/6HUP_contacts.png")

print("✓ Visualizations saved to outputs/")
EOF

echo ""
echo "================================================================"
echo "  ✅ Pipeline complete!"
echo "================================================================"
echo ""
echo "  Checkpoints : checkpoints/best_model.pt"
echo "  Outputs     : outputs/"
echo ""
echo "  Launch demo:"
echo "    python demo/app.py"
echo ""
echo "  View results:"
echo "    ls outputs/"
echo ""
