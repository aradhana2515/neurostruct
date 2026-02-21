"""
train.py
========
Training loop for NeuroStruct binding affinity prediction.

Features:
  - AdamW optimizer with cosine LR schedule + linear warmup
  - Pearson correlation + RMSE evaluation metrics
  - Weights & Biases (W&B) experiment tracking
  - Gradient clipping & early stopping
  - Model checkpointing (best val loss)
  - Ablation flag to compare GNN vs. baseline MLP

Usage:
  python models/train.py --epochs 100 --lr 1e-3 --hidden_dim 256 --heads 4
  python models/train.py --baseline   # train MLP baseline for comparison
"""

import os
import math
import time
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.binding_gnn import BindingGNN, BaselineMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(data_dir: Path, split: str) -> List[Data]:
    path = data_dir / f"{split}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}\n"
            "Run `python data/preprocess.py` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Pearson r, RMSE, MAE."""
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae  = float(np.mean(np.abs(preds - targets)))
    if len(preds) > 1:
        r, pval = pearsonr(preds, targets)
    else:
        r, pval = 0.0, 1.0
    return {"pearson_r": float(r), "rmse": rmse, "mae": mae, "pval": float(pval)}


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    criterion = nn.MSELoss()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy().flatten())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args):
    # ── Setup ────────────────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() or
                          args.device == "cpu" else "cpu")
    log.info(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ─────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="neurostruct",
                name=f"{'baseline_mlp' if args.baseline else 'gnn'}_{int(time.time())}",
                config=vars(args),
            )
        except ImportError:
            log.warning("wandb not installed; skipping logging")
            use_wandb = False

    # ── Data ─────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    log.info("Loading dataset...")
    train_data = load_split(data_dir, "train")
    val_data   = load_split(data_dir, "val")
    test_data  = load_split(data_dir, "test")

    log.info(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    node_dim = train_data[0].x.shape[1]
    log.info(f"Node feature dim: {node_dim}")

    if args.baseline:
        model = BaselineMLP(node_dim=node_dim, hidden_dim=args.hidden_dim)
        log.info("Training BASELINE MLP (no graph structure)")
    else:
        model = BindingGNN(
            node_dim=node_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
        )
        log.info(f"Training BindingGNN  (layers={args.num_layers}, heads={args.heads})")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {num_params:,}")

    # ── Optimizer & Schedule ──────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = int(0.05 * total_steps)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    best_ckpt_path = out_dir / "best_model.pt"

    log.info(f"\n{'='*55}")
    log.info(f"Starting training for {args.epochs} epochs")
    log.info(f"{'='*55}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_clip
        )
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_r={val_metrics['pearson_r']:.3f}  "
            f"val_rmse={val_metrics['rmse']:.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({elapsed:.1f}s)"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "train/loss": train_loss,
                "val/loss":    val_metrics["loss"],
                "val/pearson_r": val_metrics["pearson_r"],
                "val/rmse":    val_metrics["rmse"],
                "lr":          scheduler.get_last_lr()[0],
            }, step=epoch)

        # Checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "args":        vars(args),
            }, best_ckpt_path)
            log.info(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── Final evaluation on test set ──────────────────────────────────────
    log.info(f"\nLoading best checkpoint from {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, device)
    import time
    import json
    from pathlib import Path

    # --- Save a reproducible run artifact folder ---
    run_id = f"{'baseline_mlp' if args.baseline else 'gnn'}_{int(time.time())}"
    run_dir = Path("results") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics + config
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # Save split sizes (quick sanity)
    split_info = {
        "train_n": len(train_data),
        "val_n": len(val_data),
        "test_n": len(test_data),
    }
    (run_dir / "splits.json").write_text(json.dumps(split_info, indent=2))

    # Save which checkpoint corresponds to these metrics
    (run_dir / "checkpoint_path.txt").write_text(str(best_ckpt_path) + "\n")

    print(f"✅ Saved run artifacts to: {run_dir}")
    log.info(f"\n{'='*55}")
    log.info("TEST SET RESULTS")
    log.info(f"{'='*55}")
    log.info(f"  Pearson r  : {test_metrics['pearson_r']:.4f}")
    log.info(f"  RMSE       : {test_metrics['rmse']:.4f} kcal/mol")
    log.info(f"  MAE        : {test_metrics['mae']:.4f} kcal/mol")

    if use_wandb:
        import wandb
        wandb.log({
            "test/pearson_r": test_metrics["pearson_r"],
            "test/rmse":      test_metrics["rmse"],
            "test/mae":       test_metrics["mae"],
        })
        wandb.finish()

    return model, test_metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train NeuroStruct binding affinity model")

    # Data
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--out_dir",  type=str, default="checkpoints")

    # Model
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--num_layers",  type=int,   default=4)
    p.add_argument("--heads",       type=int,   default=4)
    p.add_argument("--dropout",     type=float, default=0.15)
    p.add_argument("--baseline",    action="store_true",
                   help="Train baseline MLP instead of GNN")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--patience",     type=int,   default=20,
                   help="Early stopping patience (epochs)")

    # Infra
    p.add_argument("--device",    type=str, default="cuda")
    p.add_argument("--no_wandb",  action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
