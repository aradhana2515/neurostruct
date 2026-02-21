"""
preprocess.py
=============
Converts raw PDB structures and binding affinity data into PyTorch Geometric
graph objects ready for GNN training.

For each receptor–ligand pair:
  - Extracts binding-site residues within a shell of the co-crystallized ligand
  - Builds a residue contact graph  (Cα–Cα distance < CONTACT_THRESHOLD)
  - Computes per-residue biochemical features
  - Saves a PyG Data object with:
      x         : node features  (N, F)
      edge_index: contact edges  (2, E)
      edge_attr : edge distances (E, 1)
      y         : binding affinity ΔG (scalar, kcal/mol)
      meta      : dict with pdb_id, receptor, ligand_smiles

Usage:
  python data/preprocess.py --pdb_dir data/raw/structures \
                             --binding_dir data/raw/binding \
                             --out_dir data/processed
"""

import os
import json
import math
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch_geometric.data import Data

# Biopython for PDB parsing
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch, PDBIO
from Bio.PDB.Polypeptide import is_aa

warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTACT_THRESHOLD = 8.0      # Å — Cα–Cα distance for edge creation
BINDING_SHELL     = 10.0     # Å — residues within this distance of ligand
CONVERSION_FACTOR = 0.592    # RT at 300K in kcal/mol, for Kd→ΔG

# One-hot amino acid encoding (20 standard AAs)
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# Per-residue biochemical properties (normalized to [0,1])
# Format: {AA: [hydrophobicity, charge, polarity, mw_normalized, aromatic]}
AA_PROPERTIES: Dict[str, List[float]] = {
    "A": [0.62, 0.00, 0.00, 0.20, 0], "C": [0.29, 0.00, 0.00, 0.28, 0],
    "D": [-0.90, -1.0, 1.00, 0.31, 0], "E": [-0.74, -1.0, 1.00, 0.36, 0],
    "F": [1.19, 0.00, 0.00, 0.49, 1], "G": [0.48, 0.00, 0.00, 0.15, 0],
    "H": [-0.40, 0.50, 0.50, 0.45, 1], "I": [1.38, 0.00, 0.00, 0.40, 0],
    "K": [-1.50, 1.00, 1.00, 0.41, 0], "L": [1.06, 0.00, 0.00, 0.40, 0],
    "M": [0.64, 0.00, 0.00, 0.46, 0], "N": [-0.78, 0.00, 1.00, 0.34, 0],
    "P": [0.12, 0.00, 0.00, 0.30, 0], "Q": [-0.85, 0.00, 1.00, 0.40, 0],
    "R": [-2.53, 1.00, 1.00, 0.51, 0], "S": [-0.18, 0.00, 1.00, 0.24, 0],
    "T": [-0.05, 0.00, 1.00, 0.30, 0], "V": [1.08, 0.00, 0.00, 0.33, 0],
    "W": [0.81, 0.00, 0.00, 0.62, 1], "Y": [0.26, 0.00, 1.00, 0.53, 1],
    "X": [0.00, 0.00, 0.00, 0.35, 0],  # unknown
}

# Normalize hydrophobicity to [0,1]
_hydro_vals = [v[0] for v in AA_PROPERTIES.values()]
_hydro_min, _hydro_max = min(_hydro_vals), max(_hydro_vals)
for _aa, _props in AA_PROPERTIES.items():
    _props[0] = (_props[0] - _hydro_min) / (_hydro_max - _hydro_min)

NODE_FEATURE_DIM = len(AA_LIST) + len(list(AA_PROPERTIES.values())[0])  # 20 + 5 = 25


# ---------------------------------------------------------------------------
# PDB utilities
# ---------------------------------------------------------------------------

def parse_structure(pdb_path: Path):
    """Return a Bio.PDB Structure object."""
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_path.stem, str(pdb_path))


def get_ca_atoms(structure, chain_ids: Optional[List[str]] = None):
    """Extract Cα atoms from all standard amino acid residues."""
    ca_atoms = []
    for model in structure:
        for chain in model:
            if chain_ids and chain.id not in chain_ids:
                continue
            for res in chain:
                if is_aa(res, standard=True) and "CA" in res:
                    ca_atoms.append(res["CA"])
    return ca_atoms


def get_ligand_atoms(structure, ligand_resname: Optional[str] = None):
    """Extract all HETATM atoms (non-water ligands)."""
    ligand_atoms = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0].startswith("H_") and res.resname != "HOH":
                    if ligand_resname is None or res.resname == ligand_resname:
                        for atom in res:
                            ligand_atoms.append(atom)
    return ligand_atoms


def get_binding_site_residues(structure, radius: float = BINDING_SHELL,
                               chain_ids: Optional[List[str]] = None) -> List:
    """
    Return residues within `radius` Å of any heteroatom ligand.
    Falls back to all residues if no ligand is found.
    """
    all_protein_atoms = []
    for model in structure:
        for chain in model:
            if chain_ids and chain.id not in chain_ids:
                continue
            for res in chain:
                if is_aa(res, standard=True):
                    for atom in res:
                        all_protein_atoms.append(atom)

    ligand_atoms = get_ligand_atoms(structure)

    if not ligand_atoms:
        # No ligand — return all protein residues
        residues = []
        for model in structure:
            for chain in model:
                if chain_ids and chain.id not in chain_ids:
                    continue
                for res in chain:
                    if is_aa(res, standard=True):
                        residues.append(res)
        return residues

    ns = NeighborSearch(all_protein_atoms)
    near_residues = set()
    for latom in ligand_atoms:
        for close_atom in ns.search(latom.coord, radius, "A"):
            near_residues.add(close_atom.get_parent())

    return list(near_residues)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def residue_one_letter(res) -> str:
    """Get one-letter code; return 'X' for non-standard."""
    try:
        from Bio.PDB.Polypeptide import three_to_one
        return three_to_one(res.resname)
    except (KeyError, Exception):
        return "X"


def residue_features(res) -> np.ndarray:
    """
    Build feature vector for a single residue:
      [one-hot AA (20)] + [hydrophobicity, charge, polarity, mw_norm, aromatic (5)]
    Returns shape (25,)
    """
    aa = residue_one_letter(res)

    # One-hot
    one_hot = np.zeros(len(AA_LIST), dtype=np.float32)
    idx = AA_TO_IDX.get(aa, -1)
    if idx >= 0:
        one_hot[idx] = 1.0

    # Biochemical properties
    props = np.array(AA_PROPERTIES.get(aa, AA_PROPERTIES["X"]), dtype=np.float32)

    return np.concatenate([one_hot, props])


def build_contact_graph(residues: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Cα–Cα contact graph.
    Returns:
      coords      (N, 3) Cα coordinates
      edge_index  (2, E) edge list
      edge_dists  (E,)   Euclidean distances
    """
    coords = []
    valid_residues = []
    for res in residues:
        if "CA" in res:
            coords.append(res["CA"].coord)
            valid_residues.append(res)

    coords = np.array(coords, dtype=np.float32)  # (N, 3)
    N = len(coords)

    src_list, dst_list, dist_list = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < CONTACT_THRESHOLD:
                src_list += [i, j]
                dst_list += [j, i]
                dist_list += [d, d]

    edge_index = np.array([src_list, dst_list], dtype=np.int64)   # (2, E)
    edge_dists = np.array(dist_list, dtype=np.float32)             # (E,)

    return coords, edge_index, edge_dists, valid_residues


# ---------------------------------------------------------------------------
# Affinity conversion
# ---------------------------------------------------------------------------

def kd_to_delta_g(kd_nM: float) -> float:
    """Convert dissociation constant (nM) to ΔG (kcal/mol) at 300K."""
    kd_M = kd_nM * 1e-9
    return CONVERSION_FACTOR * math.log(kd_M)   # ΔG = RT ln(Kd)


def ki_to_delta_g(ki_nM: float) -> float:
    """Convert inhibition constant (nM) to approximate ΔG."""
    return kd_to_delta_g(ki_nM)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def process_pdb_file(pdb_path: Path, affinity_kcal: float,
                     receptor_name: str) -> Optional[Data]:
    """
    Full pipeline for a single PDB structure:
      1. Parse structure
      2. Extract binding site residues
      3. Compute per-residue features
      4. Build contact graph
      5. Package as PyG Data
    """
    try:
        structure = parse_structure(pdb_path)
        residues = get_binding_site_residues(structure)

        if len(residues) < 5:
            print(f"    [warn] {pdb_path.name}: only {len(residues)} binding residues, skipping")
            return None

        coords, edge_index, edge_dists, valid_res = build_contact_graph(residues)
        node_features = np.stack([residue_features(r) for r in valid_res])  # (N, 25)

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_dists[:, None], dtype=torch.float),
            y=torch.tensor([affinity_kcal], dtype=torch.float),
            pos=torch.tensor(coords, dtype=torch.float),
            num_nodes=len(valid_res),
        )
        data.meta = {
            "pdb_id": pdb_path.stem,
            "receptor": receptor_name,
            "n_residues": len(valid_res),
        }
        return data

    except Exception as e:
        print(f"    [error] {pdb_path.name}: {e}")
        return None


def load_synthetic_affinities() -> Dict[str, float]:
    """
    Placeholder binding affinities (ΔG, kcal/mol) for demonstration.
    In production, these come from BindingDB / parsed JSON.
    Literature sources:
      6HUP  diazepam  Kd ~1nM  → ΔG ≈ -12.3
      6X3S  diazepam  Kd ~3nM  → ΔG ≈ -11.5
      4PE6  ifenprodil Ki ~0.3μM → ΔG ≈ -8.9
    """
    return {
        "6HUP": -12.3,
        "6X3S": -11.5,
        "6D6U": -9.8,
        "7A5V": -8.2,
        "4PE6": -8.9,
        "6MMJ": -7.4,
        "7EU7": -9.1,
        "6WHS": -11.0,
    }


RECEPTOR_LABELS = {
    "6HUP": "GABA-A", "6X3S": "GABA-A", "6D6U": "GABA-A", "7A5V": "GABA-A",
    "4PE6": "NMDA",   "6MMJ": "NMDA",   "7EU7": "NMDA",   "6WHS": "NMDA",
}


def build_dataset(pdb_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    affinities = load_synthetic_affinities()

    dataset = []
    pdb_files = sorted(pdb_dir.glob("*.pdb"))

    print(f"\n{'='*50}")
    print(f"Processing {len(pdb_files)} PDB structures")
    print(f"{'='*50}")

    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.upper()
        if pdb_id not in affinities:
            print(f"  [skip] {pdb_id} — no affinity data")
            continue

        aff = affinities[pdb_id]
        receptor = RECEPTOR_LABELS.get(pdb_id, "unknown")
        print(f"  Processing {pdb_id} ({receptor}, ΔG={aff:.1f} kcal/mol) ...")

        data = process_pdb_file(pdb_path, aff, receptor)
        if data is not None:
            dataset.append(data)
            print(f"    nodes={data.num_nodes}, edges={data.edge_index.shape[1]//2}")

    # Train / val / test split (70/15/15)
    n = len(dataset)
    np.random.seed(42)
    idx = np.random.permutation(n)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    splits = {
        "train": [dataset[i] for i in idx[:n_train]],
        "val":   [dataset[i] for i in idx[n_train:n_train+n_val]],
        "test":  [dataset[i] for i in idx[n_train+n_val:]],
    }

    for split_name, split_data in splits.items():
        out_path = out_dir / f"{split_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(split_data, f)
        print(f"\n  Saved {split_name}: {len(split_data)} graphs → {out_path}")

    print(f"\n✓ Dataset built. Total graphs: {n}")
    print(f"  Node feature dim: {NODE_FEATURE_DIM}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb_dir",     default="data/raw/structures")
    p.add_argument("--binding_dir", default="data/raw/binding")
    p.add_argument("--out_dir",     default="data/processed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(Path(args.pdb_dir), Path(args.out_dir))
