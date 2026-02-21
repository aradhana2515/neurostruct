"""
fetch_pdb.py
============
Downloads experimental PDB structures for GABA-A and NMDA receptors
and fetches binding affinity data from BindingDB for known ligands.

Structures used:
  GABA-A (α1β2γ2):  6HUP, 6X3S, 6D6U
  NMDA  (GluN1/GluN2B): 4PE6, 6MMJ, 7EU7

Usage:
  python data/fetch_pdb.py --out_dir data/raw
"""

import os
import time
import argparse
import requests
import json
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GABA_A_PDB_IDS = ["6HUP", "6X3S", "6D6U", "7A5V"]
NMDA_PDB_IDS   = ["4PE6", "6MMJ", "7EU7", "6WHS"]

ALL_PDB_IDS = GABA_A_PDB_IDS + NMDA_PDB_IDS

RECEPTOR_METADATA = {
    "6HUP": {"receptor": "GABA-A", "subunits": "a1b2g2", "resolution_A": 2.5,
              "note": "Full receptor with benzodiazepine site"},
    "6X3S": {"receptor": "GABA-A", "subunits": "a1b3g2", "resolution_A": 2.9,
              "note": "Diazepam-bound"},
    "6D6U": {"receptor": "GABA-A", "subunits": "a1b2g2", "resolution_A": 3.1,
              "note": "GABA + picrotoxin bound"},
    "7A5V": {"receptor": "GABA-A", "subunits": "a1b2", "resolution_A": 2.2,
              "note": "Bicuculline-bound"},
    "4PE6": {"receptor": "NMDA",   "subunits": "GluN1/GluN2B", "resolution_A": 3.0,
              "note": "Antagonist-bound, ligand binding domain"},
    "6MMJ": {"receptor": "NMDA",   "subunits": "GluN1/GluN2B", "resolution_A": 3.9,
              "note": "Full tetrameric receptor"},
    "7EU7": {"receptor": "NMDA",   "subunits": "GluN1/GluN2A", "resolution_A": 3.3,
              "note": "Open channel conformation"},
    "6WHS": {"receptor": "NMDA",   "subunits": "GluN1/GluN2B", "resolution_A": 2.14,
              "note": "High resolution LBD with glycine"},
}

BINDINGDB_TARGETS = {
    "GABA-A": "P14867",   # GABRA1 (alpha-1 subunit) UniProt ID
    "NMDA":   "Q05586",   # GRIN1 (GluN1 subunit) UniProt ID
}

PDB_DOWNLOAD_URL  = "https://files.rcsb.org/download/{pdb_id}.pdb"
BINDINGDB_API_URL = (
    "https://www.bindingdb.org/axis2/services/BDBService/getLigandsByUniprots"
    "?uniprot={uniprot}&response=json&cutoff=10000"
)


# ---------------------------------------------------------------------------
# PDB download
# ---------------------------------------------------------------------------

def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    """Download a single PDB file; skip if already present."""
    out_path = out_dir / f"{pdb_id}.pdb"
    if out_path.exists():
        print(f"  [skip] {pdb_id}.pdb already present")
        return out_path

    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"  [warn] Could not download {pdb_id}: HTTP {resp.status_code}")
        return out_path

    out_path.write_text(resp.text)
    print(f"  [ok]   {pdb_id}.pdb  ({len(resp.text)//1024} KB)")
    return out_path


def download_all_structures(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print("Downloading PDB structures")
    print(f"{'='*50}")
    for pdb_id in ALL_PDB_IDS:
        meta = RECEPTOR_METADATA[pdb_id]
        print(f"  {pdb_id} | {meta['receptor']} | {meta['note']}")
        download_pdb(pdb_id, out_dir)
        time.sleep(0.5)  # be polite to RCSB

    # Save metadata
    meta_path = out_dir / "structure_metadata.json"
    meta_path.write_text(json.dumps(RECEPTOR_METADATA, indent=2))
    print(f"\nMetadata written to {meta_path}")


# ---------------------------------------------------------------------------
# BindingDB fetch
# ---------------------------------------------------------------------------

def fetch_bindingdb(uniprot_id: str, receptor_name: str, out_dir: Path) -> Path:
    """
    Query BindingDB for all ligands binding to a given UniProt target.
    Returns a TSV file with columns: ligand_smiles, Ki_nM, IC50_nM, Kd_nM, pdb_id
    """
    out_path = out_dir / f"bindingdb_{receptor_name.replace('-','_').lower()}.json"
    if out_path.exists():
        print(f"  [skip] BindingDB data for {receptor_name} already present")
        return out_path

    print(f"  Fetching BindingDB data for {receptor_name} ({uniprot_id})...")
    url = BINDINGDB_API_URL.format(uniprot=uniprot_id)
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            out_path.write_text(resp.text)
            print(f"  [ok]   Saved to {out_path}")
        else:
            print(f"  [warn] HTTP {resp.status_code} for {receptor_name}")
            # Write empty placeholder so pipeline can continue
            out_path.write_text(json.dumps({"affinities": [], "note": "fetch_failed"}))
    except requests.exceptions.RequestException as e:
        print(f"  [warn] Request failed: {e}")
        out_path.write_text(json.dumps({"affinities": [], "note": str(e)}))
    return out_path


def fetch_all_binding_data(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print("Fetching BindingDB affinity data")
    print(f"{'='*50}")
    for receptor_name, uniprot_id in BINDINGDB_TARGETS.items():
        fetch_bindingdb(uniprot_id, receptor_name, out_dir)
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Sequence fetch from UniProt
# ---------------------------------------------------------------------------

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"

RECEPTOR_SEQUENCES = {
    # Key subunits — we'll embed just the extracellular + TM domains
    "GABRA1_HUMAN": "P14867",
    "GABRB2_HUMAN": "P47870",
    "GABRG2_HUMAN": "P18507",
    "GRIN1_HUMAN":  "Q05586",
    "GRIN2B_HUMAN": "Q13224",
}


def fetch_sequences(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print("Fetching UniProt sequences (FASTA)")
    print(f"{'='*50}")
    for name, uniprot_id in RECEPTOR_SEQUENCES.items():
        out_path = out_dir / f"{name}.fasta"
        if out_path.exists():
            print(f"  [skip] {name}.fasta")
            continue
        url = UNIPROT_FASTA_URL.format(uniprot=uniprot_id)
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            out_path.write_text(resp.text)
            seq_len = sum(
                len(line) for line in resp.text.splitlines() if not line.startswith(">")
            )
            print(f"  [ok]   {name}.fasta  ({seq_len} aa)")
        else:
            print(f"  [warn] {name}: HTTP {resp.status_code}")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fetch PDB structures and binding data")
    p.add_argument("--out_dir", type=str, default="data/raw",
                   help="Root output directory (default: data/raw)")
    p.add_argument("--skip_bindingdb", action="store_true",
                   help="Skip BindingDB fetch (useful for offline testing)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.out_dir)

    download_all_structures(root / "structures")
    fetch_sequences(root / "sequences")

    if not args.skip_bindingdb:
        fetch_all_binding_data(root / "binding")
    else:
        print("\n[skip] BindingDB fetch (--skip_bindingdb flag set)")

    print(f"\n✓ Data collection complete. All files in {root}/")


if __name__ == "__main__":
    main()
