"""
esm_embedder.py
===============
Wraps Meta's ESM-2 protein language model to produce per-residue
embeddings for receptor sequences.

ESM-2 (650M param) produces 1280-dim representations per token.
We use these as rich, evolution-informed node features for the GNN —
far more informative than one-hot amino acid encodings alone.

Usage:
  from models.esm_embedder import ESM2Embedder

  embedder = ESM2Embedder(model_name="esm2_t33_650M_UR50D", device="cuda")
  embeddings = embedder.embed_sequence("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSR")
  # embeddings: Tensor (L, 1280)
"""

import torch
import torch.nn as nn
from typing import List, Union, Optional
from pathlib import Path


class ESM2Embedder(nn.Module):
    """
    Per-residue ESM-2 embeddings.

    Parameters
    ----------
    model_name : str
        ESM-2 checkpoint.  Choices (smallest to largest):
          'esm2_t6_8M_UR50D'       —   8M params, 320-dim
          'esm2_t12_35M_UR50D'     —  35M params, 480-dim
          'esm2_t30_150M_UR50D'    — 150M params, 640-dim
          'esm2_t33_650M_UR50D'    — 650M params, 1280-dim  ← default
          'esm2_t36_3B_UR50D'      —   3B params, 2560-dim
    layer : int
        Which transformer layer to extract representations from.
        -1 = last layer (default).
    device : str
        'cpu', 'cuda', or 'mps'
    """

    ESM_OUTPUT_DIMS = {
        "esm2_t6_8M_UR50D":    320,
        "esm2_t12_35M_UR50D":  480,
        "esm2_t30_150M_UR50D": 640,
        "esm2_t33_650M_UR50D": 1280,
        "esm2_t36_3B_UR50D":   2560,
    }

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        layer: int = -1,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = self.ESM_OUTPUT_DIMS[model_name]

        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._num_layers = None

    def _load_model(self):
        """Lazy-load ESM-2 on first use (avoids slow import at import time)."""
        if self._model is not None:
            return
        try:
            import esm
        except ImportError as e:
            raise ImportError(
                "fair-esm not installed. Run: pip install fair-esm"
            ) from e

        print(f"Loading ESM-2 model: {self.model_name} ...")
        self._model, self._alphabet = esm.pretrained.load_model_and_alphabet(
            self.model_name
        )
        self._model = self._model.eval().to(self.device)
        self._batch_converter = self._alphabet.get_batch_converter()
        self._num_layers = self._model.num_layers
        self._repr_layer = self._num_layers if self.layer == -1 else self.layer
        print(f"  ✓ Loaded. Output dim: {self.output_dim}, Layer: {self._repr_layer}")

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """
        Embed a single protein sequence.

        Parameters
        ----------
        sequence : str  (single-letter AA codes, no gaps)

        Returns
        -------
        Tensor of shape (L, output_dim) — one vector per residue
        """
        self._load_model()

        data = [("seq", sequence)]
        _, _, tokens = self._batch_converter(data)
        tokens = tokens.to(self.device)

        results = self._model(tokens, repr_layers=[self._repr_layer])
        # Shape: (1, L+2, D) — strip BOS/EOS tokens
        embeddings = results["representations"][self._repr_layer][0, 1:-1, :]

        return embeddings.cpu()  # (L, D)

    @torch.no_grad()
    def embed_batch(
        self, sequences: List[str], max_len: int = 1024
    ) -> List[torch.Tensor]:
        """
        Embed a list of sequences (variable length).

        Sequences longer than max_len are truncated with a warning.
        Returns a list of tensors, each of shape (L_i, output_dim).
        """
        self._load_model()
        results = []

        for seq in sequences:
            if len(seq) > max_len:
                import warnings
                warnings.warn(
                    f"Sequence length {len(seq)} > max_len {max_len}; truncating.",
                    UserWarning,
                )
                seq = seq[:max_len]
            results.append(self.embed_sequence(seq))

        return results

    @torch.no_grad()
    def embed_fasta(self, fasta_path: Union[str, Path]) -> dict:
        """
        Embed all sequences in a FASTA file.

        Returns dict: {sequence_id: Tensor(L, D)}
        """
        self._load_model()
        embeddings = {}

        from Bio import SeqIO
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            seq_id = record.id
            seq = str(record.seq)
            print(f"  Embedding {seq_id} (L={len(seq)}) ...")
            embeddings[seq_id] = self.embed_sequence(seq)

        return embeddings


# ---------------------------------------------------------------------------
# Lightweight fallback embedder (no ESM installed — uses random projections)
# ---------------------------------------------------------------------------

class RandomProjectionEmbedder(nn.Module):
    """
    Fallback embedder that maps one-hot residues to a 1280-dim space
    via a fixed random projection. Useful for testing the pipeline
    without downloading ESM weights (~1.3 GB).

    Drop-in replacement for ESM2Embedder.
    """

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

    def __init__(self, output_dim: int = 1280, seed: int = 42):
        super().__init__()
        self.output_dim = output_dim
        torch.manual_seed(seed)
        self.projection = nn.Linear(len(self.AA_LIST), output_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        # Freeze weights — this is not trained
        for p in self.projection.parameters():
            p.requires_grad = False

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Returns (L, output_dim) pseudo-embeddings."""
        L = len(sequence)
        one_hot = torch.zeros(L, len(self.AA_LIST))
        for i, aa in enumerate(sequence):
            idx = self.AA_TO_IDX.get(aa, 0)
            one_hot[i, idx] = 1.0
        with torch.no_grad():
            return self.projection(one_hot)  # (L, output_dim)

    def embed_batch(self, sequences: List[str], **kwargs) -> List[torch.Tensor]:
        return [self.embed_sequence(s) for s in sequences]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(use_esm: bool = True, **kwargs) -> nn.Module:
    """
    Return an ESM2Embedder if `use_esm=True` and fair-esm is installed,
    otherwise fall back to RandomProjectionEmbedder.
    """
    if use_esm:
        try:
            import esm  # noqa: F401
            return ESM2Embedder(**kwargs)
        except ImportError:
            print("[warn] fair-esm not available; using RandomProjectionEmbedder")
    return RandomProjectionEmbedder(output_dim=kwargs.get("output_dim", 1280))


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing RandomProjectionEmbedder (no ESM download required)...")
    emb = RandomProjectionEmbedder(output_dim=64)
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSR"
    out = emb.embed_sequence(seq)
    print(f"  Input sequence length: {len(seq)}")
    print(f"  Output shape: {out.shape}")  # (40, 64)
    assert out.shape == (len(seq), 64)
    print("  ✓ Embedder OK")
