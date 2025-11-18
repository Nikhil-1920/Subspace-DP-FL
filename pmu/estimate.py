import torch
from torch import Tensor
from typing import Tuple

@torch.no_grad()
def TopkFromSketch(snoisyavg: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Top-k eigendecomposition of a symmetric sketch."""
    evals, evecs = torch.linalg.eigh(snoisyavg)

    if k <= 0:
        return evals, evecs

    keff = min(k, evals.shape[0])
    vals, idx = torch.sort(evals, descending=True)
    vecs = evecs[:, idx]

    topvals = vals[:keff]
    topvecs = vecs[:, :keff]
    return topvals, topvecs

@torch.no_grad()
def LiftSubspace(vecsm: Tensor, proj: Tensor) -> Tensor:
    """Lift sketch-space eigenvectors via projection and orthonormalize."""
    utilde = proj @ vecsm
    u, _ = torch.linalg.qr(utilde, mode="reduced")
    return u
