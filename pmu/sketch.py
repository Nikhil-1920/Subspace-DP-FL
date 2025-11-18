import torch
from torch import Tensor

@torch.no_grad()
def MakeJlMatrix(d: int, m: int, device=None, seed: int = 1234) -> Tensor:
    """Create a JL random projection matrix."""
    if device is None:
        device = torch.device("cpu")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    mat = torch.randn(d, m, generator=gen, device=device)
    mat = mat / (m ** 0.5)
    return mat

@torch.no_grad()
def SketchVector(vec: Tensor, proj: Tensor) -> Tensor:
    """Project a vector into sketch space."""
    return proj.t() @ vec

@torch.no_grad()
def UpdateSketchSum(ssum: Tensor, svec: Tensor) -> Tensor:
    """Update sketch second-moment sum with one sketched vector."""
    return ssum + torch.outer(svec, svec)

@torch.no_grad()
def AddNoiseToSketch(ssum: Tensor, sigmasum: float) -> Tensor:
    """Add symmetric Gaussian noise to a sketch sum matrix."""
    e = torch.randn_like(ssum) * sigmasum
    e = (e + e.t()) / 2.0
    return ssum + e
