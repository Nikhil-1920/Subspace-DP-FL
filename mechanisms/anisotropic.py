import torch
from torch import Tensor

@torch.no_grad()
def PNorm(g: Tensor, u: Tensor, lam: Tensor, tau: float) -> Tensor:
    """P-norm of g for P = u diag(lam) u^T + tau I."""
    a = torch.matmul(u.t(), g)
    valspan = torch.sum(lam * (a * a))
    valeu = torch.sum(g * g)
    val = valspan + tau * valeu
    return torch.sqrt(torch.clamp(val, min=0.0))

@torch.no_grad()
def ClipP(g: Tensor, u: Tensor, lam: Tensor, tau: float, c: float) -> Tensor:
    """P-norm clipping with radius c."""
    nrm = PNorm(g, u, lam, tau)
    scale = torch.clamp(c / (nrm + 1e-12), max=1.0)
    return g * scale

@torch.no_grad()
def SampleEllipticalNoise(
    dim: int,
    sigma: float,
    u: Tensor,
    lam: Tensor,
    tau: float,
    device=None,
) -> Tensor:
    """Sample Gaussian noise with covariance sigma² P⁻¹."""
    if device is None:
        device = u.device

    k = lam.shape[0]

    a = torch.randn(k, device=device)
    spancoef = (sigma / torch.sqrt(lam + tau)) * a
    spansamp = torch.matmul(u, spancoef)

    w = torch.randn(dim, device=device)
    proj = torch.matmul(u, torch.matmul(u.t(), w))
    ortho = w - proj
    ortho = (sigma / torch.sqrt(torch.tensor(tau, device=device))) * ortho

    return spansamp + ortho

@torch.no_grad()
def AggregatePClipped(
    clippedupdates: Tensor,
    u: Tensor,
    lam: Tensor,
    tau: float,
    sigma: float,
) -> Tensor:
    """Average P-clipped updates with one draw of elliptical noise."""
    n, d = clippedupdates.shape
    s = clippedupdates.sum(dim=0)
    z = SampleEllipticalNoise(d, sigma, u, lam, tau, device=s.device)
    return (s + z) / max(n, 1)
