import torch

def SpiK(
    gammasorted: torch.Tensor,
    lamsorted: torch.Tensor,
    sigma: float,
    shat: float,
    k: int,
    tau: float,
) -> float:
    """Compute SPI_k for top-k dimensions."""
    k = min(k, len(gammasorted), len(lamsorted))
    if k == 0:
        return 0.0
    topg = gammasorted[:k]
    topl = lamsorted[:k]
    clipdist = (1.0 - shat) ** 2 * topg
    noisedist = (sigma ** 2) / (tau + topl)
    return (clipdist + noisedist).sum().item()

@torch.no_grad()
def EstimateClipFactors(
    clientupdates: torch.Tensor,
    U: torch.Tensor,
    lam: torch.Tensor,
    tau: float,
    C: float,
) -> float:
    """Estimate average clipping factor ŝ."""
    from mechanisms.anisotropic import pnorm

    scales = []
    for g in clientupdates:
        nrm = pnorm(g, U, lam, tau)
        scale = (C / (nrm + 1e-12)).clamp(max=1.0)
        scales.append(scale.item())

    return sum(scales) / len(scales) if scales else 1.0
