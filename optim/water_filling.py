import torch
import numpy as np

def WaterFilling(
    gammas: torch.Tensor,
    a: float,
    b: float,
    tau: float,
    B: float,
    d: int,
    tol: float = 1e-6,
    maxiter: int = 60,
) -> np.ndarray:
    """Solve single-group water-filling for optimal eigenvalues."""
    if gammas.ndim != 1:
        gammas = gammas.view(-1)
    gammas = gammas.detach().cpu()

    k = gammas.shape[0]
    if d < k:
        raise ValueError("d must be >= len(gammas).")

    budget = B - tau * d
    if budget <= 0:
        return np.zeros(k, dtype=float)

    def lamOfKappa(kappa: float, gammas: torch.Tensor, tau: float) -> torch.Tensor:
        denom = a * gammas + kappa
        denom = torch.clamp(denom, min=1e-12)
        lamplus = torch.sqrt(b / denom)
        lam = torch.clamp(lamplus - tau, min=0.0)
        return lam

    lowk = 0.0
    highk = 1.0
    for _ in range(40):
        lam = lamOfKappa(highk, gammas, tau)
        s = lam.sum().item()
        if s > budget:
            break
        highk *= 2.0

    for _ in range(maxiter):
        midk = 0.5 * (lowk + highk)
        lam = lamOfKappa(midk, gammas, tau)
        s = lam.sum().item()
        if abs(s - budget) <= tol:
            break
        if s > budget:
            lowk = midk
        else:
            highk = midk

    lam = lamOfKappa(highk, gammas, tau)
    return lam.numpy()