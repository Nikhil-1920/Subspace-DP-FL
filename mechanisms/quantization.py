import torch
from torch import Tensor
from typing import Tuple

@torch.no_grad()
def AnisotropicQuantize(gp: Tensor, steps: Tensor, dither: bool = True) -> Tuple[Tensor, Tensor]:
    """Mid-rise scalar quantization with optional subtractive dither."""
    if dither:
        u = torch.rand_like(gp) - 0.5
        y = gp / steps + u
        q = torch.round(y)
        gq = (q - u) * steps
    else:
        q = torch.round(gp / steps)
        gq = q * steps

    err = gq - gp
    return gq, err

@torch.no_grad()
def QuantizeInSubspace(
    g: Tensor,
    u: Tensor,
    lam: Tensor,
    c: float,
    dither: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Project to span(u), quantize, then project back."""
    gp = torch.matmul(u.t(), g)
    steps = c / torch.sqrt(lam.clamp(min=1e-8))
    gq, qerr = AnisotropicQuantize(gp, steps, dither=dither)
    gqfull = torch.matmul(u, gq)
    return gqfull, qerr, steps
