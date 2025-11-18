import numpy as np
from .accountant import RDPAccountantPoisson

def SigmaFromTargetEpsilon(
    q: float,
    rounds: int,
    clipnorm: float,
    epsilon: float,
    delta: float,
    searchrange: tuple = (0.1, 100.0),
) -> float:
    """Binary search for noise multiplier that meets (epsilon, delta)."""
    low, high = searchrange

    for _step in range(80):
        mid = 0.5 * (low + high)
        if mid == low or mid == high:
            break

        try:
            acc = RDPAccountantPoisson()
            acc.AddGaussianRound(q=q, noisemu=mid, steps=rounds)
            eps, _order = acc.GetEps(delta)

            if eps > epsilon:
                low = mid
            else:
                high = mid
        except OverflowError:
            low = mid

    return high
