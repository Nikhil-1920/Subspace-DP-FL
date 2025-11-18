import numpy as np
import math

class RDPAccountantPoisson:
    """RDP accountant for Poisson-subsampled Gaussian."""

    def __init__(self, orders=None):
        self.orders = orders or list(np.concatenate([
            np.arange(1.25, 10.0, 0.25),
            np.arange(10, 256, 1),
        ]))
        self.rdpvals = np.zeros_like(self.orders, dtype=float)

    def AddGaussianRound(self, q: float, noisemu: float, steps: int = 1):
        """Accumulate RDP cost for one or more rounds."""
        for idx, alpha in enumerate(self.orders):
            rdpval = self.ComputeRdp(q, noisemu, alpha)
            self.rdpvals[idx] += rdpval * steps

    def GetEps(self, delta: float):
        """Convert total RDP to (epsilon, delta)-DP."""
        if delta <= 0:
            raise ValueError("delta must be positive")

        ordersarr = np.array(self.orders, dtype=float)
        epsvec = self.rdpvals + math.log(1.0 / delta) / (ordersarr - 1.0)
        bestidx = np.nanargmin(epsvec)
        return float(epsvec[bestidx]), float(self.orders[bestidx])

    def ComputeRdp(self, q: float, noisemu: float, alpha: float) -> float:
        """RDP of Poisson-subsampled Gaussian (simple upper bound)."""
        if noisemu <= 0:
            return float("inf")
        if q <= 0:
            return 0.0
        return (q * q) * alpha / (2.0 * noisemu * noisemu)

    @property
    def RdpVec(self) -> np.ndarray:
        return self.rdpvals.copy()
