import numpy as np

try:
    from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
except ImportError:
    raise ImportError("Please install tensorflow-privacy: `pip install tensorflow-privacy`")


def RdpToEpsilonDelta(rdpvec: np.ndarray, orders: list, delta: float) -> float:
    """Convert RDP to (epsilon, delta)."""
    eps, _, _ = get_privacy_spent(orders=orders, rdp=rdpvec, delta=delta)
    return eps


def ApproxGdpMuFromRdp(rdpvec: np.ndarray, orders: list) -> float:
    """Approximate GDP μ from RDP."""
    ordersarr = np.array(orders)
    rdpvec = np.array(rdpvec)

    xi = np.min(rdpvec / ordersarr)
    mu = np.sqrt(max(2.0 * xi, 1e-12))
    return mu
