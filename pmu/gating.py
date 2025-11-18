import torch

@torch.no_grad()
def DpGateStat(prevs: torch.Tensor, currs: torch.Tensor) -> float:
    """Frobenius norm gate statistic between two sketches."""
    diff = prevs - currs
    return diff.norm(p="fro").item()

@torch.no_grad()
def GateDecision(gatestat: float, threshold: float) -> bool:
    """Return True if gate should trigger PMU."""
    return gatestat >= threshold
