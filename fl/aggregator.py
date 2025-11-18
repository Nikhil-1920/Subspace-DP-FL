import torch

def AggregateRound(clientupdates, u=None, lam=None, tau=None, clipnorm=None, sigma=None, **kwargs):
    if u is None and "U" in kwargs:
        u = kwargs["U"]
    if lam is None and "lam" in kwargs:
        lam = kwargs["lam"]
    if tau is None and "tau" in kwargs:
        tau = kwargs["tau"]
    if clipnorm is None and "C" in kwargs:
        clipnorm = kwargs["C"]
    if sigma is None and "sigma" in kwargs:
        sigma = kwargs["sigma"]

    if not clientupdates:
        if u is not None:
            return torch.zeros_like(u[:, 0]), []
        raise ValueError("clientupdates is empty and no basis u was provided.")

    updates = torch.stack(clientupdates, dim=0)
    avgupdate = updates.mean(dim=0)
    clipscales = [1.0] * len(clientupdates)
    return avgupdate, clipscales
