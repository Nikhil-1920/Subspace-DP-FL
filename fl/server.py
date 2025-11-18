import torch

class ServerState:
    def __init__(self, model, U=None, lam=None, tau=1.0, fairnesscfg=None):
        self.model = model
        self.U = U
        self.lam = lam
        self.tau = tau
        self.fairnesscfg = fairnesscfg

    def ApplyAvgDelta(self, avgdelta, lr=1.0):
        offset = 0
        with torch.no_grad():
            for p in self.model.parameters():
                n = p.numel()
                p.add_(avgdelta[offset:offset + n].view_as(p), alpha=lr)
                offset += n
        return self

    def UpdateParams(self, avgdelta, lr=1.0):
        return self.ApplyAvgDelta(avgdelta, lr=lr)

    def RunPmu(self, clientupdates, clientgroups=None, C_S=None, sigma_S=None):
        return self
