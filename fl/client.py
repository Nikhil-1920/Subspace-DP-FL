import torch
from torch import nn
from copy import deepcopy

class LocalClient:
    def __init__(self, model, lossfn=None, optctor=None, localsteps=1, device="cpu", useamp=False, **kwargs):
        if lossfn is None and "loss_fn" in kwargs:
            lossfn = kwargs["loss_fn"]
        if optctor is None and "optimizer_ctor" in kwargs:
            optctor = kwargs["optimizer_ctor"]
        if "local_steps" in kwargs:
            localsteps = kwargs["local_steps"]

        self.model = deepcopy(model).to(device)
        self.lossfn = lossfn
        self.optctor = optctor
        self.localsteps = localsteps
        self.device = device
        self.useamp = useamp
        self.scaler = torch.cuda.amp.GradScaler() if useamp else None

    def ComputeUpdate(self, globalmodel, loader):
        if self.optctor is None or self.lossfn is None:
            raise ValueError("LocalClient requires lossfn and optctor to be set.")

        self.model.load_state_dict(globalmodel.state_dict())
        opt = self.optctor(self.model.parameters())
        self.model.train()

        for _ in range(self.localsteps):
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    raise ValueError("Each batch must provide at least (inputs, targets).")

                x = x.to(self.device)
                y = y.to(self.device)
                opt.zero_grad()

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = self.model(x)
                        loss = self.lossfn(out, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    out = self.model(x)
                    loss = self.lossfn(out, y)
                    loss.backward()
                    opt.step()

        parts = []
        with torch.no_grad():
            for lp, gp in zip(self.model.parameters(), globalmodel.parameters()):
                parts.append((lp.detach() - gp.detach()).flatten())
        return torch.cat(parts)
