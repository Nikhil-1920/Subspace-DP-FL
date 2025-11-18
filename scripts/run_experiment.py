from typing import Any, Dict, List
import os
import sys
import time
import yaml
import random
import argparse
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

from datasets.cifar10 import GetCifar10
from datasets.emnist import GetEmnist

from models.cifar_mobilenet import MakeMobilenetV2
from models.cifar_resnet import MakeResnet18
from models.cifar_simple_cnn import SimpleCnn
from models.emnist_cnn import EmnistCnn

resetc = "\033[0m"
boldc = "\033[1m"

fgcyan = "\033[36m"
fggreen = "\033[32m"
fgyellow = "\033[33m"
fgmagenta = "\033[35m"
fgblue = "\033[34m"
fgwhite = "\033[37m"


class PrettyLineLogger:
    def __init__(
        self,
        totalrounds: int,
        dev: torch.device,
        nparams: int,
        modelname: str,
        taskname: str,
    ):
        self.totalrounds = totalrounds
        self.dev = dev
        self.nparams = nparams
        self.bestacc = 0.0
        self.starttime = time.time()

        namemap = {
            "mobilenet_v2": "MobileNetV2",
            "simple_cnn": "SimpleCNN",
            "resnet18": "ResNet-18",
            "emnist_cnn": "EMNIST-CNN",
        }
        prettymodel = namemap.get(modelname.lower(), modelname)
        prettytask = taskname.upper()

        nowstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.titleplain = f"{prettytask}  |  {prettymodel}  |  Federated Training"
        self.infoplain = f"started={nowstr}  device={self.dev}  params={self.nparams:,}"

        worstr = max(self.totalrounds, 999)
        roundplain = (
            f"[ROUND*] {worstr:03d}/{worstr:03d}  "
            f"acc=0.0000  best=0.0000  elapsed=00:00"
        )

        self.width = max(
            len(self.titleplain),
            len(self.infoplain),
            len(roundplain),
            60,
        ) + 4

        self._Header()

    def _Sep(self):
        print(fgmagenta + "─" * self.width + resetc)

    def _Header(self):
        self._Sep()
        print(fgcyan + boldc + self.titleplain.center(self.width) + resetc)
        self._Sep()
        print(fgwhite + self.infoplain.center(self.width) + resetc)
        self._Sep()

    def WarmstartDone(self, accval: float, epochs: int):
        plain = f"[WARMUP] epochs={epochs}  acc={accval:0.4f}  elapsed={self._ElapsedStr()}"
        print(fgyellow + plain.center(self.width) + resetc)
        self._Sep()

    def LogRound(self, roundidx: int, accval: float | None):
        rstr = f"{roundidx:03d}/{self.totalrounds:03d}"

        if accval is None:
            msg = (
                f"{fgblue}[round]{resetc} {rstr}  "
                f"elapsed={fgmagenta}{self._ElapsedStr()}{resetc}"
            )
            print(msg)
            return

        if accval > self.bestacc:
            self.bestacc = accval
            tag = fggreen + "[EVAL*]" + resetc
        else:
            tag = fgcyan + "[EVAL ]" + resetc

        msg = (
            f"{tag} {rstr}  "
            f"acc={fggreen}{accval:0.4f}{resetc}  "
            f"best={fgyellow}{self.bestacc:0.4f}{resetc}  "
            f"elapsed={fgmagenta}{self._ElapsedStr()}{resetc}"
        )
        print(msg)

    def Done(self, outdir: str):
        self._Sep()
        plain = f"[DONE] best_acc={self.bestacc:0.4f}  total={self._ElapsedStr()}"
        print(fggreen + plain.center(self.width) + resetc)
        print(fgcyan + f"saved_to: {outdir}" + resetc)
        self._Sep()

    def _ElapsedStr(self) -> str:
        secs = int(time.time() - self.starttime)
        mins, secs = divmod(secs, 60)
        return f"{mins:02d}:{secs:02d}"


def SeedEverything(seedval: int) -> None:
    random.seed(seedval)
    torch.manual_seed(seedval)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seedval)


def ChooseDevice() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def FlattenParams(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


@torch.no_grad()
def LoadFlatParams(model: nn.Module, flat: torch.Tensor) -> None:
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.copy_(flat[off:off + n].view_as(p))
        off += n


def BuildOptimizerCtor(cfg: Dict[str, Any]):
    lrval = float(cfg.get("lr", 0.05))
    momval = float(cfg.get("momentum", 0.9))
    wdecay = float(cfg.get("weight_decay", 5e-4))

    def Ctor(params):
        return torch.optim.SGD(
            params,
            lr=lrval,
            momentum=momval,
            weight_decay=wdecay,
            nesterov=True,
        )

    return Ctor


def Evaluate(model: nn.Module, loader, dev: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return (correct / max(total, 1)) if total > 0 else 0.0


def FreezeBnRunningStats(model: nn.Module):
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = p.requires_grad


def SetHeadOnlyTraining(model: nn.Module):
    names = [name for name, _ in model.named_parameters()]

    if any(name.startswith("classifier") for name in names):
        headprefixes = ["classifier"]
    elif any(name.startswith("fc2") for name in names):
        headprefixes = ["fc2"]
    elif any(name.startswith("fc") for name in names):
        headprefixes = ["fc"]
    else:
        for _, p in model.named_parameters():
            p.requires_grad = True
        return

    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in headprefixes):
            p.requires_grad = True
        else:
            p.requires_grad = False


from copy import deepcopy


def LocalTrainReturnWeights(
    servermodel: nn.Module,
    loader,
    dev: torch.device,
    lossfn: nn.Module,
    optctor,
    localsteps: int = 1,
) -> torch.Tensor:
    local = deepcopy(servermodel).to(dev)
    local.train()

    FreezeBnRunningStats(local)
    SetHeadOnlyTraining(local)

    params = [p for p in local.parameters() if p.requires_grad]
    opt = optctor(params)

    for _ in range(localsteps):
        for x, y in loader:
            if x.size(0) <= 1:
                continue
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad(set_to_none=True)
            out = local(x)
            loss = lossfn(out, y)
            loss.backward()
            opt.step()

    return FlattenParams(local)


def CentralWarmstart(
    model: nn.Module,
    clientdatasets: List[Dict[str, Any]],
    dev: torch.device,
    optctor,
    epochs: int = 1,
    batchsize: int = 256,
) -> None:
    concat = ConcatDataset([c["loader"].dataset for c in clientdatasets])
    loader = DataLoader(concat, batch_size=batchsize, shuffle=True, drop_last=True)

    FreezeBnRunningStats(model)
    model.train().to(dev)
    opt = optctor(model.parameters())
    lossfn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = lossfn(out, y)
            loss.backward()
            opt.step()


def GetDataAndModel(cfg: Dict[str, Any], dev: torch.device):
    bs = int(cfg.get("batch_size", 128))
    alpha = float(cfg.get("alpha_dirichlet", 10.0))
    seedval = int(cfg.get("seed", 42))
    numclients = int(cfg.get("num_clients", 10))

    task = cfg.get("task", "cifar10").lower()

    if task == "cifar10":
        try:
            clientdata, testloader, _ = GetCifar10(
                numclients=numclients,
                alpha=alpha,
                seed=seedval,
                batchsize=bs,
            )
        except TypeError:
            clientdata, testloader, _ = GetCifar10(
                numclients=numclients,
                alpha=alpha,
                seed=seedval,
            )

        modelname = cfg.get("model", "mobilenet_v2").lower()
        if modelname == "mobilenet_v2":
            model = MakeMobilenetV2(numclasses=10)
        elif modelname == "resnet18":
            model = MakeResnet18(numclasses=10)
        elif modelname == "simple_cnn":
            model = SimpleCnn(numclasses=10)
        else:
            raise ValueError(f"Unknown CIFAR-10 model: {modelname}")

    elif task == "emnist":
        try:
            clientdata, testloader, modelparams = GetEmnist(
                numclients=numclients,
                alpha=alpha,
                seed=seedval,
                batchsize=bs,
            )
        except TypeError:
            clientdata, testloader, modelparams = GetEmnist(
                numclients=numclients,
                alpha=alpha,
                seed=seedval,
            )

        modelname = cfg.get("model", "emnist_cnn").lower()
        if modelname == "emnist_cnn":
            numclasses = int(modelparams.get("num_classes", 62))
            model = EmnistCnn(numclasses=numclasses)
        else:
            raise ValueError(f"Unknown EMNIST model: {modelname}")
    else:
        raise ValueError(f"Unsupported task: {task}")

    return clientdata, testloader, model.to(dev), task, modelname


def Train(cfg: Dict[str, Any], dev: torch.device) -> pd.DataFrame:
    SeedEverything(int(cfg.get("seed", 42)))

    clientdatasets, testloader, servermodel, taskname, modelname = GetDataAndModel(cfg, dev)
    d = sum(p.numel() for p in servermodel.parameters())

    rounds = int(cfg.get("rounds", 20))
    evalevery = int(cfg.get("eval_every", 2))
    localsteps = int(cfg.get("local_steps", 1))
    qval = float(cfg.get("clients_per_round_q", 1.0))
    N = int(cfg.get("num_clients", len(clientdatasets)))
    M = max(1, int(round(qval * N)))

    logger = PrettyLineLogger(
        totalrounds=rounds,
        dev=dev,
        nparams=d,
        modelname=modelname,
        taskname=taskname,
    )

    optctor = BuildOptimizerCtor(cfg)
    lossfn = nn.CrossEntropyLoss()

    warmepochs = int(cfg.get("warmstart_central_epochs", 1))
    if warmepochs > 0:
        CentralWarmstart(
            servermodel,
            clientdatasets,
            dev,
            optctor,
            epochs=warmepochs,
            batchsize=int(cfg.get("warmstart_batch_size", 256)),
        )
        warmacc = Evaluate(servermodel, testloader, dev)
        logger.WarmstartDone(warmacc, warmepochs)

    FreezeBnRunningStats(servermodel)
    SetHeadOnlyTraining(servermodel)

    rows: List[Dict[str, Any]] = []
    allids = list(range(N))

    for r in range(1, rounds + 1):
        selids = random.sample(allids, M)

        clientws: List[torch.Tensor] = []
        for cid in selids:
            wvec = LocalTrainReturnWeights(
                servermodel=servermodel,
                loader=clientdatasets[cid]["loader"],
                dev=dev,
                lossfn=lossfn,
                optctor=optctor,
                localsteps=localsteps,
            )
            clientws.append(wvec)

        avgw = torch.stack(clientws, dim=0).mean(dim=0).to(dev)
        LoadFlatParams(servermodel, avgw)

        accval = None
        if (r % evalevery == 0) or (r == rounds):
            accval = Evaluate(servermodel, testloader, dev)

        logger.LogRound(r, accval)
        rows.append({"round": r, "accuracy": accval})

    return pd.DataFrame(rows)


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("experiment_name", "cifar10_mobilenet_headonly")
    cfg.setdefault("task", "cifar10")
    cfg.setdefault("model", "mobilenet_v2")

    cfg.setdefault("num_clients", 10)
    cfg.setdefault("clients_per_round_q", 1.0)
    cfg.setdefault("local_steps", 1)
    cfg.setdefault("rounds", 20)
    cfg.setdefault("eval_every", 2)

    cfg.setdefault("batch_size", 128)
    cfg.setdefault("alpha_dirichlet", 10.0)
    cfg.setdefault("seed", 42)

    cfg.setdefault("lr", 0.05)
    cfg.setdefault("momentum", 0.9)
    cfg.setdefault("weight_decay", 5e-4)

    cfg.setdefault("warmstart_central_epochs", 1)
    cfg.setdefault("warmstart_batch_size", 256)

    dev = ChooseDevice()
    starttime = time.time()
    frame = Train(cfg, dev)
    durmin = (time.time() - starttime) / 60.0

    outdir = os.path.join(
        rootdir,
        "results",
        f"{cfg['experiment_name']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    os.makedirs(outdir, exist_ok=True)
    frame.to_csv(os.path.join(outdir, "metrics.csv"), index=False)
    with open(os.path.join(outdir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"✅ Results saved to {outdir} (elapsed {durmin:.1f} min)")


if __name__ == "__main__":
    Main()
