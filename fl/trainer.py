import pandas as pd
import random
from tqdm import trange
from privacy.accountant import RDPAccountantPoisson
from fl.aggregator import AggregateRound
from eval.metrics import EvalModel

def PoissonSample(idlist, q):
    return [i for i in idlist if random.random() < q]

def TrainFederated(serverstate, clients, clientdata, testloader, cfg, accountantclass=RDPAccountantPoisson, device="cpu"):
    allclientids = list(range(len(clients)))
    roundcount = cfg["rounds"]
    q = cfg["clients_per_round_q"]
    clipnorm = cfg["C"]
    sigma = cfg["sigma"]
    pmuevery = cfg["pmu_every"]
    delta = cfg["delta"]
    noisemu = cfg["noise_multiplier"]

    accountant = accountantclass()

    logdata = []

    for t in trange(1, roundcount + 1, desc="Federated Rounds"):
        selids = PoissonSample(allclientids, q)

        if not selids:
            accountant.add_gaussian_round(q=q, noise_multiplier=noisemu, steps=1)
            continue

        updates = []
        clientgroups = []
        for idx in selids:
            upd = clients[idx].ComputeUpdate(serverstate.model, clientdata[idx]["loader"])
            updates.append(upd.to(device))
            clientgroups.append(clientdata[idx]["group_id"])

        avgupdate, clipscales = AggregateRound(
            updates,
            U=serverstate.U,
            lam=serverstate.lam,
            tau=serverstate.tau,
            clipnorm=clipnorm,
            sigma=sigma,
        )
        serverstate.UpdateParams(avgupdate)
        accountant.add_gaussian_round(q=q, noise_multiplier=noisemu, steps=1)

        if pmuevery > 0 and (t % pmuevery == 0):
            serverstate.RunPmu(
                clientupdates=updates,
                clientgroups=clientgroups,
                C_S=cfg["pmu_C_S"],
                sigma_S=cfg["pmu_sigma_sum"],
            )

        if t % cfg["eval_every"] == 0 or t == roundcount:
            eps, _ = accountant.get_eps(delta)
            metrics = EvalModel(serverstate.model, testloader, cfg["task"], device)
            avgclip = sum(clipscales) / len(clipscales) if clipscales else 1.0
            row = {
                "round": t,
                "epsilon": eps,
                "delta": delta,
                "avg_clip_scale": avgclip,
            }
            row.update(metrics)
            logdata.append(row)
            accval = metrics.get("accuracy", "N/A")
            try:
                accstr = f"{float(accval):.4f}"
            except (TypeError, ValueError):
                accstr = str(accval)
            print(f"Round {t}/{roundcount} | ε≈{eps:.2f} | Accuracy: {accstr}")

    return pd.DataFrame(logdata)