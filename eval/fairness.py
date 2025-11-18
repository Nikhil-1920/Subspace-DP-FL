import torch
from collections import defaultdict

@torch.no_grad()
def EvalFairnessMetrics(model, testloader, numgroups, device):
    """Compute per-group accuracy and parity gap."""
    model.eval()

    groupcorrect = defaultdict(int)
    grouptotal = defaultdict(int)

    for x, y, groupid in testloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)

        corrects = pred == y
        groupids = groupid

        for i in range(numgroups):
            mask = groupids == i
            groupcorrect[i] += corrects[mask].sum().item()
            grouptotal[i] += mask.sum().item()

    pergroupaccuracy = {
        f"acc_group_{i}": groupcorrect[i] / max(1, grouptotal[i])
        for i in range(numgroups)
    }

    accvals = list(pergroupaccuracy.values())
    accuracyparity = max(accvals) - min(accvals) if accvals else 0.0

    pergroupaccuracy["accuracy_parity"] = accuracyparity
    return pergroupaccuracy
