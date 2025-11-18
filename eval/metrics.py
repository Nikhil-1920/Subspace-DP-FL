import torch
from sklearn.metrics import roc_auc_score
import numpy as np

@torch.no_grad()
def EvalModel(model, loader, task, device):
    """Evaluate model for a given task."""
    model.eval()
    if task in ["cifar10", "emnist"]:
        return {"accuracy": EvalClassification(model, loader, device)}
    if task == "movielens":
        return {"auc": EvalRecsys(model, loader, device)}
    return {}


def EvalClassification(model, loader, device):
    """Top-1 accuracy."""
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def EvalRecsys(model, loader, device):
    """AUC for recommendation."""
    allpreds = []
    alllabels = []
    for users, items, ratings in loader:
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        preds = model(users, items)
        allpreds.append(preds.cpu().numpy())
        alllabels.append(ratings.cpu().numpy())

    predsflat = np.concatenate(allpreds)
    labelsflat = np.concatenate(alllabels)

    if len(np.unique(labelsflat)) < 2:
        return 0.5
    return roc_auc_score(labelsflat, predsflat)
