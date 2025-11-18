import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from .noniid import DirichletSplit


def GetEmnist(
    numclients: int = 200,
    alpha: float = 0.5,
    seed: int = 0,
):
    """
    Loads and partitions the EMNIST (ByClass) dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainds = datasets.EMNIST(
        root="./data",
        split="byclass",
        train=True,
        download=True,
        transform=transform,
    )
    testds = datasets.EMNIST(
        root="./data",
        split="byclass",
        train=False,
        download=True,
        transform=transform,
    )

    clientinds = DirichletSplit(trainds.targets, numclients, alpha, seed)

    clientdata = []
    for cli, inds in enumerate(clientinds):
        subds = Subset(trainds, inds)

        isdigitvec = [trainds.targets[j] < 10 for j in inds]
        groupid = 0 if sum(isdigitvec) > len(isdigitvec) / 2 else 1

        loader = DataLoader(subds, batch_size=32, shuffle=True)
        clientdata.append({"loader": loader, "group_id": groupid})

    testloader = DataLoader(testds, batch_size=512, shuffle=False)
    return clientdata, testloader, {"num_classes": 62}
