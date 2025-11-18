import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from .noniid import DirichletSplit


def GetCifar10(
    numclients: int = 10,
    alpha: float = 10.0,
    seed: int = 42,
    batchsize: int = 128,
):
    """
    CIFAR-10 with standard augmentations and a Dirichlet partition.
    """
    traintfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    testtfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    trainds = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=traintfm,
    )
    testds = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=testtfm,
    )

    clientinds = DirichletSplit(trainds.targets, numclients, alpha, seed)

    clientdata = []
    for inds in clientinds:
        subds = Subset(trainds, inds)
        loader = DataLoader(subds, batch_size=batchsize, shuffle=True, drop_last=False)
        clientdata.append({"loader": loader, "group_id": 0})

    testloader = DataLoader(testds, batch_size=512, shuffle=False)
    return clientdata, testloader, {"num_classes": 10}
