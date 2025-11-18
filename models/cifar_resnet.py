import torch.nn as nn
from torchvision.models import resnet18

def MakeResnet18(numclasses: int = 10) -> nn.Module:
    """ResNet-18 adapted for CIFAR images."""
    try:
        model = resnet18(weights=None)
    except TypeError:
        model = resnet18(pretrained=False)

    model.conv1.kernel_size = (3, 3)
    model.conv1.stride = (1, 1)
    model.conv1.padding = (1, 1)
    model.maxpool = nn.Identity()

    infeatures = model.fc.in_features
    model.fc = nn.Linear(infeatures, numclasses)
    return model
