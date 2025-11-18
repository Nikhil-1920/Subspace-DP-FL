import torch.nn as nn
from torchvision.models import mobilenet_v2

def MakeMobilenetV2(numclasses: int = 10) -> nn.Module:
    """MobileNetV2 for CIFAR images."""
    model = mobilenet_v2(weights=None)

    firstconv = model.features[0][0]
    firstconv.stride = (1, 1)

    try:
        model.classifier[0] = nn.Identity()
    except Exception:
        pass

    infeatures = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(infeatures, numclasses)
    return model
