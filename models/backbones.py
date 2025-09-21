import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models.video import r3d_18  # example for video backbone

class DinoBackbone(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        # dummy Dino backbone (replace with real)
        self.encoder = nn.Linear(224*224*3, embedding_dim)

    def forward(self, x):
        # flatten input: (B*T, C, H, W) -> (B*T, -1)
        return self.encoder(x.view(x.size(0), -1))


class ResNetBackbone(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove fc
        self.fc = nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        feats = self.encoder(x).flatten(1)
        return self.fc(feats)


def build_backbone(name: str, embedding_dim: int):
    if name == "dino":
        return DinoBackbone(embedding_dim)
    elif name == "resnet18":
        return ResNetBackbone(embedding_dim)
    else:
        raise ValueError(f"Backbone {name} not implemented")
