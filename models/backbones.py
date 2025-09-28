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

class Dino2DBackbone(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        try:
            dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        except Exception as e:
            print(f"-- Dino2DBackbone: Error loading DINOv2 model from torch.hub: {e}")
        self.dino_encoder = dino_model  # Use the loaded DINOv2 model

        # DINOv2 outputs embeddings of dimension DINO_BASE_EMBEDDING_DIM
        # Add a linear layer if the desired embedding_dim is different
        DINO_BASE_EMBEDDING_DIM = 768  # Standard for ViT-Base

        if embedding_dim != DINO_BASE_EMBEDDING_DIM:
            self.fc = nn.Linear(DINO_BASE_EMBEDDING_DIM, embedding_dim)
        else:
            self.fc = nn.Identity()  # If dimensions match, use identity

    def forward(self, x):
        # Input x shape: (B*T, C, H, W) - assuming C=3 and H,W are compatible with DINOv2 input (e.g., 224x224)
        # DINOv2 forward pass: it expects input shape (B, C, H, W)
        # Since our input is (B*T, C, H, W), we can pass it directly
        # DINOv2's forward returns a tuple: (cls_token_embedding, patch_token_embeddings)
        # We take the CLS token embedding (first element of the tuple)
        cls_token_embedding = self.dino_encoder(x)[0]  # Shape: (B*T, DINO_BASE_EMBEDDING_DIM)
        return self.fc(cls_token_embedding)  # Shape: (B*T, desired_embedding_dim)


def build_backbone(name: str, embedding_dim: int):
    if name == "dino_mock":
        return DinoBackbone(embedding_dim)
    elif name == 'dino2d':
        return Dino2DBackbone(embedding_dim)
    elif name == "resnet18":
        return ResNetBackbone(embedding_dim)
    else:
        raise ValueError(f"Backbone {name} not implemented")
