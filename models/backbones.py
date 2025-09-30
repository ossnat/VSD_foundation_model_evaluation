import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models.video import r3d_18, R3D_18_Weights
#from torchvision.models.video import r3d_18  # example for video backbone

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

from transformers import CLIPVisionModel, CLIPProcessor
import torch.nn as nn


class Dino2DBackbone(nn.Module):
    def __init__(self, embedding_dim=768, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim

        # CLIP ViT output dimensions
        if "base" in model_name.lower():
            model_output_dim = 768
        elif "large" in model_name.lower():
            model_output_dim = 1024
        elif "huge" in model_name.lower():
            model_output_dim = 1280
        else:
            model_output_dim = 768  # default

        # Optional projection
        self.fc = nn.Identity() if embedding_dim == model_output_dim else nn.Linear(model_output_dim, embedding_dim)

    def forward(self, x):
        """
        x: (B*T, C, H, W) or (B, T, C, H, W)
        returns: (B*T, embedding_dim)
        """
        # print(f"CLIP Input shape: {x.shape}")

        # Handle 5D input (B, T, C, H, W) → flatten to (B*T, C, H, W)
        if x.ndim == 5:
            # print('--- FLATTENING 5D INPUT ---')
            B, T = x.size(0), x.size(1)
            x = x.view(B * T, x.size(2), x.size(3), x.size(4))
            # print(f"After flattening: {x.shape}")

        # Forward pass through CLIP Vision Model
        # print(f"Feeding to CLIP: {x.shape}")
        outputs = self.backbone(pixel_values=x)

        # Extract pooled features (CLS token equivalent)
        features = outputs.pooler_output  # Shape: (B*T, hidden_dim)
        # print(f"CLIP features shape: {features.shape}")

        # Apply optional projection
        output_embedding = self.fc(features)
        # print(f"Final output shape: {output_embedding.shape}")

        return output_embedding


class Dino3DBackbone(nn.Module):
    """
    Pretrained 3D video backbone wrapper.
    Input:  (B, T, C, H, W) with variable T
    Output: (B, embedding_dim)
    """
    def __init__(self, embedding_dim=512, model_name="r3d_18", pretrained=True, freeze=True):
        super().__init__()
        self.embedding_dim = embedding_dim

        if model_name == "r3d_18":
            base = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
            in_feat = base.fc.in_features
            modules = list(base.children())[:-1]  # drop final FC
            self.encoder = nn.Sequential(*modules)
            self._backbone_in_features = in_feat
        else:
            raise ValueError(f"Unknown 3D backbone: {model_name}")

        # projection layer if embedding_dim != backbone output
        if embedding_dim != self._backbone_in_features:
            self.proj = nn.Linear(self._backbone_in_features, embedding_dim)
        else:
            self.proj = nn.Identity()

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: (B, T, C, H, W) → permute to (B, C, T, H, W)
        """
        if x.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W) got {x.shape}")

        # Permute so time is before spatial dims
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)

        feats = self.encoder(x)       # (B, C, 1, 1, 1)
        feats = feats.flatten(1)      # (B, C)
        return self.proj(feats)       # (B, embedding_dim)


def build_backbone(name: str, embedding_dim: int):
    if name == "dino_mock":
        return DinoBackbone(embedding_dim)
    elif name == "dino2d":
        print('--- build_backbone -- dino2d')
        # Pass along the model_name so your __init__ logic is invoked
        model_name = 'openai/clip-vit-base-patch32'
        print('--- build_backbone model name: ', model_name)
        return Dino2DBackbone(embedding_dim, model_name)
    elif name == "dino3d":
        model_name = "r3d_18"
        embedding_dim = 512
        return Dino3DBackbone(embedding_dim, model_name, pretrained=True, freeze=True)
    elif name == 'videomae':
        pass
    elif name == "resnet18":
        return ResNetBackbone(embedding_dim)
    else:
        raise ValueError(f"Backbone {name} not implemented")
