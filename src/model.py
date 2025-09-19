import torch
import torch.nn as nn


class VideoBackboneWrapper(nn.Module):
    """Wraps a pretrained video or image backbone and attaches a small classifier head.

    Two usage modes:
     1) Use a pre-trained video model (e.g., VideoMAE, TimeSformer) as backbone.
     2) Use an image model (e.g., DINO/DINOv2 ViT) applied per-frame + temporal pooling.

    We'll implement both strategies.
    """

    def __init__(self, backbone, feat_dim: int, n_classes: int = 2, temporal_pool: str = 'avg'):
        super().__init__()
        self.backbone = backbone
        self.temporal_pool = temporal_pool
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim // 2, n_classes)
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # Strategy A: backbone expects video tensor -> returns (B, feat_dim)
        try:
            feats = self.backbone(x)
            # assume backbone returns features shaped (B, feat_dim)
        except Exception:
            # Strategy B: apply backbone per frame (image model)
            # collapse batch and time: (B*T, C, H, W)
            x2 = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
            x2 = x2.view(B * T, C, H, W)
            per_frame_feats = self.backbone(x2)  # expect (B*T, feat_dim)
            feat_dim = per_frame_feats.shape[-1]
            per_frame_feats = per_frame_feats.view(B, T, feat_dim)
            if self.temporal_pool == 'avg':
                feats = per_frame_feats.mean(dim=1)
            elif self.temporal_pool == 'max':
                feats, _ = per_frame_feats.max(dim=1)
            else:
                feats = per_frame_feats.mean(dim=1)
        out = self.classifier(feats)
        return out
