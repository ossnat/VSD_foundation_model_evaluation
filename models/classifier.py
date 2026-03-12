import torch.nn as nn
from .backbones import build_backbone
from .temporal_pooling import build_pooling

class VideoClassifier(nn.Module):
    def __init__(self, backbone_name, temporal_pooling, embedding_dim, num_classes, pooling_kwargs=None):
        super().__init__()
        self.backbone = build_backbone(backbone_name, embedding_dim)
        self.pooling = build_pooling(
            temporal_pooling, embedding_dim, **(pooling_kwargs or {})
        )
        self.fc = nn.Linear(embedding_dim, num_classes)

    def freeze_pretrained(self):
        """Freeze backbone and temporal pooling (e.g. LSTM); only the classifier head (fc) is trained."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.pooling.parameters():
            p.requires_grad = False

    def forward(self, clips, B, T):
        feats = self.backbone(clips)       # (B*T, D)
        pooled = self.pooling(feats, B, T) # (B, D)
        return self.fc(pooled)             # (B, num_classes)
