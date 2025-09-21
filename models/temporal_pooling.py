import torch
import torch.nn as nn

class MeanPooling(nn.Module):
    def forward(self, x, B, T):
        return x.view(B, T, -1).mean(dim=1)

class MaxPooling(nn.Module):
    def forward(self, x, B, T):
        return x.view(B, T, -1).max(dim=1).values

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x, B, T):
        x = x.view(B, T, -1)  # (B, T, D)
        weights = torch.softmax(self.att(x), dim=1)  # (B, T, 1)
        return (weights * x).sum(dim=1)

def build_pooling(name: str, embedding_dim: int):
    if name == "mean":
        return MeanPooling()
    elif name == "max":
        return MaxPooling()
    elif name == "attention":
        return AttentionPooling(embedding_dim)
    else:
        raise ValueError(f"Pooling {name} not implemented")
