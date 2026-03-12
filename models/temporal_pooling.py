import torch
import torch.nn as nn


class LSTMPooling(nn.Module):
    """LSTM over the sequence of frame embeddings; output projected back to embedding_dim for the classifier."""
    def __init__(self, embedding_dim: int, hidden_size: int = 256, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out = hidden_size * self.num_directions
        self.proj = nn.Linear(lstm_out, embedding_dim)

    def forward(self, x, B, T):
        # x: (B*T, D) -> (B, T, D)
        x = x.view(B, T, -1)
        out, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, B, hidden_size) -> take last layer
        h_last = h_n[-self.num_directions:].transpose(0, 1).contiguous().view(B, -1)
        return self.proj(h_last)


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

class IdentityPooling(nn.Module):
    def forward(self, x, B, T):
        return x #return x.view(B, T, -1)

def build_pooling(name: str, embedding_dim: int, **kwargs):
    if name == "mean":
        return MeanPooling()
    elif name == "max":
        return MaxPooling()
    elif name == "attention":
        return AttentionPooling(embedding_dim)
    elif name == "none":
        return IdentityPooling()
    elif name == "lstm":
        return LSTMPooling(
            embedding_dim,
            hidden_size=kwargs.get("lstm_hidden_size", 256),
            num_layers=kwargs.get("lstm_num_layers", 1),
            bidirectional=kwargs.get("lstm_bidirectional", False),
        )
    else:
        raise ValueError(f"Pooling {name} not implemented")
