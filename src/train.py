import torch
from torch.utils.data import DataLoader


def train_loop(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device).float()
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


def eval_loop(model, dataloader, device):
    model.eval()
    ys, yps = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device).float()
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            ys.append(y.cpu().numpy())
            yps.append(preds.cpu().numpy())
    import numpy as np
    ys = np.concatenate(ys)
    yps = np.concatenate(yps)
    return ys, yps