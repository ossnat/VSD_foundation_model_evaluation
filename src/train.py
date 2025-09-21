import torch
from torch.utils.data import DataLoader

### OLD CODE
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

#### END OLD CODE

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.classifier import VideoClassifier

def run_one_epoch(model, loader, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, total_correct, total_samples = 0, 0, 0

    for clips, labels in tqdm(loader, desc="Train" if is_train else "Valid"):
        B, T, C, H, W = clips.shape
        clips, labels = clips.view(B*T, C, H, W).to(device), labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(clips, B, T)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * B
        total_correct += (preds == labels).sum().item()
        total_samples += B

    return total_loss / total_samples, total_correct / total_samples

def train_model(config, train_loader, val_loader):
    device = config["training"]["device"]
    model = VideoClassifier(
        backbone_name=config["model"]["backbone"],
        temporal_pooling=config["model"]["temporal_pooling"],
        embedding_dim=config["model"]["embedding_dim"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
