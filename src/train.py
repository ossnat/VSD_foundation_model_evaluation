from models.classifier import VideoClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm


def run_one_epoch(model, loader, criterion, optimizer, device, is_2d, is_train=True, max_grad_norm=None):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, total_correct, total_samples = 0, 0, 0

    for clips, labels in tqdm(loader, desc="Train" if is_train else "Valid"):
        B, T, C, H, W = clips.shape
        if is_2d:
            clips, labels = clips.view(B*T, C, H, W).to(device), labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(clips, B, T)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * B
        total_correct += (preds == labels).sum().item()
        total_samples += B

    return total_loss / total_samples, total_correct / total_samples

def train_model(config, train_loader, val_loader):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    device = config["training"]["device"]
    print('#######', device)
    if device == "auto":
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      device = torch.device(device)
    model = VideoClassifier(
        backbone_name=config["model"]["backbone"],
        temporal_pooling=config["model"]["temporal_pooling"], #'attention', #
        embedding_dim=config["model"]["embedding_dim"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    is_2d = ('2d' in config["model"]["backbone"] or
              config["model"]["backbone"] in ("resnet18", "frodo_resnet"))

    criterion = nn.CrossEntropyLoss()
    lr = config["training"].get("learning_rate", 0.001)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    max_grad_norm = config["training"].get("gradient_clip")

    scheduler = None
    if config["training"].get("lr_scheduler") == "ReduceLROnPlateau":
        patience = config["training"].get("lr_scheduler_patience", 3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience, verbose=True
        )

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer,
            device, is_2d, True, max_grad_norm=max_grad_norm
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer,
            device, is_2d, False
        )
        if scheduler is not None:
            scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, val_loss={val_loss:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
    return history, model, device