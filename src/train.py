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
    pooling_kwargs = None
    if config["model"].get("temporal_pooling") == "lstm":
        pooling_kwargs = {
            "lstm_hidden_size": config["model"].get("lstm_hidden_size", 256),
            "lstm_num_layers": config["model"].get("lstm_num_layers", 1),
            "lstm_bidirectional": config["model"].get("lstm_bidirectional", False),
        }
    model = VideoClassifier(
        backbone_name=config["model"]["backbone"],
        temporal_pooling=config["model"]["temporal_pooling"],
        embedding_dim=config["model"]["embedding_dim"],
        num_classes=config["model"]["num_classes"],
        pooling_kwargs=pooling_kwargs,
    ).to(device)

    checkpoint_path = config["model"].get("checkpoint_path")
    if checkpoint_path:
        import os
        path = os.path.expanduser(checkpoint_path)
        if os.path.isfile(path):
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"Loaded model checkpoint from {path}")
        else:
            print(f"Warning: checkpoint_path not found: {path}")

    freeze_pretrained = config["model"].get("freeze_pretrained", False)
    if freeze_pretrained:
        model.freeze_pretrained()
        print("Frozen backbone and temporal pooling; training only the classifier head (fc).")

    is_2d = (
        "2d" in config["model"]["backbone"]
        or config["model"]["backbone"] in ("resnet18", "frodo_resnet")
    )

    criterion = nn.CrossEntropyLoss()
    lr = config["training"].get("learning_rate", 0.001)
    params = model.fc.parameters() if freeze_pretrained else model.parameters()
    optimizer = optim.Adam(params, lr=lr)
    max_grad_norm = config["training"].get("gradient_clip")

    scheduler = None
    if config["training"].get("lr_scheduler") == "ReduceLROnPlateau":
        patience = config["training"].get("lr_scheduler_patience", 3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience
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