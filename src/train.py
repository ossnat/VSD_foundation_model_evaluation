from models.classifier import VideoClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm


def run_one_epoch(model, loader, criterion, optimizer, device, is_2d, is_train=True):
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

    is_2d = True if '2d' in config["model"]["backbone"] else False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer,
                                              device, is_2d, True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer,
                                          device, is_2d, False)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
    return history, model, device