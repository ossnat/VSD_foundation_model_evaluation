#!/usr/bin/env python3
"""
Run horizontal vs vertical VSD trial classification using the Frodo ResNet backbone.
Data: data_2026/data/frodo_early_0 (vertical_data.npy, horizontal_data.npy).
Trains, evaluates on test set, reports metrics, and plots sample correct/misclassified trials.
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_utils.prepare_data import prepare_vsd_data, get_train_val_test_loaders
from src.dataset import VSDClipsDataset, split_trials
from src.train import train_model
from config.config_utils import get_embedding_dim

try:
    import yaml
except ImportError:
    yaml = None


def load_config_from_yaml(path):
    """Load config from YAML file; return dict with defaults for missing sections."""
    defaults = {
        "split": {"train_size": 0.7, "val_size": 0.15, "test_size": 0.15, "seed": 42},
        "output": {"results_dir": "results_frodo_classification", "class_names": ["vertical", "horizontal"]},
        "plot": {"n_per_category": 2, "confusion_matrix_dpi": 120, "sample_clips_dpi": 120},
        "data": {"frame_height": 100, "frame_width": 100},
    }
    if yaml is None or not os.path.isfile(path):
        return None
    with open(path) as f:
        config = yaml.safe_load(f)
    if not config:
        return None
    for section, default in defaults.items():
        if section not in config:
            config[section] = default
        else:
            for k, v in default.items():
                if k not in config[section]:
                    config[section][k] = v
    return config


def get_config(data_dir=None):
    """Load config from YAML; override dataset_path with data_dir if provided."""
    config_path = os.path.join(ROOT, "config", "model_config.yaml")
    config = load_config_from_yaml(config_path)
    # Force fallback (frodo_resnet + data_2026/data/frodo_early): ignore YAML
    config = None
    if config is None:
        config = {
            "training": {"epochs": 10, "batch_size": 16, "learning_rate": 0.001, "device": "auto"},
            "data": {
                "dataset_path": data_dir or "data_2026/data/frodo_early",
                "num_frames": 5, "num_workers": 0, "start_frame": 28, "end_frame": 58,
                "frame_height": 100, "frame_width": 100,
            },
            "split": {"train_size": 0.7, "val_size": 0.15, "test_size": 0.15, "seed": 42},
            "model": {"backbone": "frodo_resnet", "temporal_pooling": "mean", "embedding_dim": 512, "num_classes": 2},
            "output": {"results_dir": "results_frodo_classification", "class_names": ["vertical", "horizontal"]},
            "plot": {"n_per_category": 2, "confusion_matrix_dpi": 120, "sample_clips_dpi": 120},
        }
    if data_dir is not None:
        config["data"]["dataset_path"] = data_dir
    config["model"]["embedding_dim"] = get_embedding_dim(config)
    return config


def load_data(data_dir, config=None):
    """Load vertical and horizontal VSD data; labels: 0=vertical, 1=horizontal."""
    vertical_path = os.path.join(data_dir, "vertical_data.npy")
    horizontal_path = os.path.join(data_dir, "horizontal_data.npy")
    if not os.path.isfile(vertical_path) or not os.path.isfile(horizontal_path):
        raise FileNotFoundError(
            f"Expected {vertical_path} and {horizontal_path}. Check data_dir."
        )
    data_cfg = (config or {}).get("data", {})
    H = data_cfg.get("frame_height", 100)
    W = data_cfg.get("frame_width", 100)
    vertical_data = prepare_vsd_data(vertical_path, H=H, W=W)
    horizontal_data = prepare_vsd_data(horizontal_path, H=H, W=W)
    vertical_labels = np.zeros(len(vertical_data), dtype=np.int64)
    horizontal_labels = np.ones(len(horizontal_data), dtype=np.int64)
    data = np.concatenate([vertical_data, horizontal_data], axis=0)
    labels = np.concatenate([vertical_labels, horizontal_labels], axis=0)
    return data, labels


def evaluate(model, test_loader, device, is_2d=True):
    """Run model on test set; return predictions, labels, and per-sample info for plotting."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    # Store (clip_batch, label_batch, pred_batch, batch_start_idx) for later sampling
    batch_infos = []
    global_idx = 0
    with torch.no_grad():
        for clips, labels in tqdm(test_loader, desc="Evaluate"):
            B, T, C, H, W = clips.shape
            if is_2d:
                clips_flat = clips.view(B * T, C, H, W).to(device)
            else:
                clips_flat = clips.to(device)
            labels = labels.to(device)
            logits = model(clips_flat, B, T)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            batch_infos.append((clips.cpu().numpy(), labels.cpu().numpy(), preds.cpu().numpy(), global_idx))
            global_idx += B
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    return all_preds, all_labels, all_probs, batch_infos


def plot_samples(batch_infos, all_labels, all_preds, save_path, n_per_category=2, dpi=120):
    """
    Plot sample clips: correct vertical, correct horizontal, false vertical, false horizontal.
    Each row is a category; we show one frame (e.g. middle) or mean over time per clip.
    """
    # Flatten to per-sample (trial) level: batch_infos has (clips, labels, preds, start_idx)
    # We need to pick specific *trials* (B in each batch). So we have (batch_idx, sample_idx_in_batch).
    samples = {
        "true_vertical": [],   # label=0, pred=0
        "true_horizontal": [], # label=1, pred=1
        "false_vertical": [],  # label=1, pred=0
        "false_horizontal": [],# label=0, pred=1
    }
    idx = 0
    for clips_b, labels_b, preds_b, _ in batch_infos:
        for i in range(len(labels_b)):
            lb, pr = int(labels_b[i]), int(preds_b[i])
            clip = clips_b[i]  # (T, C, H, W)
            if lb == 0 and pr == 0:
                samples["true_vertical"].append((clip, idx))
            elif lb == 1 and pr == 1:
                samples["true_horizontal"].append((clip, idx))
            elif lb == 1 and pr == 0:
                samples["false_vertical"].append((clip, idx))
            else:
                samples["false_horizontal"].append((clip, idx))
            idx += 1
    # Take up to n_per_category from each; show middle frame or mean
    fig, axes = plt.subplots(4, n_per_category, figsize=(4 * n_per_category, 12))
    if n_per_category == 1:
        axes = axes[:, None]
    titles = [
        "Correct: Vertical (true=0, pred=0)",
        "Correct: Horizontal (true=1, pred=1)",
        "Wrong: true Horizontal, pred Vertical",
        "Wrong: true Vertical, pred Horizontal",
    ]
    for row, (key, title) in enumerate(zip(
        ["true_vertical", "true_horizontal", "false_vertical", "false_horizontal"],
        titles,
    )):
        chosen = samples[key][:n_per_category]
        for col in range(n_per_category):
            ax = axes[row, col]
            if col < len(chosen):
                clip, _ = chosen[col]
                # clip (T, C, H, W); show mean over time then squeeze channel
                frame = np.mean(clip, axis=0).squeeze()
                ax.imshow(frame, cmap="gray")
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved sample plot to {save_path}")


def main():
    config = get_config()
    data_dir = config["data"]["dataset_path"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)
    out_cfg = config.get("output", {})
    out_dir = os.path.join(ROOT, out_cfg.get("results_dir", "results_frodo_classification"))
    os.makedirs(out_dir, exist_ok=True)

    device = config["training"].get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    is_2d = config["model"]["backbone"] in ("resnet18", "frodo_resnet") or "2d" in config["model"]["backbone"]

    print("Loading data from", data_dir)
    data, labels = load_data(data_dir, config)
    print(f"Data shape: {data.shape}, labels: {labels.shape}")

    split_cfg = config.get("split", {})
    train_idx, val_idx, test_idx = split_trials(
        data, labels,
        train_size=split_cfg.get("train_size", 0.7),
        val_size=split_cfg.get("val_size", 0.15),
        test_size=split_cfg.get("test_size", 0.15),
        seed=split_cfg.get("seed", 42),
    )
    print(f"Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")

    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        data, labels, VSDClipsDataset, DataLoader, config,
        train_idx, val_idx, test_idx,
        clip_len=config["data"]["num_frames"],
    )

    print("Training...")
    history, model, _ = train_model(config, train_loader, val_loader)

    print("Evaluating on test set...")
    all_preds, all_labels, all_probs, batch_infos = evaluate(
        model, test_loader, device, is_2d
    )

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(all_labels, all_preds)
    class_names = out_cfg.get("class_names", ["vertical", "horizontal"])
    report = classification_report(all_labels, all_preds, target_names=class_names)

    print("\n--- Test metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(report)

    # Save metrics
    metrics_path = os.path.join(out_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nAUC: {auc:.4f}\n\nConfusion matrix:\n{cm}\n\n{report}")
    print(f"Saved metrics to {metrics_path}")

    # Confusion matrix figure
    plot_cfg = config.get("plot", {})
    if HAS_MATPLOTLIB:
        dpi_cm = plot_cfg.get("confusion_matrix_dpi", 120)
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, values_format="d")
        plt.title("Test set confusion matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=dpi_cm, bbox_inches="tight")
        plt.close()
        print(f"Saved confusion matrix to {out_dir}/confusion_matrix.png")
        dpi_samples = plot_cfg.get("sample_clips_dpi", 120)
        plot_samples(
            batch_infos, all_labels, all_preds,
            save_path=os.path.join(out_dir, "sample_clips.png"),
            n_per_category=plot_cfg.get("n_per_category", 2),
            dpi=dpi_samples,
        )
    else:
        print("matplotlib not available; skipping confusion matrix and sample plots.")


if __name__ == "__main__":
    main()
