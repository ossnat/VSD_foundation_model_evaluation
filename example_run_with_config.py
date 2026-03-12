#!/usr/bin/env python3
"""
Example: run Frodo classification from Colab (or any script) by passing a config dict to main().
Same config as the default fallback in run_frodo_classification.py.

In Colab:
  1. Clone/copy the repo and put data in place (e.g. data_2026/data/frodo_early/ with vertical_data.npy, horizontal_data.npy).
  2. %cd into the repo root.
  3. Run the code below (or import and run).
"""

import sys
import os

# If running from Colab, ensure project root is in path and cwd
# REPO_ROOT = "/content/VSD_foundation_model_evaluation"  # adjust to your Colab path
# os.chdir(REPO_ROOT)
# sys.path.insert(0, REPO_ROOT)

from run_frodo_classification import main

# Config dict (same as current fallback). Adjust dataset_path to your data location.
config = {
    "training": {
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 0.0005,
        "device": "auto",
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_patience": 2,
        "gradient_clip": 1.0,
    },
    "data": {
        "dataset_path": "data_2026/data/frodo_early",  # change in Colab if your data is elsewhere
        "num_frames": 5,
        "clip_stride": 1,   # 1 = max overlap; set to 5 for non-overlapping clips
        "num_workers": 0,
        "start_frame": 31,
        "end_frame": 40,
        "frame_height": 100,
        "frame_width": 100,
    },
    "split": {
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        "seed": 42,
    },
    "model": {
        "backbone": "frodo_resnet",
        "temporal_pooling": "mean",
        "embedding_dim": 512,
        "num_classes": 2,
    },
    "output": {
        "results_dir": "results_frodo_classification",
        "class_names": ["vertical", "horizontal"],
    },
    "plot": {
        "n_per_category": 2,
        "confusion_matrix_dpi": 120,
        "sample_clips_dpi": 120,
    },
}

if __name__ == "__main__":
    main(config)
