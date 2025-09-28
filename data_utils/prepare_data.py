import numpy as np
import torch
import torch.nn.functional as F
import numpy as np # Import numpy

def preprocess_vsd_clip(clip, model_name):
    """
    clip: torch.Tensor or np.ndarray with shape [T, C, H, W] or [T, H, W]
    """
    # Convert numpy array to torch tensor if necessary
    if isinstance(clip, np.ndarray):
        clip = torch.from_numpy(clip)

    # Ensure float
    clip = clip.float()

    # If missing channel dim → add one
    if clip.ndim == 3:  # [T, H, W]
        clip = clip.unsqueeze(1)  # → [T, 1, H, W]

    elif clip.ndim == 2:  # [H, W], single frame
        clip = clip.unsqueeze(0).unsqueeze(0)  # → [1, 1, H, W]

    # Now guaranteed [T, C, H, W]

    if "dino".lower() in model_name.lower():
        # Expand grayscale → RGB
        if clip.shape[1] == 1:
            clip = clip.repeat(1, 3, 1, 1)  # → [T, 3, H, W]

        # Resize frames to 224×224
        clip = F.interpolate(clip, size=(224, 224), mode="bilinear", align_corners=False)

    elif "resnet".lower() in model_name.lower():
        if clip.shape[1] == 1:
            clip = clip.repeat(1, 3, 1, 1)
        clip = F.interpolate(clip, size=(224, 224), mode="bilinear", align_corners=False)

    return clip


import numpy as np


def prepare_vsd_data(path, H=100, W=100):
    """
    Load raw VSD npy file and reshape to (N, T, C, H, W).

    Args:
        path: str, path to .npy file
        H, W: height and width of frames (default 100x100)

    Returns:
        data: np.ndarray of shape (N, T, 1, H, W)
    """
    raw = np.load(path)  # (N, T, H*W)
    N, T, flat_dim = raw.shape
    assert flat_dim == H * W, f"Expected {H * W}, got {flat_dim}"

    data = raw.reshape(N, T, H, W)  # (N, T, H, W)
    data = np.expand_dims(data, axis=2)  # (N, T, C=1, H, W)
    return data.astype(np.float32)


def get_train_val_test_loaders(data, labels, VSDClipsDataset, DataLoader, config,
                               train_idx, val_idx, test_idx,
                               clip_len=5,num_workers=2):
    # Build datasets
    print(f"get_train_val_test_loaders: Shape of data and labels: {data.shape}, {labels.shape}")
    print(f"get_train_val_test_loaders: Length of train_idx: {len(train_idx)}")

    train_dataset = VSDClipsDataset(data[train_idx], labels[train_idx], config, clip_len)
    val_dataset = VSDClipsDataset(data[val_idx], labels[val_idx], config, clip_len)
    test_dataset = VSDClipsDataset(data[test_idx], labels[test_idx], config, clip_len)
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader