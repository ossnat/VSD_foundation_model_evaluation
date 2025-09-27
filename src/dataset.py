import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# --------------------------
# Dataset class
# --------------------------
from data_utils.prepare_data import preprocess_vsd_clip


class VSDClipsDataset(Dataset):
    def __init__(self, data, labels, config, clip_len=5, transform=True,
                 start_frame=27, end_frame=57):
        """
        Args:
            data: numpy array of shape (N, T, C, H, W)
            labels: numpy array of shape (N,) with trial-level labels
            clip_len: number of frames per clip
            transform: optional transform for each clip
            start_frame, end_frame: frame range to consider within each trial
        """
        self.data = data
        self.labels = labels
        self.clip_len = clip_len
        self.transform = transform
        self.model_name = config['model']['backbone']
        # Store start and end frames instead of slicing data in __init__
        self.start_frame = start_frame or 0
        self.end_frame = end_frame or data.shape[1]
        # Removed _cut_data() call here
        self.indices = self._generate_indices()

    # Removed _cut_data method

    def _generate_indices(self):
        # Each index is (trial_idx, start_frame_in_range)
        indices = []
        # Iterate over the specified frame range
        frames_in_range = self.end_frame - self.start_frame
        for trial_idx in range(len(self.data)):
            # Ensure there are enough frames in the range to form a clip
            if frames_in_range >= self.clip_len:
                for start_in_range in range(0, frames_in_range - self.clip_len + 1):
                    indices.append((trial_idx, start_in_range))
        return indices

    def __len__(self):
        # Check if indices were generated before returning length
        return len(self.indices) if hasattr(self, 'indices') else 0


    def __getitem__(self, idx):
        trial_idx, start_in_range = self.indices[idx]
        # Calculate the actual start frame in the original data
        actual_start_frame = self.start_frame + start_in_range
        # Slice the data using the actual start frame and clip length
        clip = self.data[trial_idx][actual_start_frame:actual_start_frame+self.clip_len]   # (clip_len, C, H, W)

        label = self.labels[trial_idx]                           # trial-level label
        if self.transform:
            # print("#### clip shape before preprocess: ", clip.shape)
            clip = preprocess_vsd_clip(clip, self.model_name)
            # print("#### clip shape after preprocess: ", clip.shape)
        return clip, label

# --------------------------
# Splitting function
# --------------------------
def split_trials(data, labels, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    assert np.isclose(train_size + val_size + test_size, 1.0), "splits must sum to 1"

    N = len(data)  # number of trials
    assert N == len(labels), f"data and labels must have the same length {data.shape}, {len(data)}, {len(labels)}"
    trial_indices = np.arange(N)

    # First split train vs temp
    train_idx, temp_idx = train_test_split(
        trial_indices, test_size=(1 - train_size), random_state=seed, stratify=labels
    )

    # Then split temp into val and test
    relative_val_size = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - relative_val_size), random_state=seed, stratify=labels[temp_idx]
    )

    return train_idx, val_idx, test_idx