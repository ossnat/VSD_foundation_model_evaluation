from typing import Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset

class VDSClipsDataset(Dataset):
    """Dataset for VDS trials saved as numpy arrays.

    Expects two numpy files: horizontal_data.npy and vertical_data.npy,
    each shaped (n_trials, n_frames, H, W) OR (n_trials, n_frames, H*W).

    Produces clips of length clip_len sampled from each trial. If clip_len == n_frames,
    returns the whole trial.
    """

    def __init__(self,
                 horizontal_npy: str,
                 vertical_npy: str,
                 clip_len: int = 16,
                 step: int = 1,
                 transform=None,
                 preload: bool = True):
        # load arrays
        h = np.load(horizontal_npy)
        v = np.load(vertical_npy)

        # ensure shape is (n, T, H, W)
        def ensure_shape(a):
            if a.ndim == 3:
                # (n, T, H*W) -> reshape
                n, T, HW = a.shape
                side = int(np.sqrt(HW))
                return a.reshape(n, T, side, side)
            return a

        self.h = ensure_shape(h)
        self.v = ensure_shape(v)

        # create index mapping: (trial_id, start_frame, label)
        self.clip_len = clip_len
        self.step = step
        self.transform = transform

        self.samples = []  # tuples (which_array, trial_idx, start_frame, label)
        for arr, label in [(self.h, 0), (self.v, 1)]:
            n_trials, T, H, W = arr.shape
            for i in range(n_trials):
                if clip_len >= T:
                    # single sample using whole trial
                    self.samples.append((arr, i, 0, label))
                else:
                    for start in range(0, T - clip_len + 1, step):
                        self.samples.append((arr, i, start, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, trial_idx, start, label = self.samples[idx]
        clip = arr[trial_idx, start:start + self.clip_len]  # shape (clip_len, H, W)
        # convert to float32 and normalize to [0,1]
        clip = clip.astype('float32') / (clip.max() + 1e-8)
        # shape for model: (C, T, H, W)
        clip = np.expand_dims(clip, 1)  # (T, 1, H, W)
        clip = np.transpose(clip, (1, 0, 2, 3))
        if self.transform:
            clip = self.transform(clip)
        return torch.from_numpy(clip), torch.tensor(label, dtype=torch.long)
