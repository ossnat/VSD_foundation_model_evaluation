import torch.nn.functional as F
import numpy as np
import torch

class SimpleResizeNormalize:
    def __init__(self, size=224, mean=0.5, std=0.5):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.mean = mean
        self.std = std
    def __call__(self, clip):
        # clip: (C, T, H, W)
        # print('@@@@@@', clip.shape)
        C, T, H, W = clip.shape
        clip = torch.from_numpy(clip).float()  # (C, T, H, W)
        # print('#####', clip.shape)
        clip = torch.nn.functional.interpolate(
            clip.view(C*T, 1, H, W), size=self.size, mode="bilinear", align_corners=False
        )
        clip = clip.view(C, T, self.size[0], self.size[1])
        clip = (clip - self.mean) / self.std
        # clip = clip.permute(0, 1, 2, 3)  # no-op here, but keep consistent
        return clip