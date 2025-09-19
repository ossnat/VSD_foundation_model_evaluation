import torch.nn.functional as F
import numpy as np
import torch

class SimpleResizeNormalize:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, clip: np.ndarray):
        # clip: (C, T, H, W) as numpy
        C, T, H, W = clip.shape
        # convert to torch tensor for interpolation
        x = torch.from_numpy(clip)
        x = x.float()
        # resize frames independently: treat as NCHW with N=C*T
        x = x.view(C * T, 1, H, W)
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
        x = x.view(C, T, self.size[0], self.size[1])
        # optionally normalize - we leave it as [0,1]
        return x.numpy()
