from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt

from ours.fft import high_pass_fft
from ours.utils.CT_dataset import toZeroOne


# =========================
# Utils
# =========================

def to_patches(x, patch_size, step=1):
    """
    x: (B, C, H, W)
    return: (B, C * patch_size * patch_size, H', W')
    """
    x = x.unfold(2, patch_size, step).unfold(3, patch_size, step).contiguous()
    return rearrange(x, "b c p1 p2 h w -> b (c p1 p2) h w")


def compute_patch_weights_from_mask(mask, patch_size, step=1, eps=1e-6):
    """
    mask: (B, 1, H, W), binary
    return: (B, N) patch weights in [0, 1]
    """
    mask_patches = to_patches(mask, patch_size, step)
    valid_pixels = mask_patches.sum(dim=[-2, -1])
    total_pixels = patch_size * patch_size
    weights = valid_pixels / (total_pixels + eps)
    weights = weights / (weights.max() + eps)
    print(weights)
    return weights


# =========================
# Gradient operator
# =========================

class SobelGradient(nn.Module):
    """Compute gradient magnitude using Sobel operator"""

    def __init__(self, device):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0,  0,  0],
             [1,  2,  1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.sobel_x = sobel_x.to(device)
        self.sobel_y = sobel_y.to(device)

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        # if hasattr(self, 'mask') and self.mask is not None:
        #     mag = mag * self.mask
        mag = toZeroOne(torch.clamp(toZeroOne(mag), max=0.2))
        # plt.figure()
        # plt.imshow(mag.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()
        return mag


# =========================
# Windowed Gradient NCC
# =========================

class FFTNCC2d(nn.Module):
    """
    Patch-wise Gradient NCC using gradient magnitude
    """

    def __init__(self, patch_size=None, eps=1e-3, image_size=256, radius=118, device=torch.device('cpu'), step=1):
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps
        self.step = step
        self.grad = SobelGradient(device)
        self.register_buffer('mask', self.create_mask(image_size, radius, patch_size, step).to(device))
        self.ori_mask = self.mask
        self.wei = None

    @staticmethod
    def create_mask(image_size, radius, patch_size=None, step=1):
        center = (image_size - 1) / 2  # Center at 127.5 for 256x256
        y = torch.arange(image_size, dtype=torch.float32)
        x = torch.arange(image_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist_sq = (xx - center) ** 2 + (yy - center) ** 2
        mask = (dist_sq <= radius ** 2).float().unsqueeze(0).unsqueeze(0)
        if patch_size is not None:
            mask = to_patches(mask, patch_size, step)
        return mask  # Shape: (1, 1, H, W)

    def set_mask(self, tube_mask):
        if self.patch_size is not None:
            tube_mask = to_patches(tube_mask, self.patch_size, self.step)
        self.mask = tube_mask * self.ori_mask
        # plt.figure()
        # plt.imshow(self.mask.cpu().squeeze(0).permute(1, 2, 0), cmap='gray')
        # plt.show()

        # valid = self.mask.sum(dim=[-2, -1], keepdim=True) <= 30
        valid = self.mask.sum(dim=[-2, -1], keepdim=True) <= 0
        idx = torch.where(~valid)
        valid_idx = idx[1]
        self.valid_idx = valid_idx
        self.mask = self.mask[:, valid_idx, :, :]
        self.mask_sum = self.mask.sum()

        if hasattr(self, 'wei') and self.wei is not None:
            self.wei = self.wei[:, valid_idx]

    def set_wei(self, mask, step):
        if hasattr(self, 'patch_size') and self.patch_size is not None:
            wei = compute_patch_weights_from_mask(mask, self.patch_size, step)
            # a = (wei < 0.05).float().sum()
            self.wei = wei

    def forward(self, x1, x2):
        # 归一化
        x1 = toZeroOne(x1)
        x2 = toZeroOne(x2)
        x1 = high_pass_fft(x1, cutoff_ratio=0.05)
        x2 = high_pass_fft(x2, cutoff_ratio=0.05)

        # patch 化
        if self.patch_size is not None:
            x1 = to_patches(x1, self.patch_size, self.step)
            x2 = to_patches(x2, self.patch_size, self.step)

        # 有效 patch
        g1 = x1[:, self.valid_idx, :, :]
        g2 = x2[:, self.valid_idx, :, :]

        # mask
        g1 = g1 * self.mask
        g2 = g2 * self.mask
        # for i in range(g1.shape[1]):
        #     plt.figure()
        #     plt.imshow(g1[0][i].detach().cpu().squeeze(0), cmap='gray')
        #     plt.show()
        mean1 = (g1).sum(dim=[-2, -1], keepdim=True) / \
                (self.mask.sum(dim=[-2, -1], keepdim=True) + self.eps)
        mean2 = (g2).sum(dim=[-2, -1], keepdim=True) / \
                (self.mask.sum(dim=[-2, -1], keepdim=True) + self.eps)

        numerator = ((g1 - mean1) * (g2 - mean2)).sum(dim=[-2, -1])
        denom1 = torch.sqrt(((g1 - mean1) ** 2).sum(dim=[-2, -1]))
        denom2 = torch.sqrt(((g2 - mean2) ** 2).sum(dim=[-2, -1]))

        score = numerator / (denom1 * denom2 + 1e-6)

        c = g1.shape[1]

        # for i in range(score.shape[1]):
        #     plt.figure()
        #     plt.imshow(x1[0][i].detach().cpu(), cmap='gray')
        #     plt.title(f"{score[0][i]}")
        #     plt.show()
        #     plt.figure()
        #     plt.imshow(x2[0][i].detach().cpu(), cmap='gray')
        #     plt.title(f"{score[0][i]}")
        #     plt.show()
        #     print(score[0][i])

        if hasattr(self, 'wei') and self.wei is not None:
            score = (score * self.wei).sum() / c
        else:
            score = score.sum() / c

        return score


# =========================
# Multi-scale Gradient NCC
# =========================

class MultiscaleFFTNCC2d(nn.Module):
    """
    Multi-scale Gradient NCC with different patch sizes
    """

    def __init__(self, patch_sizes=[None], patch_weights=[1.0], eps=1e-5, device=torch.device('cpu'), step=[1]):
        super().__init__()
        self.scales = nn.ModuleList([
            FFTNCC2d(patch_size, device=device, step=s)
            for patch_size, s in zip(patch_sizes, step)
        ])
        self.patch_weights = patch_weights
        self.step = step

    def set_mask(self, mask):
        for s in self.scales:
            s.set_mask(mask)
    def set_wei(self, mask):
        for ncc, s in zip(self.scales, self.step):
            ncc.set_wei(mask, s)

    def forward(self, img1, img2, mask=None):
        scores = []
        for w, s in zip(self.patch_weights, self.scales):
            scores.append(w * s(img1, img2))
        return torch.stack(scores).sum()
