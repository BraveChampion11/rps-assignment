"""Custom (non-pretrained) CNN for Rock-Paper-Scissors + shared preprocessing.

Lighting invariance comes from two things:
  1. Training augmentation that simulates real lighting changes (brightness, contrast, gamma,
     colour jitter, grayscale, directional light ramps and soft shadows).
  2. Per-image standardisation (PerImageStandardize) applied identically in training, evaluation
     and the GUI, which removes global brightness / contrast / colour-cast differences.
"""
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

# ImageFolder sorts class folders alphabetically -> this order must match.
CLASSES = ["paper", "rock", "scissors"]
IMG_SIZE = 128


class PerImageStandardize:
    """Per-image, per-channel (x - mean) / std. Makes the input independent of overall light level."""

    def __call__(self, x):
        m = x.mean(dim=(1, 2), keepdim=True)
        s = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-2)
        return (x - m) / s


class RandomGamma:
    """Random gamma in [low, high] (log-uniform): simulates under/over-exposed cameras."""

    def __init__(self, p=0.5, low=0.5, high=2.0):
        self.p, self.low, self.high = p, low, high

    def __call__(self, img):
        if random.random() < self.p:
            g = float(np.exp(random.uniform(np.log(self.low), np.log(self.high))))
            return TF.adjust_gamma(img, g)
        return img


class RandomLightField:
    """Uneven illumination: a smooth brightness ramp (window/lamp from one side) or a soft shadow."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        a = np.asarray(img, dtype=np.float32) / 255.0
        h, w = a.shape[:2]
        ang = random.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = ((xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang)) / (0.5 * np.hypot(h, w))  # ~[-1, 1]
        if random.random() < 0.5:   # smooth ramp
            field = 1.0 + random.uniform(0.2, 0.7) * d
        else:                       # soft-edged shadow over one side
            edge, soft = random.uniform(-0.3, 0.3), random.uniform(0.05, 0.3)
            m = 1.0 / (1.0 + np.exp(-(d - edge) / soft))
            field = 1.0 - random.uniform(0.3, 0.7) * m
        a = np.clip(a * field[..., None], 0.0, 1.0)
        return Image.fromarray((a * 255).astype(np.uint8))


class LightPerturb:
    """Deterministic lighting change, used to TEST robustness (not for training)."""

    def __init__(self, brightness=1.0, contrast=1.0, gamma=1.0):
        self.b, self.c, self.g = brightness, contrast, gamma

    def __call__(self, img):
        img = TF.adjust_brightness(img, self.b)
        img = TF.adjust_contrast(img, self.c)
        return TF.adjust_gamma(img, self.g)


def eval_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        PerImageStandardize(),
    ])


def train_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        # strong geometry: the model must not rely on hand size, framing or viewpoint
        transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=35, translate=(0.1, 0.1), scale=(0.75, 1.25)),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.04),
        RandomGamma(p=0.5),
        RandomLightField(p=0.5),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        transforms.ToTensor(),
        PerImageStandardize(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])


def conv_block(cin, cout, p_drop):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(p_drop),
    )


class RPSNet(nn.Module):
    """4 conv blocks (32-64-128-256) -> global avg pool -> MLP head. ~1.2M params."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, 0.05),
            conv_block(32, 64, 0.10),
            conv_block(64, 128, 0.15),
            conv_block(128, 256, 0.20),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = RPSNet(len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt["classes"], ckpt.get("img_size", IMG_SIZE)