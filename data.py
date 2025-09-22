"""Data generation and configuration for toy dataset."""
import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

def make_toy_dataset() -> Tuple[Tensor, Tensor]:
    """Generate a more complex, multi-class, noisy dataset."""
    torch.manual_seed(10)
    random.seed(10)

    # Three classes: inner circle, middle ring, outer ring
    num_classes = 3
    num_samples = 1000
    radii = torch.empty(num_samples)
    angles = torch.rand(num_samples) * 2 * math.pi
    labels = torch.empty(num_samples, dtype=torch.long)

    for i in range(num_samples):
        p = random.random()
        if p < 0.33:
            # Inner circle, class 0
            radii[i] = torch.distributions.Normal(1.5, 0.4).sample()
            labels[i] = 0
        elif p < 0.66:
            # Middle ring, class 1
            radii[i] = torch.distributions.Normal(3.0, 0.5).sample()
            labels[i] = 1
        else:
            # Outer ring, class 2
            radii[i] = torch.distributions.Normal(4.5, 0.6).sample()
            labels[i] = 2

    # Add noise and overlap
    radii += torch.randn(num_samples) * 0.5
    angles += torch.randn(num_samples) * 0.2

    xs = torch.stack((radii * torch.cos(angles), radii * torch.sin(angles)), dim=1)
    ys = labels
    return xs, ys

def split_dataset(xs: Tensor, ys: Tensor, train_ratio: float) -> Tuple[TensorDataset, TensorDataset]:
    num_train = int(len(xs) * train_ratio)
    indices = torch.randperm(len(xs))
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    return (
        TensorDataset(xs[train_idx], ys[train_idx]),
        TensorDataset(xs[val_idx], ys[val_idx]),
    )
