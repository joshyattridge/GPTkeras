"""Dataset utilities powered by scikit-learn."""
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def make_toy_dataset() -> Tuple[Tensor, Tensor]:
    """Load the Wine dataset, normalize features, and return tensors."""
    dataset = load_wine()
    features = dataset["data"]
    targets = dataset["target"]

    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)

    xs = torch.tensor(normalized, dtype=torch.float32)
    ys = torch.tensor(targets, dtype=torch.long)
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
