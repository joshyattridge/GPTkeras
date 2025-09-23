"""Dataset utilities powered by scikit-learn."""
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from sklearn.preprocessing import StandardScaler
import openml


def make_toy_dataset() -> Tuple[Tensor, Tensor]:
    """Load the Higgs dataset from OpenML, normalize features, and return tensors."""
    # Higgs dataset: OpenML ID 23512 (1.1M rows, 28 features)
    print("Downloading Higgs dataset from OpenML (ID 23512)...")
    dataset = openml.datasets.get_dataset(23512)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    # Only use a subset for memory reasons (e.g., first 10000 rows)
    X = X.iloc[:10000]
    y = y.iloc[:10000]

    scaler = StandardScaler()
    normalized = scaler.fit_transform(X)

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    xs = torch.tensor(normalized, dtype=torch.float32)
    ys = torch.tensor(y_encoded, dtype=torch.long)
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
