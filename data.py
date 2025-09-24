"""Utility helpers for loading image data from scikit-learn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from sklearn.datasets import load_digits, make_classification, make_regression

if TYPE_CHECKING:  # pragma: no cover - only for static type checking
    from synthetic_image_classification import TrainingConfig


def load_digits_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load the scikit-learn digits dataset and adapt it to the training config."""
    dataset = load_digits()
    images = dataset.images.astype(np.float32) / 16.0
    labels = dataset.target.astype(np.int32)

    images = images[..., np.newaxis]
    total_samples = images.shape[0]
    requested_samples = 1000
    rng = np.random.default_rng(42)

    if requested_samples <= 0 or requested_samples > total_samples:
        sampled_images = images
        sampled_labels = labels
        effective_samples = total_samples
    else:
        indices = rng.choice(total_samples, size=requested_samples, replace=False)
        sampled_images = images[indices]
        sampled_labels = labels[indices]
        effective_samples = requested_samples

    shuffle_indices = rng.permutation(effective_samples)
    sampled_images = sampled_images[shuffle_indices]
    sampled_labels = sampled_labels[shuffle_indices]

    return sampled_images, sampled_labels


def load_tabular_classification_dataset(n_samples: int = 1000, n_features: int = 20, n_classes: int = 3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic tabular classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.2),
        n_classes=n_classes,
        random_state=random_state
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    return X, y


def load_regression_dataset(n_samples: int = 1000, n_features: int = 20, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic tabular regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        noise=0.2,
        random_state=random_state
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    return X, y
