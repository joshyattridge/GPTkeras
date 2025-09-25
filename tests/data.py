"""Utility helpers for loading diverse datasets from scikit-learn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_linnerud,
    load_wine,
    make_classification,
    make_multilabel_classification,
    make_regression,
)

if TYPE_CHECKING:  # pragma: no cover - only for static type checking
    from synthetic_image_classification import TrainingConfig


def load_digits_dataset(requested_samples: int | None = 1000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load the scikit-learn digits dataset and adapt it to the training config."""
    dataset = load_digits()
    images = dataset.images.astype(np.float32) / 16.0
    labels = dataset.target.astype(np.int32)

    images = images[..., np.newaxis]
    total_samples = images.shape[0]
    rng = np.random.default_rng(random_state)

    if requested_samples is None or requested_samples <= 0 or requested_samples > total_samples:
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


def load_binary_classification_dataset(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load the Breast Cancer Wisconsin dataset for binary classification."""
    dataset = load_breast_cancer()
    features = dataset.data.astype(np.float32)
    labels = dataset.target.astype(np.int32)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(features.shape[0])
    return features[indices], labels[indices]


def load_multiclass_tabular_dataset(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load the Wine dataset for multi-class classification."""
    dataset = load_wine()
    features = dataset.data.astype(np.float32)
    labels = dataset.target.astype(np.int32)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(features.shape[0])
    return features[indices], labels[indices]


def load_multioutput_regression_dataset(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load the Linnerud dataset to test multi-output regression."""
    dataset = load_linnerud()
    features = dataset.data.astype(np.float32)
    targets = dataset.target.astype(np.float32)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(features.shape[0])
    return features[indices], targets[indices]


def load_multilabel_classification_dataset(n_samples: int = 1000, n_features: int = 20, n_classes: int = 5, n_labels: int = 2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic multi-label classification dataset."""
    X, Y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        allow_unlabeled=False,
        random_state=random_state,
    )
    return X.astype(np.float32), Y.astype(np.int32)


def load_imbalanced_classification_dataset(n_samples: int = 2000, n_features: int = 20, weights: Tuple[float, float] = (0.95, 0.05), random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an imbalanced binary classification dataset to stress metrics and loss choices."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.5),
        n_redundant=int(n_features * 0.1),
        n_classes=2,
        weights=list(weights),
        flip_y=0.01,
        class_sep=1.5,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.int32)


def load_synthetic_text_classification_dataset(
    n_samples: int = 1000,
    sequence_length: int = 40,
    vocab_size: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a toy text classification dataset of integer token sequences."""
    rng = np.random.default_rng(random_state)
    sequences = rng.integers(1, vocab_size, size=(n_samples, sequence_length), dtype=np.int32)
    threshold = sequence_length * (vocab_size / 2)
    left_scores = sequences[:, : sequence_length // 2].sum(axis=1)
    right_scores = sequences[:, sequence_length // 2 :].sum(axis=1)
    labels = (left_scores > right_scores + threshold * 0.01).astype(np.int32)
    return sequences, labels


def load_sine_wave_forecasting_dataset(
    n_samples: int = 1024,
    window_size: int = 24,
    horizon: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a multi-step time-series forecasting dataset from noisy sine waves."""
    rng = np.random.default_rng(random_state)
    timesteps = np.arange(n_samples + window_size + horizon, dtype=np.float32)
    series = (
        np.sin(timesteps / 10.0)
        + 0.5 * np.sin(timesteps / 4.0)
        + 0.25 * np.sin(timesteps / 20.0)
        + rng.normal(scale=0.1, size=timesteps.shape)
    ).astype(np.float32)

    windowed_x = []
    windowed_y = []
    for start in range(n_samples):
        end = start + window_size
        target_end = end + horizon
        windowed_x.append(series[start:end])
        windowed_y.append(series[end:target_end])

    X = np.array(windowed_x, dtype=np.float32)[..., np.newaxis]
    y = np.array(windowed_y, dtype=np.float32)
    return X, y


def load_synthetic_color_image_dataset(
    n_samples: int = 600,
    image_size: int = 16,
    n_classes: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create small RGB images with simple color-based class patterns."""
    rng = np.random.default_rng(random_state)
    images = np.zeros((n_samples, image_size, image_size, 3), dtype=np.float32)
    labels = np.zeros((n_samples,), dtype=np.int32)

    for idx in range(n_samples):
        label = idx % n_classes
        base = np.zeros((image_size, image_size, 3), dtype=np.float32)
        if label == 0:
            base[:, : image_size // 2, 0] = 1.0  # red left half
        elif label == 1:
            base[image_size // 4 : 3 * image_size // 4, :, 1] = 1.0  # green band
        else:
            base[:, :, 2] = np.tile(np.linspace(0.0, 1.0, image_size), (image_size, 1))  # blue gradient

        noise = rng.normal(scale=0.05, size=base.shape).astype(np.float32)
        images[idx] = np.clip(base + noise, 0.0, 1.0)
        labels[idx] = label

    shuffle_idx = rng.permutation(n_samples)
    return images[shuffle_idx], labels[shuffle_idx]
