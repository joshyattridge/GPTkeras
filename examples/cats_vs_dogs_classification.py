"""Example: train KerasCrafterGPT on the classic Cats vs Dogs dataset."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

# Allow running the example directly from the repository without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from KerasCrafterGPT import KerasCrafterGPT

_DATASET_ORIGIN = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"


def _download_dataset() -> Path:
    """Download and locate the extracted Cats vs Dogs dataset."""
    zip_path = Path(
        tf.keras.utils.get_file(
            "cats_and_dogs_filtered.zip",
            origin=_DATASET_ORIGIN,
            extract=True,
        )
    )
    print(f"Downloaded zip path: {zip_path}")

    # Always use the correct extraction directory
    extracted_dir = zip_path.parent / "cats_and_dogs_filtered_extracted"
    nested_dataset_dir = extracted_dir / "cats_and_dogs_filtered"
    train_dir = nested_dataset_dir / "train"
    if train_dir.exists():
        print(f"Found train dir at: {train_dir}")
        return nested_dataset_dir
    else:
        print(f"Could not find train dir at: {train_dir}")
        print(f"Contents of {extracted_dir}:")
        if extracted_dir.exists():
            for item in extracted_dir.iterdir():
                print(f"  {item}")
        raise FileNotFoundError(
            "Unable to locate extracted Cats vs Dogs dataset after download."
        )


def _load_numpy_data(
    image_size: tuple[int, int] = (128, 128),
    limit_per_class: int = 200,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create balanced numpy arrays from the dataset for KerasCrafterGPT."""
    data_dir = _download_dataset()
    train_dir = data_dir / "train"

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    images: list[np.ndarray] = []
    labels: list[int] = []
    class_counts = {0: 0, 1: 0}

    for batch_images, batch_labels in dataset:
        batch_images = batch_images.numpy().astype("float32") / 255.0
        batch_labels = batch_labels.numpy().astype("int32")

        for idx in range(len(batch_labels)):
            label = int(batch_labels[idx])
            if class_counts[label] >= limit_per_class:
                continue

            images.append(batch_images[idx])
            labels.append(label)
            class_counts[label] += 1

        if all(count >= limit_per_class for count in class_counts.values()):
            break

    if not images or not labels:
        raise RuntimeError("Failed to collect any samples from Cats vs Dogs dataset.")

    train_x = np.stack(images, axis=0)
    train_y = np.array(labels, dtype="int32")
    return train_x, train_y


def main() -> None:

    # run the command below to set your OpenAI API key
    # export OPENAI_API_KEY="your_api_key_here"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")

    train_x, train_y = _load_numpy_data()

    model = KerasCrafterGPT(
        train_x,
        train_y,
        api_key=api_key,
    )

    # Limit iterations to keep the demo lightweight.
    model.fit(max_iterations=10)


if __name__ == "__main__":
    main()
