"""Example: train GPTkeras on the Kaggle House Prices regression challenge."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow import keras

# Allow running the example directly from the repository without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GPTkeras import GPTkeras

_KAGGLE_COMPETITION = "house-prices-advanced-regression-techniques"
_SUBMISSION_FILENAME = "house_prices_submission.csv"


def _resolve_dataset_dir() -> Path:
    """Locate the directory containing the Kaggle dataset CSV files."""
    env_dir = os.getenv("HOUSE_PRICES_DATA_DIR")
    candidate_dirs = []
    if env_dir:
        candidate_dirs.append(Path(env_dir).expanduser())

    candidate_dirs.extend(
        [
            REPO_ROOT / _KAGGLE_COMPETITION,
            REPO_ROOT / "examples" / _KAGGLE_COMPETITION,
            REPO_ROOT / "data" / "house_prices",
        ]
    )

    # Remove duplicates while preserving order.
    candidate_dirs = list(dict.fromkeys(candidate_dirs))

    for directory in candidate_dirs:
        if (directory / "train.csv").exists():
            return directory

    search_list = "\n".join(f"  - {path}" for path in candidate_dirs)
    raise FileNotFoundError(
        "Could not find Kaggle House Prices `train.csv`.\n"
        "Checked the following locations:\n"
        f"{search_list}\n"
        "Download the competition files with `kaggle competitions download -c "
        f"{_KAGGLE_COMPETITION}` and either extract them into one of the "
        "directories above or set HOUSE_PRICES_DATA_DIR to the extracted folder."
    )


def _prepare_house_price_arrays(
    limit_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Kaggle house price data and return train/test arrays with aligned columns."""

    data_dir = _resolve_dataset_dir()
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"

    train_frame = pd.read_csv(train_csv)
    test_frame = pd.read_csv(test_csv)

    if "SalePrice" not in train_frame.columns:
        raise ValueError("Expected `SalePrice` column to be present in Kaggle training data.")

    train_target = train_frame["SalePrice"].astype("float32")

    feature_columns = [col for col in train_frame.columns if col not in {"SalePrice"}]
    train_features = train_frame[feature_columns].copy()
    test_features = test_frame[feature_columns].copy()

    # Keep the Id column for the submission file but remove it from the feature set.
    train_ids = train_features.pop("Id") if "Id" in train_features.columns else train_frame["Id"]
    test_ids = test_features.pop("Id") if "Id" in test_features.columns else test_frame["Id"]

    combined_features = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    numeric_cols = combined_features.select_dtypes(include=["number"]).columns
    categorical_cols = combined_features.columns.difference(numeric_cols)

    numeric_frame = combined_features[numeric_cols].copy()
    numeric_frame = numeric_frame.fillna(numeric_frame.median()).astype("float32")

    categorical_frame = combined_features[categorical_cols].copy()
    categorical_frame = categorical_frame.fillna("missing").astype(str)

    combined = pd.concat([numeric_frame, categorical_frame], axis=1)
    encoded = pd.get_dummies(combined, drop_first=False)

    train_encoded = encoded.iloc[: len(train_features)]
    test_encoded = encoded.iloc[len(train_features) :]

    if limit_rows is not None and len(train_encoded) > limit_rows:
        train_encoded = train_encoded.iloc[:limit_rows]
        train_target = train_target.iloc[:limit_rows]
        train_ids = train_ids.iloc[:limit_rows]

    train_x = train_encoded.to_numpy(dtype="float32", copy=True)
    train_y = train_target.to_numpy(dtype="float32", copy=True)
    test_x = test_encoded.to_numpy(dtype="float32", copy=True)
    test_id_array = test_ids.to_numpy(copy=True)

    return train_x, train_y, test_x, test_id_array


def _load_house_price_data(limit_rows: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning only the training arrays."""

    train_x, train_y, _, _ = _prepare_house_price_arrays(limit_rows=limit_rows)
    return train_x, train_y


def _write_submission(predictions: np.ndarray, test_ids: np.ndarray) -> Path:
    """Create a Kaggle-formatted submission CSV and return its path."""

    submission_frame = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
    submission_path = REPO_ROOT / _SUBMISSION_FILENAME
    submission_frame.to_csv(submission_path, index=False)
    return submission_path


def main() -> None:
    # run the command below to set your OpenAI API key
    # export OPENAI_API_KEY="your_api_key_here"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")

    train_x, train_y, test_x, test_ids = _prepare_house_price_arrays()

    model = GPTkeras(
        train_x,
        train_y,
        api_key=api_key,
    )

    # Limit iterations to keep the demo lightweight.
    model.fit(max_iterations=100)

    if not model.best_model_path.exists():
        raise RuntimeError(
            "best_model.keras was not created. Check the training logs for errors before generating a submission."
        )

    print(f"Loading best model from {model.best_model_path}...")
    best_model = keras.models.load_model(model.best_model_path)

    print("Generating predictions for the Kaggle test set...")
    predictions = best_model.predict(test_x, batch_size=256, verbose=0).reshape(-1)
    predictions = np.clip(predictions, a_min=0.0, a_max=None)

    submission_path = _write_submission(predictions, test_ids)
    print(f"Submission file written to: {submission_path}")


if __name__ == "__main__":
    main()


# Model 9: loss=923071744.0000, mae=22241.4102, val_loss=5116783616.0000, val_mae=34013.1211, learning_rate=0.0001105505