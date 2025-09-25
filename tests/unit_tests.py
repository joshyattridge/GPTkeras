import os
import sys

sys.path.append(os.path.abspath("../"))
from kerasgpt import kerasgpt

if __name__ == "__main__":

    from data import (
        load_binary_classification_dataset,
        load_digits_dataset,
        load_imbalanced_classification_dataset,
        load_multiclass_tabular_dataset,
        load_multilabel_classification_dataset,
        load_multioutput_regression_dataset,
        load_sine_wave_forecasting_dataset,
        load_synthetic_color_image_dataset,
        load_synthetic_text_classification_dataset,
        load_regression_dataset,
        load_tabular_classification_dataset,
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to run the GPTmodel demonstrations.")

    dataset_configs = [
        ("digits_image_classification", load_digits_dataset, {"requested_samples": 1000}, 1),
        ("synthetic_tabular_classification", load_tabular_classification_dataset, {}, 1),
        ("synthetic_regression", load_regression_dataset, {}, 1),
        ("breast_cancer_binary_classification", load_binary_classification_dataset, {}, 1),
        ("wine_multiclass_classification", load_multiclass_tabular_dataset, {}, 1),
        ("linnerud_multioutput_regression", load_multioutput_regression_dataset, {}, 1),
        ("synthetic_multilabel_classification", load_multilabel_classification_dataset, {}, 1),
        ("synthetic_imbalanced_classification", load_imbalanced_classification_dataset, {}, 1),
        ("synthetic_text_classification", load_synthetic_text_classification_dataset, {}, 1),
        ("sine_wave_forecasting", load_sine_wave_forecasting_dataset, {}, 1),
        ("synthetic_color_image_classification", load_synthetic_color_image_dataset, {}, 1),
    ]

    for dataset_name, loader, loader_kwargs, max_iterations in dataset_configs:
        features, targets = loader(**loader_kwargs)
        print(f"{dataset_name}: {features.shape}, {targets.shape}")
        model = kerasgpt(features, targets, api_key=api_key)
        model.fit(max_iterations=max_iterations)