# this project uses the power of chatgpt to generate keras models for any dataset/machine learning problem

# flowchart TD
#     A[Define problem & constraints\n• Task, metric, budget, latency\n• Success criteria] --> B[Data audit & split\n• Train/Val/Test (stratified or time-based)\n• Leakage checks]
#     B --> C[Preprocess & features\n• Normalisation/Tokenisation\n• Augmentation (if images/audio)\n• Imbalance handling]
#     C --> D[Baseline model (Keras)\n• Small, proven arch\n• Sensible defaults]
#     D --> E[Compile\n• Loss & metric aligned to task\n• Optimiser: AdamW/SGD\n• Learning-rate schedule]
#     E --> F[Train (v1)\n• Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint\n• Mixed precision if supported]
#     F --> G[Evaluate on validation\n• Curves: loss/metric vs epoch\n• Confusion/PR curves as needed]
#     G --> H{Diagnosis}
#     H -->|Underfitting| I[Increase capacity\n• Wider/deeper\n• Train longer\n• Improve features]
#     H -->|Overfitting| J[Regularise more\n• Augment, Dropout, Weight decay\n• Reduce width/depth]
#     H -->|Optimisation issues| K[Tune training\n• LR range test\n• Batch size, schedule\n• Gradient clipping]
#     I --> L[Iterate & retrain]
#     J --> L
#     K --> L
#     L --> G
#     G --> M[Hyperparameter search (small)\n• Random/Bayesian over LR, WD, dropout, layers]
#     M --> N[Select best checkpoint\n• Re-train on Train+Val if appropriate\n• Final metrics on Test]
#     N --> O[Documentation & tracking\n• Seed, config, code hash\n• Model card, data version]
#     O --> P[Export & deploy\n• SavedModel/TF Serving/TFLite/ONNX\n• Latency & resource checks]
#     P --> Q[Monitor in production\n• Drift, performance, errors\n• Feedback loop to data/labels]
#     Q --> B



from __future__ import annotations

import difflib
import json
import os
import random
import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from openai import OpenAI

def chat(message):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


def A_define_the_problem_and_constraints(train_x: np.ndarray, train_y: np.ndarray) -> str:
    prompt = f"""
You are an expert machine learning engineer. A user has provided you with a dataset to build a machine learning model.
The training data has {train_x.shape[0]} samples and {train_x.shape[1]} features.
The target variable has shape {train_y.shape}.
Define the machine learning problem and constraints based on the data provided.
this should include:
- The type of task
- The evaluation metric to be used
- Any constraints such as budget, latency, or resource limitations
Provide your answer in a concise manner.

machine power {os.cpu_count()} cpu cores, {tf.config.experimental.get_memory_info('GPU:0')['current'] / (1024 ** 3):.2f} GB GPU memory
"""
    return chat(prompt)

def B_preprocess_and_feature_engineering(train_x: np.ndarray, train_y: np.ndarray, problem_definition: str):
    prompt = f"""
You are an expert machine learning engineer. A user has provided you with a dataset to build a machine learning model.
The training data has {train_x.shape[0]} samples and {train_x.shape[1]} features.
The target variable has shape {train_y.shape}.
Please provide a python function that takes in the training data and performs preprocessing and feature engineering.
The function should return the preprocessed training data.
The function should be named `preprocess_data` and have the following signature:
def preprocess_data(train_x: np.ndarray, train_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
The function should include:
- Normalization or standardization of numerical features
- Encoding of categorical features
- Handling of missing values
- Any feature engineering steps you deem necessary
The function should not include any model training or evaluation code.
The function should be compatible with TensorFlow and Keras.
Here is the problem definition:
{problem_definition}
Provide your answer in a python code block and nothing else.
"""
    return chat(prompt)

def C_build_a_baseline_model(train_x: np.ndarray, train_y: np.ndarray, problem_definition: str):
    prompt = f"""
You are an expert machine learning engineer. A user has provided you with a dataset to build a machine learning model.
The training data has {train_x.shape[0]} samples and {train_x.shape[1]} features.
The target variable has shape {train_y.shape}.
Please provide a python function that builds a baseline Keras model for the given problem.
The function should be named `build_model` and have the following signature:
def build_model(input_shape: Tuple[int, ...]) -> keras.Model:
The function should include:
- A simple and proven architecture suitable for the problem
- Sensible default values for layers and activations
- Compilation of the model with appropriate loss function, optimizer, and metrics
The function should not include any data preprocessing or training code.
The function should be compatible with TensorFlow and Keras.
Here is the problem definition:
{problem_definition}
Provide your answer in a python code block and nothing else.
"""   
    return chat(prompt)



def main(train_x: np.ndarray, train_y: np.ndarray):




if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")

    train_x, train_y, test_x, test_ids = _prepare_house_price_arrays()

    # normalize all the data
    train_x_mean = train_x.mean(axis=0, keepdims=True)
    train_x_std = train_x.std(axis=0, keepdims=True) + 1
    train_x = (train_x - train_x_mean) / train_x_std
    test_x = (test_x - train_x_mean) / train_x_std
    train_y = np.log1p(train_y)

    main(train_x, train_y)


