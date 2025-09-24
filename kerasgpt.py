from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

class OpenAIChatClient:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", system_prompt: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai package to use OpenAIChatClient") from exc

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.messages: list[dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, content: str, role: str = "user", **kwargs) -> str:
        if not content:
            raise ValueError("Message content must be provided")

        self.messages.append({"role": role, "content": content})
        completion = self.client.chat.completions.create(model=self.model, messages=self.messages, **kwargs)
        message = completion.choices[0].message
        response = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        self.messages.append({"role": "assistant", "content": response})
        return response

    def reset(self, system_prompt: str | None = None) -> None:
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def history(self) -> list[dict[str, str]]:
        return list(self.messages)




class GPTmodel:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.model = self.build_model()

    def build_model(self) -> keras.Model:
        inputs = keras.Input(shape=(int(train_x.shape[1]), int(train_x.shape[1]), 1))
        x = keras.layers.Flatten()(inputs)
        outputs = keras.layers.Dense(int(train_y.max()) + 1, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def fit(self):
        self.model.fit(self.train_x, self.train_y, epochs=12, batch_size=128, validation_split=0.2)








if __name__ == "__main__":

    from data import load_digits_dataset, load_regression_dataset, load_tabular_classification_dataset

    # Example usage with the digits dataset
    images, labels = load_digits_dataset()
    print(images.shape, labels.shape)
    model = GPTmodel(images, labels)
    model.fit()

    # Example usage with the tabular classification dataset
    X_class, y_class = load_tabular_classification_dataset()
    print(X_class.shape, y_class.shape)
    model_class = GPTmodel(X_class, y_class)
    model_class.fit()

    # Example usage with the regression dataset
    X_reg, y_reg = load_regression_dataset()
    print(X_reg.shape, y_reg.shape)
    model_reg = GPTmodel(X_reg, y_reg)
    model_reg.fit()
