from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras






class GPTmodel:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.image_size = int(train_x.shape[1])
        self.num_classes = int(train_y.max()) + 1
        self.model = self.build_model()

    def build_model(self) -> keras.Model:
        regularizer = keras.regularizers.l2(1e-4) if 1e-4 > 0 else None
        inputs = keras.Input(shape=(self.image_size, self.image_size, 1))
        x = keras.layers.Conv2D(16, kernel_size=5, padding="same", activation="relu", kernel_regularizer=regularizer)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", kernel_regularizer=regularizer)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", kernel_regularizer=regularizer)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(self.num_classes, activation="softmax", kernel_regularizer=regularizer)(x)

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
