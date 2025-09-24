from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data import load_digits_dataset




class GPTmodel:
    batch_size: int = 128
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 8
    num_classes: int = 10

    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.image_size = int(train_x.shape[1])
        self.num_classes = int(train_y.max()) + 1
        self.model = self.build_model()

    def build_model(self) -> keras.Model:
        regularizer = keras.regularizers.l2(self.weight_decay) if self.weight_decay > 0 else None
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
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    
    def fit(self):
        self.model.fit(self.train_x, self.train_y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)













if __name__ == "__main__":
    images, labels = load_digits_dataset()

    print(images.shape, labels.shape)

    model = GPTmodel(images, labels)
    model.fit()
