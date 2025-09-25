from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from openai import OpenAI

class OpenAIChatClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        system_prompt: str | None = None,
        history_path: str | os.PathLike[str] | None = "chat_history.jsonl",
        continue_from_history: bool = False,
    ):

        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.messages: list[dict[str, str]] = []
        self.history_path = self._prepare_history_path(history_path)
        self._initialize_history(continue_from_history)
        if system_prompt:
            self._append_message(role="system", content=system_prompt)

    def chat(self, content: str, role: str = "user", **kwargs) -> str:
        if not content:
            raise ValueError("Message content must be provided")

        self._append_message(role=role, content=content)
        completion = self.client.chat.completions.create(model=self.model, messages=self.messages, **kwargs)
        message = completion.choices[0].message
        response = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        if not isinstance(response, str):
            response = "" if response is None else str(response)
        self._append_message(role="assistant", content=response)
        return response

    def reset(self, system_prompt: str | None = None) -> None:
        self.messages = []
        self._record_history({"role": "event", "content": "conversation_reset"})
        if system_prompt:
            self._append_message(role="system", content=system_prompt)

    def history(self) -> list[dict[str, str]]:
        return list(self.messages)

    def _prepare_history_path(self, history_path: str | os.PathLike[str] | None) -> Path | None:
        if history_path is None:
            return None

        path = Path(history_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _append_message(self, role: str, content: str) -> None:
        if not isinstance(content, str):
            content = str(content)
        message = {"role": role, "content": content}
        self.messages.append(message)
        self._record_history(message)

    def _record_history(self, message: dict[str, str]) -> None:
        if not self.history_path:
            return

        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            **message,
        }
        with self.history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(record) + "\n")

    def _initialize_history(self, continue_from_history: bool) -> None:
        if not self.history_path:
            return

        if continue_from_history:
            if not self.history_path.exists():
                return

            try:
                with self.history_path.open("r", encoding="utf-8") as history_file:
                    for line in history_file:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        role = record.get("role")
                        content = record.get("content")
                        if role not in {"system", "user", "assistant"} or content is None:
                            continue
                        if not isinstance(content, str):
                            content = str(content)
                        self.messages.append({"role": role, "content": content})
            except OSError:
                pass
        else:
            try:
                self.history_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass




class GPTmodel:
    def __init__(
        self,
        train_x,
        train_y,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        history_path: str | os.PathLike[str] | None = "chat_history.jsonl",
        continue_from_history: bool = False,
    ):
        self.train_x = train_x
        self.train_y = train_y
        self.input_shape = tuple(train_x.shape[1:]) if train_x.ndim > 1 else (1,)
        self.num_classes = int(train_y.max()) + 1 if np.issubdtype(train_y.dtype, np.integer) else 1
        self.gpt_client = OpenAIChatClient(
            api_key=api_key,
            model=model,
            history_path=history_path,
            continue_from_history=continue_from_history,
        )
        self.training_config: dict[str, int] = {}
        self.best_model_path = Path("best_model.keras")
        self.best_val_loss = float("inf")
        self.callbacks_factory: Callable[[], list[keras.callbacks.Callback]] | None = None

    def build_model(self) -> keras.Model:
        if self.gpt_client is None:
            raise ValueError("OpenAIChatClient instance is required to build the model")

        self.callbacks_factory = None
        sample_size = min(5, len(self.train_x))
        sample_inputs = self.train_x[: min(sample_size, len(self.train_x))]
        sample_outputs = self.train_y[: min(sample_size, len(self.train_y))]
        prompt = self._build_prompt(sample_inputs=sample_inputs, sample_outputs=sample_outputs)

        response = self.gpt_client.chat(prompt)

        print("Generated model code:\n", response)

        exec_globals = {
            "keras": keras,
            "tf": tf,
            "np": np,
            "Path": Path,
            "BEST_MODEL_PATH": str(self.best_model_path),
        }
        exec_locals: dict[str, object] = {}
        try:
            exec(response, exec_globals, exec_locals)
        except Exception as exc:
            raise RuntimeError("Failed to execute model code generated by GPT") from exc

        batch_size = exec_locals.get("BATCH_SIZE")
        epochs = exec_locals.get("EPOCHS")

        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("GPT response must define a positive integer BATCH_SIZE constant.")
        if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
            raise ValueError("GPT response must define a positive integer EPOCHS constant.")

        self.training_config["batch_size"] = int(batch_size)
        self.training_config["epochs"] = int(epochs)

        create_model = exec_locals.get("create_model")
        if not callable(create_model):
            raise ValueError("GPT response did not define a callable create_model function")

        create_callbacks = exec_locals.get("create_callbacks")
        if create_callbacks is not None and not callable(create_callbacks):
            raise ValueError("create_callbacks must be callable when defined")
        if callable(create_callbacks):
            self.callbacks_factory = create_callbacks

        model = create_model()
        if not isinstance(model, keras.Model):
            raise TypeError("create_model must return a compiled keras.Model instance")

        return model

    def _build_prompt(self, sample_inputs: np.ndarray, sample_outputs: np.ndarray) -> str:
        input_shape = self.input_shape
        output_shape = tuple(sample_outputs.shape[1:]) if sample_outputs.ndim > 1 else ()
        task_type = "classification" if self.num_classes > 1 else "regression"

        prompt = f"""
You are an expert TensorFlow engineer. Generate Python source code for a function called create_model() that builds and compiles a tf.keras.Model for a {task_type} problem.
If you know of any previous models that you created and the results they produced then use this information to influence your design.

Project constraints:
- Training input shape: {input_shape}
- Training output shape: {output_shape if output_shape else 'scalar'}
- Number of target classes: {self.num_classes}
- Sample input batch (first {len(sample_inputs)} rows): {sample_inputs.tolist()}
- Sample target values: {sample_outputs.tolist()}
- The constant BEST_MODEL_PATH is available for saving checkpoints.

Requirements:
1. The function must be pure Python using tf.keras layers and return a compiled keras.Model instance.
2. Return valid Python that defines create_model() and sets integer constants BATCH_SIZE and EPOCHS at module scope (outside create_model); do not include explanations or markdown fences.
3. Ensure BATCH_SIZE and EPOCHS are positive integers tailored to the task and data size.
4. Design the network so every convolution, pooling, or downsampling step keeps all spatial dimensions at least 1 for the provided input shape; adjust kernel sizes, strides, or padding (e.g., prefer padding="same" when needed) to avoid invalid tensor shapes.
5. Optionally define a create_callbacks() function with no parameters that returns a list (or tuple) of tf.keras.callbacks.Callback instances to use during training; return an empty list if no callbacks are needed.
6. If using callbacks that write checkpoints, target BEST_MODEL_PATH and avoid writing to other locations.
"""
        return prompt.strip()

    def _build_callbacks(self) -> list[keras.callbacks.Callback]:
        if self.callbacks_factory is None:
            return []

        try:
            callbacks = self.callbacks_factory()
        except Exception as exc:
            raise RuntimeError("create_callbacks() raised an exception") from exc

        if callbacks is None:
            return []
        if not isinstance(callbacks, (list, tuple)):
            raise TypeError("create_callbacks must return a list or tuple of keras.callbacks.Callback instances")

        validated_callbacks: list[keras.callbacks.Callback] = []
        for callback in callbacks:
            if not isinstance(callback, keras.callbacks.Callback):
                raise TypeError("create_callbacks must return keras.callbacks.Callback instances")
            validated_callbacks.append(callback)

        return validated_callbacks

    def _can_results_be_improved_prompt(self, history: dict) -> str:
        prompt = f"""
        This is the results of training the model you generated:
        {history}
        Do you think these results can be improved further with a different architecture or training configuration? Answer with a short yes or no and nothing else.
        """
        return prompt.strip()

    def fit(self, max_iterations: int = 1) -> None:

        for iteration in range(max_iterations):
            self.model = self.build_model()
            if "epochs" not in self.training_config or "batch_size" not in self.training_config:
                raise ValueError("Training configuration missing; GPT must provide BATCH_SIZE and EPOCHS.")

            epochs = int(self.training_config["epochs"])
            batch_size = int(self.training_config["batch_size"])
            callbacks = self._build_callbacks()
            results = self.model.fit(
                self.train_x,
                self.train_y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
            )

            val_losses = results.history.get("val_loss")
            candidate_loss = min(val_losses) if val_losses else None
            if candidate_loss is None:
                train_losses = results.history.get("loss") or []
                candidate_loss = min(train_losses) if train_losses else None

            if candidate_loss is not None and candidate_loss < self.best_val_loss:
                self.best_val_loss = candidate_loss
                self.model.save(self.best_model_path)
                print(f"Saved new best model to {self.best_model_path}")

            can_results_be_improved = self.gpt_client.chat(self._can_results_be_improved_prompt(results.history)).strip().lower()

            if "yes" not in can_results_be_improved.lower() and "no" not in can_results_be_improved.lower():
                raise ValueError("GPT response to improvement prompt must be 'yes' or 'no'")

            if "no" in can_results_be_improved.lower():
                print("GPT determined that the model cannot be improved further.")
                break








if __name__ == "__main__":

    from data import load_digits_dataset, load_regression_dataset, load_tabular_classification_dataset
    api_key = os.getenv("OPENAI_API_KEY")


    # Example usage with the digits dataset
    images, labels = load_digits_dataset()
    print(images.shape, labels.shape)
    model = GPTmodel(images, labels, api_key=api_key)
    model.fit(max_iterations=3)

    # # Example usage with the tabular classification dataset
    # X_class, y_class = load_tabular_classification_dataset()
    # print(X_class.shape, y_class.shape)
    # model_class = GPTmodel(X_class, y_class)
    # model_class.fit()

    # # Example usage with the regression dataset
    # X_reg, y_reg = load_regression_dataset()
    # print(X_reg.shape, y_reg.shape)
    # model_reg = GPTmodel(X_reg, y_reg)
    # model_reg.fit()
