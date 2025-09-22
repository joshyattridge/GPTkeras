# Imports
import os
import json
import inspect
from typing import Any, Dict, List

import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset

from data import Config

# Optional OpenAI import
try:
    import openai  # type: ignore[import-untyped]
except ImportError:
    openai = None  # type: ignore[assignment]

def list_available_layers() -> List[Dict[str, Any]]:
    """Return a list of all available torch.nn layers and their __init__ arguments."""
    layer_list: List[Dict[str, Any]] = []
    for name in dir(nn):
        obj = getattr(nn, name)
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
            try:
                sig = inspect.signature(obj.__init__)
                args = [p for p in sig.parameters if p != "self"]
                layer_list.append({"layer": name, "args": args})
            except Exception:
                layer_list.append({"layer": name, "args": "N/A (signature not available)"})
    return layer_list

def _strip_code_fences(text: str) -> str:
    """Remove simple Markdown code fences from a model response."""
    result = text.strip()
    if result.startswith("```"):
        lines = list(result.splitlines())
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        result = "\n".join(lines).strip()
    return result

def _format_for_responses(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert chat-completions style messages into Responses API payload."""
    return [
        {
            "role": m["role"],
            "content": [{"type": "text", "text": m["content"]}],
        }
        for m in messages
    ]

def _call_openai_chat(messages: List[Dict[str, str]], model_name: str, api_key: str) -> str:
    """Send a chat request to OpenAI, supporting both legacy and responses APIs."""
    if openai is None:
        raise ImportError("The `openai` package is required to programmatically generate layers.")

    # OpenAI v1 API
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key)  # type: ignore[attr-defined]
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
            )
            return completion.choices[0].message.content or ""
        response = client.responses.create(
            model=model_name,
            input=_format_for_responses(messages),
            temperature=0,
        )
        return getattr(response, "output_text", "").strip()

    # OpenAI legacy API
    if hasattr(openai, "ChatCompletion"):
        openai.api_key = api_key  # type: ignore[attr-defined]
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0,
        )
        return completion["choices"][0]["message"]["content"] or ""

    raise RuntimeError("Unsupported OpenAI client version detected.")

def generate_layers(cfg: Config, model_name: str = "gpt-4o-mini") -> str:
    """
    Ask ChatGPT to propose an assignment for `self.layers` based on available layers.
    Returns a string assignment for self.layers.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set; unable to contact ChatGPT.")

    layer_catalog = json.dumps(available_layers, indent=2)
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You write concise Python for PyTorch modules. Respond with a single "
                "assignment that sets `self.layers` for a neural network."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here is the list of available torch.nn layers and their constructor"
                f" details:\n{layer_catalog}\n\n"
                "Using only these building blocks, craft a PyTorch `nn.Sequential` suitable"
                f" for a classifier with input_dim={cfg.input_dim}, hidden_dim={cfg.hidden_dim},"
                f" and num_classes={cfg.num_classes}. Only respond with the Python assignment"
                " statement `self.layers = ...` and nothing else. Do not include code fences."
            ),
        },
    ]
    response = _call_openai_chat(messages, model_name, api_key)
    return _strip_code_fences(response)

# Cache available layers for use in generate_layers
available_layers = list_available_layers()

# ----------------------
# Model Definitions
# ----------------------

class SimpleClassifier(nn.Module):
    """
    Dynamically generated classifier using OpenAI API.
    The layers are assigned by executing code returned from generate_layers().
    """
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        try:
            generated_layers = generate_layers(cfg)
        except Exception as exc:
            raise RuntimeError(f"generate_layers failed: {exc}")

        print("Generated layers assignment:")
        print(generated_layers)

        if generated_layers and "self.layers" in generated_layers:
            try:
                exec(
                    generated_layers,
                    {"nn": nn, "torch": torch, "cfg": cfg},
                    {"self": self},
                )
            except Exception as exc:
                raise RuntimeError(f"Unable to apply generated layers: {exc}")
        else:
            raise RuntimeError("No valid self.layers assignment generated.")

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

# ----------------------
# Training Utilities
# ----------------------

class GPTTrainer:
    """
    Utility class for training and evaluating a SimpleClassifier.
    """
    def __init__(self, cfg: Config, xs: torch.Tensor, ys: torch.Tensor):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = SimpleClassifier(cfg).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        train_ds, val_ds = self.split_dataset(xs, ys, cfg.train_split)
        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size)

    @staticmethod
    def split_dataset(xs: torch.Tensor, ys: torch.Tensor, train_ratio: float):
        """Split dataset into train and validation sets."""
        num_train = int(len(xs) * train_ratio)
        indices = torch.randperm(len(xs))
        train_idx = indices[:num_train]
        val_idx = indices[num_train:]
        return (
            TensorDataset(xs[train_idx], ys[train_idx]),
            TensorDataset(xs[val_idx], ys[val_idx]),
        )

    def train(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        for xb, yb in self.train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * xb.size(0)
        return total_loss / len(self.train_loader.dataset)

    def evaluate(self) -> (float, float):
        """Evaluate on validation set. Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = correct / len(self.val_loader.dataset)
        return avg_loss, accuracy

    def fit(self):
        """Train for cfg.epochs epochs, printing progress."""
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train()
            val_loss, val_acc = self.evaluate()
            print(
                f"Epoch {epoch}/{self.cfg.epochs} | Train loss: {train_loss:.3f} "
                f"| Val loss: {val_loss:.3f} | Val accuracy: {val_acc:.2%}"
            )

    def predict(self, sample: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities for a sample batch."""
        with torch.no_grad():
            logits = self.model(sample.to(self.device))
            probs = torch.softmax(logits, dim=1)
            return probs.cpu()