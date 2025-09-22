# Imports
import os
import json
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _load_recent_training_runs(log_path: Path, limit: int = 5) -> List[Dict[str, Any]]:
    """Read the most recent training run records from a JSONL log file."""
    if limit <= 0:
        return []
    if not log_path.exists():
        return []
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            recent_lines = deque(handle, maxlen=limit)
    except OSError:
        return []

    runs: List[Dict[str, Any]] = []
    for raw_line in recent_lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            runs.append(json.loads(raw_line))
        except json.JSONDecodeError:
            continue
    return runs


def _summarize_runs_for_prompt(runs: List[Dict[str, Any]]) -> str:
    """Create a concise summary of prior runs ordered by validation accuracy."""
    if not runs:
        return ""

    def _as_float(value: Any) -> Any:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    sorted_runs = sorted(
        runs,
        key=lambda rec: _as_float(rec.get("final_metrics", {}).get("val_acc")) or float("-inf"),
        reverse=True,
    )

    summary_lines: List[str] = []
    for idx, run in enumerate(sorted_runs, start=1):
        final_metrics = run.get("final_metrics", {})
        val_acc = _as_float(final_metrics.get("val_acc"))
        val_loss = _as_float(final_metrics.get("val_loss"))
        train_loss = _as_float(final_metrics.get("train_loss"))

        def _format_metric(metric: Any, pct: bool = False) -> str:
            if metric is None:
                return "n/a"
            return f"{metric:.2%}" if pct else f"{metric:.4f}"

        layer_code = run.get("generated_layers", "").strip()
        single_line_layers = " ".join(layer_code.split())
        if len(single_line_layers) > 200:
            single_line_layers = f"{single_line_layers[:197]}..."

        summary_lines.append(
            (
                f"Run {idx}: val_acc={_format_metric(val_acc, pct=True)}, "
                f"val_loss={_format_metric(val_loss)}, train_loss={_format_metric(train_loss)}, "
                f"layers={single_line_layers}"
            )
        )

    return "\n".join(summary_lines)


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

    log_path = Path(getattr(cfg, "results_log_path", "training_runs.jsonl")).expanduser()
    previous_runs = _load_recent_training_runs(log_path, limit=5)
    prior_summary = _summarize_runs_for_prompt(previous_runs)
    if prior_summary:
        history_context = (
            "Recent training history (best first):\n"
            f"{prior_summary}\n\n"
            "Blend the strengths of higher-performing runs and address weaknesses where accuracy and loss stalled."
        )
    else:
        history_context = (
            "No previous training runs were found. Choose a reasonable architecture for the task."
        )

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
                f"{history_context}\n"
                "Using only these building blocks, craft a PyTorch `nn.Sequential` suitable"
                f" for a classifier that accepts inputs of dimension {cfg.input_dim} and outputs"
                f" {cfg.num_classes} logits. Explore different depths and hidden widths when"
                " prior performance plateaus, but ensure the final module produces the correct"
                " number of class scores. Only respond with the Python assignment"
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

        # Keep a copy so training summaries can reference the generated architecture
        self.generated_layers_code = generated_layers

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
    def __init__(
        self,
        cfg: Config,
        xs: torch.Tensor,
        ys: torch.Tensor,
        results_log_path: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = SimpleClassifier(cfg).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.training_history: List[Dict[str, Any]] = []
        self.training_runs: List[Dict[str, Any]] = []
        self.latest_results: Dict[str, Any] = {}
        resolved_log_path = results_log_path or getattr(cfg, "results_log_path", "training_runs.jsonl")
        self.results_log_path = Path(resolved_log_path).expanduser()

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
        history: List[Dict[str, Any]] = []
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train()
            val_loss, val_acc = self.evaluate()
            print(
                f"Epoch {epoch}/{self.cfg.epochs} | Train loss: {train_loss:.3f} "
                f"| Val loss: {val_loss:.3f} | Val accuracy: {val_acc:.2%}"
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                }
            )

        history_copy = [dict(record) for record in history]
        final_metrics = dict(history_copy[-1]) if history_copy else {}
        run_record = {
            "generated_layers": getattr(self.model, "generated_layers_code", ""),
            "history": history_copy,
            "final_metrics": final_metrics,
        }
        self.training_history = history_copy
        self.latest_results = run_record
        self.training_runs.append(run_record)
        self._append_run_to_log(run_record)

    def predict(self, sample: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities for a sample batch."""
        with torch.no_grad():
            logits = self.model(sample.to(self.device))
            probs = torch.softmax(logits, dim=1)
            return probs.cpu()

    def _append_run_to_log(self, run_record: Dict[str, Any]) -> None:
        try:
            log_path = self.results_log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(run_record))
                log_file.write("\n")
        except OSError as exc:
            print(f"Warning: unable to write training log to {log_path}: {exc}")
