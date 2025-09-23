# Imports
import os
import json
import inspect
import re
import hashlib
from numbers import Integral
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset

try:
    import openai  # type: ignore[import-untyped]
except ImportError:
    openai = None  # type: ignore[assignment]

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


def _normalize_assignment(text: str) -> str:
    """Collapse whitespace so layer assignments can be compared reliably."""
    return " ".join(text.split())


def _split_generated_assignments(
    assignments: str,
) -> Tuple[
    str,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    """Return each required `self.*` assignment block extracted from the response."""

    blocks: Dict[str, List[str]] = {
        "layers": [],
        "optimizer": [],
        "training_epochs": [],
        "scheduler": [],
        "loss": [],
        "stop_training": [],
        "early_stop_patience": [],
        "early_stop_min_delta": [],
        "run_signature": [],
        "run_signature_hash": [],
    }

    current_key: Optional[str] = None
    for raw_line in assignments.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            if current_key:
                blocks[current_key].append(raw_line)
            continue

        if stripped.startswith("#") and current_key:
            blocks[current_key].append(raw_line)
            continue

        match = re.match(r"self\.(\w+)\s*=", stripped)
        if match:
            key = match.group(1)
            current_key = key if key in blocks else None
            if current_key is not None:
                blocks[current_key] = [raw_line]
            continue

        if current_key is not None:
            blocks[current_key].append(raw_line)
        elif blocks["layers"]:
            blocks["layers"].append(raw_line)
        else:
            # Treat orphaned lines as part of the layers assignment by default
            blocks["layers"].append(raw_line)

    def _collect(key: str) -> Optional[str]:
        lines = blocks.get(key, [])
        if not lines:
            return None
        return "\n".join(lines)

    layers_code = _collect("layers") or ""
    optimizer_code = _collect("optimizer")
    epochs_code = _collect("training_epochs")
    scheduler_code = _collect("scheduler")
    loss_code = _collect("loss")
    stop_code = _collect("stop_training")
    patience_code = _collect("early_stop_patience")
    min_delta_code = _collect("early_stop_min_delta")
    signature_code = _collect("run_signature")
    signature_hash_code = _collect("run_signature_hash")

    return (
        layers_code,
        optimizer_code,
        epochs_code,
        scheduler_code,
        loss_code,
        stop_code,
        patience_code,
        min_delta_code,
        signature_code,
        signature_hash_code,
    )


def _repair_inline_comments(block: str) -> str:
    """Relocate inline comments so they do not consume subsequent code tokens."""
    lines = block.splitlines()
    fixed_lines: List[str] = []

    for line in lines:
        if "#" not in line:
            fixed_lines.append(line)
            continue

        before, comment_tail = line.split("#", 1)
        prefix = before.rstrip()
        indent_match = re.match(r"(\s*)", before)
        indent = indent_match.group(1) if indent_match else ""

        comment_tail = comment_tail.rstrip()
        remainder = ""
        for marker in ("nn.", "torch.", "self.", "cfg."):
            marker_index = comment_tail.find(marker)
            if marker_index != -1:
                remainder = comment_tail[marker_index:].strip()
                comment_tail = comment_tail[:marker_index].strip()
                break
        else:
            comment_tail = comment_tail.strip()

        if prefix:
            if comment_tail:
                fixed_lines.append(f"{prefix}  # {comment_tail}")
            else:
                fixed_lines.append(prefix)
        elif comment_tail:
            fixed_lines.append(f"{indent}# {comment_tail}")

        if remainder:
            rest_line = f"{indent}    {remainder}"
            fixed_lines.append(rest_line)

    return "\n".join(fixed_lines)


def _load_recent_training_runs(log_path: Path, limit: int = 50) -> List[Dict[str, Any]]:
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


def _extract_metric_history(history: List[Dict[str, Any]], metric: str) -> List[Tuple[int, float]]:
    """Return sorted (epoch, value) pairs for the requested metric."""
    series: List[Tuple[int, float]] = []
    for record in history:
        epoch = record.get("epoch")
        value = record.get(metric)
        try:
            epoch_int = int(epoch)
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        series.append((epoch_int, value_float))
    series.sort(key=lambda item: item[0])
    return series


def _sample_series_points(series: List[Tuple[int, float]], desired: int = 4) -> List[Tuple[int, float]]:
    """Select up to `desired` evenly spaced points from the series."""
    if not series:
        return []
    total = len(series)
    if total <= desired:
        return series
    indices = {0, total - 1}
    if desired > 2:
        step = (total - 1) / (desired - 1)
        for idx in range(1, desired - 1):
            indices.add(int(round(idx * step)))
    return [series[idx] for idx in sorted(indices)]


def _summarize_val_acc_history(history: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Produce a short trend summary and plateau note for validation accuracy."""
    val_series = _extract_metric_history(history, "val_acc")
    if len(val_series) < 2:
        return "", ""

    sampled_points = _sample_series_points(val_series, desired=4)
    trend_parts = [f"e{epoch}={value:.2%}" for epoch, value in sampled_points]
    trend_summary = "val_acc trend " + " -> ".join(trend_parts)

    best_epoch, best_acc = max(val_series, key=lambda item: item[1])
    final_epoch, final_acc = val_series[-1]
    plateau_delta = 0.0025
    plateau_note = ""

    if (final_epoch - best_epoch) >= 3 and abs(final_acc - best_acc) <= plateau_delta:
        plateau_note = f"plateau near epoch {best_epoch} (best {best_acc:.2%})"
    elif final_epoch == best_epoch:
        plateau_note = f"best {best_acc:.2%} at final epoch {final_epoch}"
    elif best_acc - final_acc > plateau_delta:
        plateau_note = (
            f"best {best_acc:.2%} at epoch {best_epoch}; final slipped to {final_acc:.2%}"
        )
    else:
        plateau_note = f"best {best_acc:.2%} at epoch {best_epoch}"

    return trend_summary, plateau_note


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

        best_epoch = run.get("final_metrics", {}).get("best_epoch")
        epochs_trained = run.get("final_metrics", {}).get("epochs_trained")

        epoch_info = ""
        if best_epoch is not None:
            epoch_info += f" @epoch {int(best_epoch)}"
        if epochs_trained is not None:
            epoch_info += f" (trained {int(epochs_trained)} ep)"

        base_line = (
            f"Run {idx}: best_val_acc={_format_metric(val_acc, pct=True)}, "
            f"best_val_loss={_format_metric(val_loss)}, train_loss={_format_metric(train_loss)}, "
            f"layers={single_line_layers}{epoch_info}"
        )

        history = run.get("history", [])
        trend_summary, plateau_note = _summarize_val_acc_history(history)
        if trend_summary:
            base_line += f" | {trend_summary}"
        if plateau_note:
            base_line += f" ({plateau_note})"

        summary_lines.append(base_line)

    return "\n".join(summary_lines)


def _call_openai_chat(messages: List[Dict[str, str]], model_name: str, api_key: str) -> str:
    """Send a chat request to OpenAI, supporting both legacy and responses APIs."""
    if openai is None:
        raise ImportError("The `openai` package is required to programmatically generate layers.")

    # record the messages to a new file
    open("messages_log.jsonl", "a").write(json.dumps(messages) + "\n")

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

def generate_layers(
    cfg: Any,
    model_name: str = "gpt-4o-mini",
    *,
    feedback: Optional[str] = None,
) -> str:
    """
    Ask ChatGPT to propose an assignment for `self.layers` based on available layers.
    Returns a string assignment for self.layers.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set; unable to contact ChatGPT.")

    log_path = Path(getattr(cfg, "results_log_path", "training_runs.jsonl")).expanduser()
    previous_runs = _load_recent_training_runs(log_path, limit=50)
    prior_summary = _summarize_runs_for_prompt(previous_runs)
    used_assignments: List[str] = []
    seen_assignments = set()
    for run in previous_runs:
        layers_code, *_ = _split_generated_assignments(run.get("generated_layers", ""))
        assignment = _normalize_assignment(layers_code)
        if not assignment or assignment in seen_assignments:
            continue
        seen_assignments.add(assignment)
        used_assignments.append(assignment)

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

    if used_assignments:
        assignment_lines = []
        for idx, assignment in enumerate(used_assignments, start=1):
            truncated = assignment
            if len(truncated) > 200:
                truncated = f"{truncated[:197]}..."
            assignment_lines.append(f"{idx}. {truncated}")
        prior_assignments_context = (
            "Previously used layer assignments (avoid exact repeats):\n"
            + "\n".join(assignment_lines)
            + "\n"
        )
    else:
        prior_assignments_context = ""

    feedback_context = ""
    if feedback:
        summarized = feedback.strip()
        if summarized:
            feedback_context = (
                "Previous attempt failed with the following issue:\n"
                f"{summarized}\n\n"
                "Please adjust the generated assignments to avoid this problem.\n"
            )

    base_user_content = (
        f"{history_context}\n"
        f"{prior_assignments_context}"
        f"{feedback_context}"
        f"Context: tabular inputs with {cfg.input_dim} normalised features; target optimised with "
        f"`torch.nn.CrossEntropyLoss` over {cfg.num_classes} logits. Classes may be imbalanced, so incorporate"
        " techniques that handle imbalance when helpful."
        " Log the tuple (layers_repr, optimiser, lr, schedule, batch_size, weight_decay, seed) by assigning"
        " it directly to `self.run_signature = (...)` and immediately hashing it via"
        " `self.run_signature_hash = hashlib.sha256(repr(self.run_signature).encode('utf-8')).hexdigest()`"
        " to avoid reusing near-duplicates. Use only literal strings or numeric scalars in that tuple—"
        " never place nn.Module instances, optimizers, or schedulers inside it."
        f" Generate a PyTorch `nn.Sequential` for inputs of {cfg.input_dim} features and exactly {cfg.num_classes} logits;"
        f" ensure the final layer is `nn.Linear(..., {cfg.num_classes})`."
        " Design the architecture and regularisation strategy as you see fit, but never repeat an identical layer + hyperparameter tuple."
        " When validation performance plateaus, adjust architecture or hyperparameters before the next run and"
        " include a brief inline Python comment in the `self.layers` assignment summarising the adaptation and"
        " confirming the logits output."
        " Choose context-aware settings for the optimizer, learning rate, batch size, weight decay, scheduler, and seed"
        " while avoiding prior near-duplicate configurations."
        " Always construct optimizers and schedulers against `self.model.parameters()` (or `self.parameters()` when appropriate)."
        " Finally, emit concise inline Python comments after the optimizer, epoch, early stopping,"
        " and scheduler/loss assignments summarising how these choices differ from the prior best configuration."
        " Format `self.layers` so each module appears on its own line inside `nn.Sequential`, with trailing commas where needed,"
        " and place comments after the code they describe—never immediately after an opening parenthesis."
        " Reflect explicitly on whether the existing trajectory already represents the most optimal result we can achieve."
        " Encode that verdict as a boolean assignment `self.stop_training = <True|False>` with an inline comment"
        " answering the question \"Have we reached optimal results such that further model/hyperparameter changes would not help?\""
        " Use `True` only when you are confident further exploration is unlikely to help; otherwise emit `False`."
        " Do not assume any fallback defaults will be applied—omit nothing and ensure each assignment is fully executable."
        " Respond with each required Python assignment on separate lines in the order specified by the system instruction."
        " Do not include code fences or extra narration."
    )

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You write concise Python for PyTorch modules. Respond with assignments in this exact order: "
                "`self.layers = ...`, `self.optimizer = ...`, `self.training_epochs = ...`, "
                "`self.early_stop_patience = ...`, `self.early_stop_min_delta = ...`, `self.scheduler = ...`, "
                "`self.loss = ...`, `self.run_signature = ...`, `self.run_signature_hash = ...`, and finish with "
                "a boolean verdict `self.stop_training = ...`."
            ),
        },
        {
            "role": "user",
            "content": base_user_content,
        },
    ]

    if feedback_context:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Address the error described above by updating the optimizer and/or "
                    "training epoch assignments so that they can be executed without raising "
                    "exceptions."
                ),
            }
        )

    max_attempts = 3
    normalized = ""
    for attempt in range(1, max_attempts + 1):
        response = _call_openai_chat(messages, model_name, api_key)
        cleaned = _strip_code_fences(response)
        (
            layers_part,
            optimizer_part,
            epochs_part,
            scheduler_part,
            loss_part,
            stop_part,
            patience_part,
            min_delta_part,
            signature_part,
            signature_hash_part,
        ) = _split_generated_assignments(cleaned)
        normalized = _normalize_assignment(layers_part)

        if not layers_part or "self.layers" not in layers_part:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Reminder: respond with `self.layers = ...`, `self.optimizer = ...`,"
                        " and `self.training_epochs = ...` as separate lines."
                    ),
                }
            )
            continue

        if normalized in seen_assignments:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "We have already used that exact configuration. Please return"
                        " a different layer assignment that is not identical to any"
                        " previously listed."
                    ),
                }
            )
            continue

        if not optimizer_part:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You must also provide an optimizer assignment on a separate"
                        " line in the form `self.optimizer = ...`."
                    ),
                }
            )
            continue

        if not epochs_part:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please include a training duration assignment like"
                        " `self.training_epochs = <positive integer>`."
                    ),
                }
            )
            continue

        if not patience_part:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Add an early stopping patience line in the form "
                        "`self.early_stop_patience = <positive integer>` with an inline summary."
                    ),
                }
            )
            continue

        if not min_delta_part:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Include an early stopping delta line `self.early_stop_min_delta = <positive float>` "
                        "with a brief inline rationale."
                    ),
                }
            )
            continue

        if not scheduler_part:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Include a scheduler assignment in the form `self.scheduler = ...` with an inline"
                        " summary; use `self.scheduler = None` if no scheduler is desired."
                    ),
                }
            )
            continue

        if not loss_part:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Add a loss assignment like `self.loss = ...` describing the chosen criterion."
                    ),
                }
            )
            continue

        if not signature_part:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Define `self.run_signature = (...)` capturing (layers_repr, optimiser, lr, schedule,"
                        " batch_size, weight_decay, seed) with a verifying inline comment."
                    ),
                }
            )
            continue

        if not signature_hash_part:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Compute `self.run_signature_hash = ...` (for example via hashlib.sha256) so the"
                        " tuple above has a reproducible identifier."
                    ),
                }
            )
            continue

        if not stop_part:
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Add a boolean verdict `self.stop_training = ...` with an inline comment"
                        " indicating whether to halt further exploration." 
                    ),
                }
            )
            continue

        ordered_parts = [
            layers_part,
            optimizer_part,
            epochs_part,
            patience_part,
            min_delta_part,
            scheduler_part,
            loss_part,
            signature_part,
            signature_hash_part,
            stop_part,
        ]

        return "\n".join([part for part in ordered_parts if part])


    if normalized:
        raise RuntimeError(
            "generate_layers failed to produce a novel architecture after multiple attempts."
        )
    raise RuntimeError("generate_layers returned an empty layer assignment.")

# ----------------------
# Model Definitions
# ----------------------

class SimpleClassifier(nn.Module):
    """
    Dynamically generated classifier using OpenAI API.
    The layers are assigned by executing code returned from generate_layers().
    """
    def __init__(
        self,
        cfg: Any,
        model_name: str = "gpt-4o-mini",
        feedback: Optional[str] = None,
    ) -> None:
        super().__init__()
        try:
            generated_layers = generate_layers(cfg, model_name=model_name, feedback=feedback)
        except Exception as exc:
            raise RuntimeError(f"generate_layers failed: {exc}")

        print("Generated layers assignment:")
        print(generated_layers)

        (
            layer_code,
            optimizer_code,
            epochs_code,
            scheduler_code,
            loss_code,
            stop_code,
            patience_code,
            min_delta_code,
            signature_code,
            signature_hash_code,
        ) = _split_generated_assignments(generated_layers)

        # Keep a copy so training summaries can reference the generated architecture
        self.generated_layers_code = generated_layers
        self.generated_optimizer_code = optimizer_code or ""
        self.generated_epochs_code = epochs_code or ""
        self.generated_scheduler_code = scheduler_code or ""
        self.generated_loss_code = loss_code or ""
        self.generated_stop_code = stop_code or ""
        self.generated_patience_code = patience_code or ""
        self.generated_min_delta_code = min_delta_code or ""
        self.generated_signature_code = signature_code or ""
        self.generated_signature_hash_code = signature_hash_code or ""

        if not self.generated_signature_code:
            raise RuntimeError("Generated response omitted the required `self.run_signature = ...` assignment.")
        if not self.generated_signature_hash_code:
            raise RuntimeError("Generated response omitted the required `self.run_signature_hash = ...` assignment.")

        if layer_code and "self.layers" in layer_code:
            try:
                layer_code = _repair_inline_comments(layer_code)
                exec(
                    layer_code,
                    {"nn": nn, "torch": torch, "cfg": cfg, "hashlib": hashlib},
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
        xs: torch.Tensor,
        ys: torch.Tensor,
        *,
        device: Optional[str] = None,
        train_split: float = 0.8,
        batch_size: int = 64,
        results_log_path: Optional[str] = "training_runs.jsonl",
        model_name: str = "gpt-4o-mini",
    ) -> None:
        if xs.ndim != 2:
            raise ValueError("`xs` must be a 2D tensor shaped [num_samples, num_features].")
        if ys.ndim != 1:
            raise ValueError("`ys` must be a 1D tensor of class labels.")
        if train_split <= 0 or train_split >= 1:
            raise ValueError("`train_split` must be between 0 and 1.")
        if batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        input_dim = xs.shape[1]
        unique_classes = torch.unique(ys)
        if unique_classes.numel() == 0:
            raise ValueError("`ys` must contain at least one class label.")
        num_classes = int(unique_classes.numel())

        log_path_value = str(results_log_path) if results_log_path is not None else "training_runs.jsonl"

        self._model_context = SimpleNamespace(
            input_dim=int(input_dim),
            num_classes=num_classes,
            results_log_path=log_path_value,
            device=device,
            learning_rate=None,
            epochs=None,
            train_split=float(train_split),
            batch_size=int(batch_size),
            seed=42,
            class_weights=None,
            scheduler=None,
            pos_weight=None,
            loss=None,
            early_stop_patience=None,
            early_stop_min_delta=None,
        )

        self.device = torch.device(device)
        self.input_dim = int(input_dim)
        self.model_name = model_name

        seed_value = 42
        try:
            seed_value = int(getattr(self._model_context, "seed", 42))
        except (TypeError, ValueError):
            seed_value = 42
        self._model_context.seed = seed_value
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

        self.rng = torch.Generator().manual_seed(seed_value)
        self.loader_rng = torch.Generator().manual_seed(seed_value)

        self.criterion: nn.Module

        self.model: Optional[SimpleClassifier] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.epochs = 0
        self.early_stop_patience = 12
        self.early_stop_min_delta = 0.002

        self.training_history: List[Dict[str, Any]] = []
        self.training_runs: List[Dict[str, Any]] = []
        self.latest_results: Dict[str, Any] = {}
        self.results_log_path = Path(log_path_value).expanduser()
        self.stop_requested = False
        self.stop_verdict_code = ""
        self.best_model_bundle: Optional[Dict[str, Any]] = None
        self._best_val_acc: float = float("-inf")

        train_ds, val_ds = self.split_dataset(
            xs,
            ys,
            train_split,
            generator=self.rng,
            num_classes=num_classes,
        )

        ys_train = train_ds.tensors[1]
        class_counts = torch.bincount(ys_train, minlength=num_classes).float()
        total = class_counts.sum().clamp(min=1)
        denom = (class_counts * float(num_classes)).clamp(min=1)
        class_weights = total / denom
        self._model_context.class_weights = class_weights.tolist()
        pos_weight_value: Optional[float] = None
        if num_classes == 2 and class_counts[1] > 0:
            pos_weight_value = float((class_counts[0] / class_counts[1]).item())
        self._model_context.pos_weight = pos_weight_value
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        self.train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=self.loader_rng,
        )
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    @staticmethod
    def split_dataset(
        xs: torch.Tensor,
        ys: torch.Tensor,
        train_ratio: float,
        *,
        generator: Optional[torch.Generator] = None,
        num_classes: int = 2,
    ) -> Tuple[TensorDataset, TensorDataset]:
        """Stratified split into train and validation sets when possible."""
        if generator is None:
            generator = torch.Generator()

        if num_classes == 2:
            pos_idx = torch.nonzero(ys == 1, as_tuple=True)[0]
            neg_idx = torch.nonzero(ys == 0, as_tuple=True)[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                indices = torch.randperm(len(xs), generator=generator)
            else:
                pos_perm = pos_idx[torch.randperm(len(pos_idx), generator=generator)]
                neg_perm = neg_idx[torch.randperm(len(neg_idx), generator=generator)]
                n_pos_train = max(1, int(len(pos_perm) * train_ratio))
                n_neg_train = max(1, int(len(neg_perm) * train_ratio))
                n_pos_train = min(n_pos_train, len(pos_perm) - 1 if len(pos_perm) > 1 else len(pos_perm))
                n_neg_train = min(n_neg_train, len(neg_perm) - 1 if len(neg_perm) > 1 else len(neg_perm))

                if n_pos_train == 0 or n_neg_train == 0:
                    indices = torch.randperm(len(xs), generator=generator)
                else:
                    train_idx = torch.cat([pos_perm[:n_pos_train], neg_perm[:n_neg_train]])
                    val_idx = torch.cat([pos_perm[n_pos_train:], neg_perm[n_neg_train:]])
                    perm_t = torch.randperm(train_idx.size(0), generator=generator)
                    perm_v = torch.randperm(val_idx.size(0), generator=generator)
                    train_idx = train_idx[perm_t]
                    val_idx = val_idx[perm_v]
                    return (
                        TensorDataset(xs[train_idx], ys[train_idx]),
                        TensorDataset(xs[val_idx], ys[val_idx]),
                    )
        else:
            # General multi-class stratified split
            train_parts: List[torch.Tensor] = []
            val_parts: List[torch.Tensor] = []
            for class_id in range(num_classes):
                class_idx = torch.nonzero(ys == class_id, as_tuple=True)[0]
                if len(class_idx) == 0:
                    continue
                permuted = class_idx[torch.randperm(len(class_idx), generator=generator)]
                cutoff = int(len(permuted) * train_ratio)
                cutoff = min(max(cutoff, 1), len(permuted))
                train_parts.append(permuted[:cutoff])
                if cutoff < len(permuted):
                    val_parts.append(permuted[cutoff:])
            if train_parts and val_parts:
                train_idx = torch.cat(train_parts)
                val_idx = torch.cat(val_parts)
                perm_t = torch.randperm(train_idx.size(0), generator=generator)
                perm_v = torch.randperm(val_idx.size(0), generator=generator)
                train_idx = train_idx[perm_t]
                val_idx = val_idx[perm_v]
                return (
                    TensorDataset(xs[train_idx], ys[train_idx]),
                    TensorDataset(xs[val_idx], ys[val_idx]),
                )

        # Fallback if stratification fails
        indices = torch.randperm(len(xs), generator=generator)
        num_train = int(len(xs) * train_ratio)
        train_idx = indices[:num_train]
        val_idx = indices[num_train:]
        return (
            TensorDataset(xs[train_idx], ys[train_idx]),
            TensorDataset(xs[val_idx], ys[val_idx]),
        )

    def train(self) -> float:
        """Train for one epoch and return average loss."""
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Call `fit` before `train`; the model has not been initialized.")
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

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set. Returns (avg_loss, accuracy)."""
        if self.model is None:
            raise RuntimeError("Call `fit` before `evaluate`; the model has not been initialized.")
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

    def fit(self, *, max_iterations: Optional[int] = None) -> None:
        """Train for the configured number of epochs, optionally repeating multiple runs."""
        if max_iterations is None:
            iterations = 1
        else:
            if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral):
                raise TypeError("`max_iterations` must be an integer.")
            iterations = int(max_iterations)
            if iterations <= 0:
                raise ValueError("`max_iterations` must be a positive integer.")

        for iteration in range(iterations):
            if iterations > 1:
                print(f"Starting training iteration {iteration + 1}/{iterations}")

            self._initialize_training_components()

            if self.stop_requested:
                saved_path = self._save_best_model()
                verdict = (self.stop_verdict_code or "self.stop_training = True").strip()
                if saved_path is not None:
                    print(
                        f"ChatGPT recommended stopping further training ({verdict}). "
                        f"Saved best model to {saved_path}."
                    )
                else:
                    print(
                        f"ChatGPT recommended stopping further training ({verdict}), "
                        "but no trained model was available to save."
                    )
                break

            patience = int(self.early_stop_patience)
            min_delta = float(self.early_stop_min_delta)
            stalled = 0
            best_val_loss = float("inf")
            best_val_acc = 0.0
            best_train_loss = float("inf")
            best_epoch = 0
            best_state: Optional[Dict[str, torch.Tensor]] = None

            history: List[Dict[str, Any]] = []

            for epoch in range(1, self.epochs + 1):
                train_loss = self.train()
                val_loss, val_acc = self.evaluate()

                if self.scheduler is not None:
                    try:
                        self.scheduler.step(val_loss)
                    except TypeError:
                        self.scheduler.step()

                print(
                    f"Epoch {epoch}/{self.epochs} | Train loss: {train_loss:.3f} "
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

                improved = (val_loss < (best_val_loss - min_delta)) or (
                    abs(val_loss - best_val_loss) <= min_delta and val_acc > best_val_acc
                )

                if improved or best_state is None:
                    best_val_loss = float(val_loss)
                    best_val_acc = float(val_acc)
                    best_train_loss = float(train_loss)
                    best_epoch = epoch
                    stalled = 0
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                else:
                    stalled += 1

                if stalled >= patience:
                    print(
                        f"Early stopping at epoch {epoch}: no val_loss improvement >={min_delta:.3f}"
                        f" for {patience} checks."
                    )
                    break

            if best_state is not None:
                self.model.load_state_dict(best_state)

            history_copy = [dict(record) for record in history]
            final_metrics = {
                "best_epoch": int(best_epoch),
                "epochs_trained": len(history_copy),
                "train_loss": float(best_train_loss),
                "val_loss": float(best_val_loss),
                "val_acc": float(best_val_acc),
            }
            run_record = {
                "generated_layers": getattr(self.model, "generated_layers_code", ""),
                "history": history_copy,
                "final_metrics": final_metrics,
                "patience": patience,
                "min_delta": min_delta,
                "patience_code": getattr(self.model, "generated_patience_code", ""),
                "min_delta_code": getattr(self.model, "generated_min_delta_code", ""),
                "stopped_early": stalled >= patience,
                "scheduler": self.scheduler.__class__.__name__ if self.scheduler else None,
                "seed": self._model_context.seed,
                "run_signature": getattr(self.model, "run_signature", None),
                "run_signature_hash": getattr(self.model, "run_signature_hash", None),
                "signature_code": getattr(self.model, "generated_signature_code", ""),
                "signature_hash_code": getattr(self.model, "generated_signature_hash_code", ""),
            }
            run_record["stop_code"] = getattr(self.model, "generated_stop_code", "")
            self.training_history = history_copy
            self.latest_results = run_record
            self.training_runs.append(run_record)
            self._append_run_to_log(run_record)

            if best_state is not None:
                val_acc_score = float(final_metrics.get("val_acc", float("-inf")))
                if val_acc_score >= self._best_val_acc:
                    self._best_val_acc = val_acc_score
                    state_copy = {key: tensor.clone() for key, tensor in best_state.items()}
                    self.best_model_bundle = {
                        "state_dict": state_copy,
                        "generated_layers": getattr(self.model, "generated_layers_code", ""),
                        "optimizer": getattr(self.model, "generated_optimizer_code", ""),
                        "epochs_assignment": getattr(self.model, "generated_epochs_code", ""),
                        "scheduler": getattr(self.model, "generated_scheduler_code", ""),
                        "loss": getattr(self.model, "generated_loss_code", ""),
                        "stop_code": getattr(self.model, "generated_stop_code", ""),
                        "patience_code": getattr(self.model, "generated_patience_code", ""),
                        "min_delta_code": getattr(self.model, "generated_min_delta_code", ""),
                        "signature_code": getattr(self.model, "generated_signature_code", ""),
                        "signature_hash_code": getattr(self.model, "generated_signature_hash_code", ""),
                        "run_signature": getattr(self.model, "run_signature", None),
                        "run_signature_hash": getattr(self.model, "run_signature_hash", None),
                        "patience": patience,
                        "min_delta": min_delta,
                        "final_metrics": final_metrics,
                        "iteration": iteration + 1,
                        "input_dim": self.input_dim,
                        "num_classes": self._model_context.num_classes,
                        "train_split": self._model_context.train_split,
                    }

    def predict(self, sample: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities for a sample batch."""
        if self.model is None:
            raise RuntimeError("Call `fit` before `predict`; the model has not been initialized.")
        if sample.ndim != 2:
            raise ValueError("`sample` must be a 2D tensor shaped [batch, num_features].")
        if sample.shape[1] != self.input_dim:
            raise ValueError(
                f"`sample` has {sample.shape[1]} features, but the model expects {self.input_dim}."
            )
        with torch.no_grad():
            logits = self.model(sample.to(self.device))
            probs = torch.softmax(logits, dim=1)
            return probs.cpu()

    def _initialize_training_components(self) -> None:
        """(Re)create the model, optimizer, and training epoch count."""
        max_attempts = 5
        feedback: Optional[str] = None
        last_error: Optional[str] = None

        self.stop_requested = False
        self.stop_verdict_code = ""
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epochs = 0
        self.early_stop_patience = 12
        self.early_stop_min_delta = 0.002

        for attempt in range(1, max_attempts + 1):
            self._model_context.learning_rate = None
            self._model_context.scheduler = None
            self._model_context.epochs = None
            self._model_context.early_stop_patience = None
            self._model_context.early_stop_min_delta = None

            try:
                model = SimpleClassifier(
                    self._model_context,
                    model_name=self.model_name,
                    feedback=feedback,
                ).to(self.device)
            except Exception as exc:
                error_text = str(exc)
                last_error = error_text
                feedback = f"Layer generation failed with error: {error_text}"
                print(
                    f"Regenerating architecture (attempt {attempt}/{max_attempts}) after failure: {error_text}"
                )
                continue

            self.model = model

            (
                _layer_assignment,
                optimizer_assignment,
                epochs_assignment,
                scheduler_assignment,
                loss_assignment,
                stop_assignment,
                patience_assignment,
                min_delta_assignment,
                signature_assignment,
                signature_hash_assignment,
            ) = _split_generated_assignments(
                getattr(model, "generated_layers_code", "")
            )

            self.stop_verdict_code = stop_assignment or ""

            try:
                should_stop = self._resolve_stop_signal(stop_assignment)
            except RuntimeError as exc:
                last_error = str(exc)
                feedback = last_error
                print(
                    f"Regenerating architecture (attempt {attempt}/{max_attempts}) after failure: {last_error}"
                )
                self.model = None
                continue

            if should_stop:
                self.stop_requested = True
                self.model = None
                self.optimizer = None
                self.scheduler = None
                self.epochs = 0
                return

            try:
                optimizer = self._build_optimizer(optimizer_assignment)
            except RuntimeError as exc:
                command = (optimizer_assignment or "<missing optimizer command>").strip()
                error_text = str(exc)
                last_error = f"Optimizer command `{command}` failed: {error_text}"
                if "no attribute 'layers'" in error_text:
                    last_error += (
                        " Use `self.model.parameters()` (or `self.parameters()` within the optimizer"
                        " context) when constructing the optimizer."
                    )
                feedback = last_error
                print(
                    f"Regenerating architecture (attempt {attempt}/{max_attempts}) after failure: {error_text}"
                )
                self.model = None
                self.optimizer = None
                continue

            try:
                resolved_epochs = self._resolve_training_epochs(epochs_assignment)
            except RuntimeError as exc:
                command = (epochs_assignment or "<missing training epochs command>").strip()
                error_text = str(exc)
                last_error = f"Training epochs command `{command}` failed: {error_text}"
                feedback = last_error
                print(
                    f"Regenerating architecture (attempt {attempt}/{max_attempts}) after failure: {error_text}"
                )
                self.model = None
                self.optimizer = None
                continue

            self.optimizer = optimizer
            self.epochs = resolved_epochs
            self._model_context.epochs = resolved_epochs
            if optimizer.param_groups:
                lr_value = optimizer.param_groups[0].get("lr")
                if lr_value is not None:
                    self._model_context.learning_rate = float(lr_value)
            else:
                self._model_context.learning_rate = None

            try:
                patience_value = self._resolve_early_stop_patience(patience_assignment)
            except RuntimeError as exc:
                command = (patience_assignment or "<missing early stop patience>").strip()
                error_text = str(exc)
                last_error = f"Early stop patience command `{command}` failed: {error_text}"
                feedback = last_error
                print(
                    f"Regenerating architecture (attempt {attempt}/{max_attempts}) after failure: {error_text}"
                )
                self.model = None
                self.optimizer = None
                continue

            try:
                min_delta_value = self._resolve_early_stop_min_delta(min_delta_assignment)
            except RuntimeError as exc:
                command = (min_delta_assignment or "<missing early stop min_delta>").strip()
                error_text = str(exc)
                last_error = f"Early stop min_delta command `{command}` failed: {error_text}"
                feedback = last_error
                print(
                    f"Regenerating architecture (attempt {attempt}/{max_attempts}) after failure: {error_text}"
                )
                self.model = None
                self.optimizer = None
                continue

            self.early_stop_patience = patience_value
            self.early_stop_min_delta = min_delta_value
            self._model_context.early_stop_patience = patience_value
            self._model_context.early_stop_min_delta = min_delta_value

            try:
                self.scheduler = self._build_scheduler(scheduler_assignment)
            except RuntimeError as exc:
                command = (scheduler_assignment or "<scheduler not provided>").strip()
                error_text = str(exc)
                last_error = f"Scheduler command `{command}` failed: {error_text}"
                feedback = last_error
                print(
                    f"Regenerating architecture (attempt {attempt}/{max_attempts}) after failure: {error_text}"
                )
                self.model = None
                self.optimizer = None
                continue

            try:
                self._maybe_apply_custom_loss(loss_assignment)
            except RuntimeError as exc:
                command = (loss_assignment or "<loss not provided>").strip()
                error_text = str(exc)
                last_error = f"Loss command `{command}` failed: {error_text}"
                feedback = last_error
                print(
                    f"Regenerating architecture (attempt {attempt}/{max_attempts}) after failure: {error_text}"
                )
                self.model = None
                self.optimizer = None
                continue

            try:
                self._apply_run_signature(signature_assignment, signature_hash_assignment)
            except RuntimeError as exc:
                command = (signature_assignment or "<run signature missing>").strip()
                error_text = str(exc)
                last_error = f"Run signature command `{command}` failed: {error_text}"
                feedback = last_error
                print(
                    f"Regenerating architecture (attempt {attempt}/{max_attempts}) after failure: {error_text}"
                )
                self.model = None
                self.optimizer = None
                self.scheduler = None
                continue

            return

        raise RuntimeError(
            "Failed to initialize training components after multiple regeneration attempts. "
            f"Last error: {last_error}"
        )

    def _append_run_to_log(self, run_record: Dict[str, Any]) -> None:
        try:
            log_path = self.results_log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(run_record))
                log_file.write("\n")
        except OSError as exc:
            print(f"Warning: unable to write training log to {log_path}: {exc}")

    def _build_optimizer(self, command: Optional[str]) -> torch.optim.Optimizer:
        if not command:
            raise RuntimeError("Generated response omitted an optimizer assignment.")

        try:
            class _OptimizerContext:
                def __init__(self, trainer: "GPTTrainer") -> None:
                    self._trainer = trainer
                    self.model = trainer.model
                    self.optimizer: Optional[torch.optim.Optimizer] = None

                def parameters(self):
                    return self.model.parameters()

            context = _OptimizerContext(self)
            exec(command, {"torch": torch, "nn": nn, "cfg": self._model_context}, {"self": context})
            optimizer = context.optimizer
        except Exception as exc:
            raise RuntimeError(
                f"Generated optimizer command failed with error: {exc}"
            ) from exc

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeError(
                "Generated optimizer command did not assign a torch.optim.Optimizer to `self.optimizer`."
            )

        return optimizer

    def _build_scheduler(self, command: Optional[str]) -> Optional[Any]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be initialized before creating a scheduler.")

        if not command:
            raise RuntimeError("Generated response omitted a scheduler assignment.")

        try:
            class _SchedulerContext:
                def __init__(self, trainer: "GPTTrainer") -> None:
                    self.optimizer = trainer.optimizer
                    self.scheduler: Optional[Any] = None
                    self.training_epochs = trainer.epochs
                    self.epochs = trainer.epochs

            context = _SchedulerContext(self)
            exec(command, {"torch": torch, "nn": nn, "cfg": self._model_context}, {"self": context})
            scheduler = context.scheduler
        except Exception as exc:
            raise RuntimeError(
                f"Generated scheduler command failed with error: {exc}"
            ) from exc

        if scheduler is not None and not hasattr(scheduler, "step"):
            raise RuntimeError("Generated scheduler object must define a `step` method.")

        self._model_context.scheduler = scheduler.__class__.__name__ if scheduler else None
        return scheduler

    def _maybe_apply_custom_loss(self, command: Optional[str]) -> None:
        if not command:
            raise RuntimeError("Generated response omitted a loss assignment.")

        try:
            class _LossContext:
                def __init__(self, trainer: "GPTTrainer") -> None:
                    self.model = trainer.model
                    self.loss: Optional[nn.Module] = None
                    class_weights = trainer._model_context.class_weights or []
                    if class_weights:
                        tensor = torch.tensor(class_weights, dtype=torch.float32, device=trainer.device)
                    else:
                        tensor = None
                    self.class_weights = tensor
                    pos_weight_value = trainer._model_context.pos_weight
                    if pos_weight_value is not None:
                        self.pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=trainer.device)
                    else:
                        self.pos_weight = None
                    self.device = trainer.device

            context = _LossContext(self)
            exec(command, {"torch": torch, "nn": nn, "cfg": self._model_context}, {"self": context})
            loss_module = context.loss
        except Exception as exc:
            raise RuntimeError(
                f"Generated loss command failed with error: {exc}"
            ) from exc

        if loss_module is None:
            raise RuntimeError("Generated loss command did not assign `self.loss`.")
        if not isinstance(loss_module, nn.Module):
            raise RuntimeError(
                "Generated loss command must assign a torch.nn.Module instance to `self.loss`."
            )

        self.criterion = loss_module.to(self.device)
        self._model_context.loss = loss_module.__class__.__name__

    def _apply_run_signature(
        self,
        signature_command: Optional[str],
        signature_hash_command: Optional[str],
    ) -> None:
        if self.model is None:
            raise RuntimeError("Model must be initialized before applying the run signature.")
        if not signature_command:
            raise RuntimeError("Generated response omitted the required `self.run_signature = ...` assignment.")
        if not signature_hash_command:
            raise RuntimeError("Generated response omitted the required `self.run_signature_hash = ...` assignment.")

        model = self.model
        optimizer = self.optimizer

        setattr(model, "optimizer", optimizer)
        setattr(model, "scheduler", self.scheduler)
        setattr(model, "batch_size", int(self._model_context.batch_size))
        setattr(model, "seed", int(self._model_context.seed))

        weight_decay: Optional[float] = None
        learning_rate: Optional[float] = None
        if optimizer is not None and optimizer.param_groups:
            params = optimizer.param_groups[0]
            lr_value = params.get("lr")
            wd_value = params.get("weight_decay")
            if lr_value is not None:
                try:
                    learning_rate = float(lr_value)
                except (TypeError, ValueError):
                    learning_rate = None
            if wd_value is not None:
                try:
                    weight_decay = float(wd_value)
                except (TypeError, ValueError):
                    weight_decay = None
        setattr(model, "learning_rate", learning_rate)
        setattr(model, "weight_decay", weight_decay)

        signature_code = _repair_inline_comments(signature_command)
        signature_hash_code = _repair_inline_comments(signature_hash_command)

        try:
            exec(
                signature_code,
                {"nn": nn, "torch": torch, "cfg": self._model_context, "hashlib": hashlib},
                {"self": model},
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to apply generated run signature: {exc}")

        if not hasattr(model, "run_signature"):
            raise RuntimeError("Generated run signature command did not assign `self.run_signature`." )

        signature_value = getattr(model, "run_signature")
        if not isinstance(signature_value, tuple) or len(signature_value) != 7:
            raise RuntimeError(
                "`self.run_signature` must be a tuple with seven elements: (layers_repr, optimiser, lr, schedule, batch_size, weight_decay, seed)."
            )

        layer_repr, optimiser_name, lr_value, schedule_name, batch_size_value, weight_decay_value, seed_value = signature_value

        if not isinstance(layer_repr, str) or not layer_repr.strip():
            raise RuntimeError("First element of `self.run_signature` must be a non-empty string describing the layers.")
        if not isinstance(optimiser_name, str) or not optimiser_name.strip():
            raise RuntimeError("Second element of `self.run_signature` must be a non-empty string naming the optimizer.")
        if not isinstance(lr_value, (int, float)):
            raise RuntimeError("Third element of `self.run_signature` must be a numeric learning rate.")
        if not isinstance(schedule_name, str) or not schedule_name.strip():
            raise RuntimeError("Fourth element of `self.run_signature` must be a non-empty string naming the scheduler.")
        if not isinstance(batch_size_value, (int, float)):
            raise RuntimeError("Fifth element of `self.run_signature` must be a numeric batch size.")
        if not isinstance(weight_decay_value, (int, float)):
            raise RuntimeError("Sixth element of `self.run_signature` must be a numeric weight decay.")
        if not isinstance(seed_value, (int, float)):
            raise RuntimeError("Seventh element of `self.run_signature` must be a numeric seed.")

        try:
            exec(
                signature_hash_code,
                {"nn": nn, "torch": torch, "cfg": self._model_context, "hashlib": hashlib},
                {"self": model},
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to apply generated run signature hash: {exc}")

        if not hasattr(model, "run_signature_hash"):
            raise RuntimeError("Generated run signature hash command did not assign `self.run_signature_hash`." )

        signature_hash = getattr(model, "run_signature_hash")
        if not isinstance(signature_hash, str) or not signature_hash.strip():
            raise RuntimeError("`self.run_signature_hash` must be a non-empty string.")

        expected_hash = hashlib.sha256(repr(signature_value).encode("utf-8")).hexdigest()
        if signature_hash != expected_hash:
            raise RuntimeError(
                "`self.run_signature_hash` must equal `hashlib.sha256(repr(self.run_signature).encode('utf-8')).hexdigest()`."
            )

    def _resolve_training_epochs(self, command: Optional[str]) -> int:
        if not command:
            raise RuntimeError("Generated response omitted a training epochs assignment.")

        try:
            class _EpochContext:
                def __init__(self) -> None:
                    self.training_epochs: Optional[int] = None

            context = _EpochContext()
            exec(command, {"cfg": self._model_context}, {"self": context})
            value = getattr(context, "training_epochs", None)
        except Exception as exc:
            raise RuntimeError(
                f"Generated training epochs command failed with error: {exc}"
            ) from exc

        if value is None:
            raise RuntimeError(
                "Generated training epochs command did not assign `self.training_epochs`."
            )

        resolved = int(value)
        if resolved <= 0:
            raise RuntimeError("Generated training epochs must be a positive integer.")

        return resolved

    def _resolve_early_stop_patience(self, command: Optional[str]) -> int:
        if not command:
            raise RuntimeError("Generated response omitted an early stop patience assignment.")

        try:
            class _PatienceContext:
                def __init__(self) -> None:
                    self.early_stop_patience: Optional[Any] = None

            context = _PatienceContext()
            exec(command, {"cfg": self._model_context}, {"self": context})
            value = getattr(context, "early_stop_patience", None)
        except Exception as exc:
            raise RuntimeError(
                f"Generated early stop patience command failed with error: {exc}"
            ) from exc

        if value is None:
            raise RuntimeError(
                "Generated early stop patience command did not assign `self.early_stop_patience`."
            )

        resolved = int(value)
        if resolved <= 0:
            raise RuntimeError("Early stop patience must be a positive integer.")
        if resolved > 500:
            raise RuntimeError("Early stop patience must be <= 500 epochs.")

        return resolved

    def _resolve_early_stop_min_delta(self, command: Optional[str]) -> float:
        if not command:
            raise RuntimeError("Generated response omitted an early stop min_delta assignment.")

        try:
            class _DeltaContext:
                def __init__(self) -> None:
                    self.early_stop_min_delta: Optional[Any] = None

            context = _DeltaContext()
            exec(command, {"cfg": self._model_context}, {"self": context})
            value = getattr(context, "early_stop_min_delta", None)
        except Exception as exc:
            raise RuntimeError(
                f"Generated early stop min_delta command failed with error: {exc}"
            ) from exc

        if value is None:
            raise RuntimeError(
                "Generated early stop min_delta command did not assign `self.early_stop_min_delta`."
            )

        try:
            resolved = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Early stop min_delta must be a numeric value.") from exc

        if resolved <= 0.0:
            raise RuntimeError("Early stop min_delta must be greater than zero.")
        if resolved >= 1.0:
            raise RuntimeError("Early stop min_delta must be less than 1.0.")

        return resolved

    @staticmethod
    def _resolve_stop_signal(command: Optional[str]) -> bool:
        if not command:
            return False

        try:
            class _StopContext:
                def __init__(self) -> None:
                    self.stop_training: Optional[Any] = None

            context = _StopContext()
            exec(command, {}, {"self": context})
            value = getattr(context, "stop_training", None)
        except Exception as exc:
            raise RuntimeError(
                f"Generated stop command failed with error: {exc}"
            ) from exc

        if value is None:
            raise RuntimeError(
                "Generated stop command did not assign `self.stop_training`."
            )

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y"}:
                return True
            if lowered in {"false", "no", "n"}:
                return False

        raise RuntimeError("`self.stop_training` must resolve to a boolean value.")

    def _save_best_model(self, path: Optional[str] = None) -> Optional[Path]:
        if self.best_model_bundle is None:
            return None

        target = Path(path).expanduser() if path else Path("best_model.pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.best_model_bundle, target)
        return target
