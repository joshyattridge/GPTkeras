# Imports
import os
import json
import inspect
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

# def list_available_layers() -> List[Dict[str, Any]]:
#     """Return a list of all available torch.nn layers and their __init__ arguments."""
#     layer_list: List[Dict[str, Any]] = []
#     for name in dir(nn):
#         obj = getattr(nn, name)
#         if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
#             try:
#                 sig = inspect.signature(obj.__init__)
#                 args = [p for p in sig.parameters if p != "self"]
#                 layer_list.append({"layer": name, "args": args})
#             except Exception:
#                 layer_list.append({"layer": name, "args": "N/A (signature not available)"})
#     return layer_list

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
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract layer, optimizer, epoch, scheduler, and loss assignments."""
    optimizer_prefix = "self.optimizer ="
    epochs_prefix = "self.training_epochs ="
    scheduler_prefix = "self.scheduler ="
    loss_prefix = "self.loss ="
    lines = [line.strip() for line in assignments.splitlines() if line.strip()]
    layer_lines: List[str] = []
    optimizer_line: Optional[str] = None
    epochs_line: Optional[str] = None
    scheduler_line: Optional[str] = None
    loss_line: Optional[str] = None
    for line in lines:
        if line.startswith(optimizer_prefix):
            optimizer_line = line
        elif line.startswith(epochs_prefix):
            epochs_line = line
        elif line.startswith(scheduler_prefix):
            scheduler_line = line
        elif line.startswith(loss_prefix):
            loss_line = line
        else:
            layer_lines.append(line)
    return "\n".join(layer_lines), optimizer_line, epochs_line, scheduler_line, loss_line


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
        layers_code, _, _, _, _ = _split_generated_assignments(run.get("generated_layers", ""))
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

    # layer_catalog = json.dumps(available_layers, indent=2)
    base_user_content = (
        # "Here is the list of available torch.nn layers and their constructor"
        # f" details:\n{layer_catalog}\n\n"
        f"{history_context}\n"
        f"{prior_assignments_context}"
        f"{feedback_context}"
        f"Context: tabular inputs with {cfg.input_dim} normalised features; target optimised with"
        f" `torch.nn.CrossEntropyLoss` over {cfg.num_classes} logits. Classes may be imbalanced up to 70/30, so incorporate"
        " class-weighted loss terms or samplers when helpful. Batch size must be in {64, 128}."
        " Weight decay must be in {0, 1e-4, 5e-4}. Learning rate seeds are {3e-4, 1e-3, 3e-3}."
        " Permissible optimisers: Adam, AdamW, or SGD with momentum {0.8, 0.9}; for Adam/AdamW use betas"
        " in {(0.9, 0.999), (0.9, 0.98)}. Seeds rotate across {42, 13, 37}."
        "Keep parameter count near or below 60k and favour simpler designs when metrics tie."
        " Log (layers_repr, optimiser, lr, schedule, batch_size, weight_decay, seed) and hash that tuple"
        " to avoid reusing near-duplicates."
        f" Generate a PyTorch `nn.Sequential` for inputs of {cfg.input_dim} features and exactly {cfg.num_classes} logits;"
        f" ensure the final layer is `nn.Linear(..., {cfg.num_classes})`. Hidden widths may draw from"
        " {64, 72, 80, 88, 96, 100, 112, 128}. Activations may be ReLU or GELU (prefer GELU when deep)."
        " You may introduce residual MLP blocks, LayerNorm, Dropout, BatchNorm1d, or weighted samplers as needed,"
        " but never repeat an identical layer + hyperparameter tuple."
        "Plateau rule: val_acc is plateaued when it fails to improve by ≥0.2% across any 10-epoch window."
        " After every plateau event, enforce this adaptation cycle on subsequent runs:"
        " (1) widen hidden widths by +50–150%; (2) deepen by adding 1–2 layers or a residual/LayerNorm block;"
        " (3) regularise by tuning Dropout within [0.2, 0.6] or swapping BatchNorm/LayerNorm."
        " The cycle must rotate in order (widen → deepen → regularise → repeat) before widening again."
        " Each adaptation must include a one-line inline Python comment in the `self.layers` assignment"
        " explaining how it satisfies the current adaptation stage and confirms the two-logit output."
        "When a plateau has occurred, also switch the optimizer schedule to"
        " `torch.optim.lr_scheduler.CosineAnnealingLR` or `torch.optim.lr_scheduler.OneCycleLR` for the next run."
        " Base training budget should target 90–100 epochs with patience-based early stopping (patience 10, min_delta 0.002)."
        "Also choose optimizer, learning-rate, batch size, weight decay, scheduler, and seed values that honour the exploration policy"
        " above, referencing `self.model.parameters()` (or `self.parameters()`)."
        "Finally, emit concise inline Python comments after the optimizer and epoch (and scheduler/loss if provided) assignments"
        " summarising how these hyperparameters differ from the prior best configuration."
        " Respond with at least three Python assignments on separate lines in the order required by the system instruction."
        "Do not include code fences or extra narration."
    )

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You write concise Python for PyTorch modules. Respond with at least three "
                "assignments in this order: `self.layers = ...`, `self.optimizer = ...`, "
                "`self.training_epochs = ...`. Optional `self.scheduler = ...` and "
                "`self.loss = ...` assignments may follow if needed."
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

        ordered_parts = [layers_part, optimizer_part, epochs_part]
        if scheduler_part:
            ordered_parts.append(scheduler_part)
        if loss_part:
            ordered_parts.append(loss_part)

        return "\n".join([part for part in ordered_parts if part])


    if normalized:
        raise RuntimeError(
            "generate_layers failed to produce a novel architecture after multiple attempts."
        )
    raise RuntimeError("generate_layers returned an empty layer assignment.")

# # Cache available layers for use in generate_layers
# available_layers = list_available_layers()

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
        ) = _split_generated_assignments(generated_layers)

        # Keep a copy so training summaries can reference the generated architecture
        self.generated_layers_code = generated_layers
        self.generated_optimizer_code = optimizer_code or ""
        self.generated_epochs_code = epochs_code or ""
        self.generated_scheduler_code = scheduler_code or ""
        self.generated_loss_code = loss_code or ""

        if layer_code and "self.layers" in layer_code:
            try:
                exec(
                    layer_code,
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
            device = "cuda" if torch.cuda.is_available() else "cpu"

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

        self.training_history: List[Dict[str, Any]] = []
        self.training_runs: List[Dict[str, Any]] = []
        self.latest_results: Dict[str, Any] = {}
        self.results_log_path = Path(log_path_value).expanduser()

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

            patience = 12
            min_delta = 0.002
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
                "stopped_early": stalled >= patience,
                "scheduler": self.scheduler.__class__.__name__ if self.scheduler else None,
                "seed": self._model_context.seed,
            }
            self.training_history = history_copy
            self.latest_results = run_record
            self.training_runs.append(run_record)
            self._append_run_to_log(run_record)

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

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epochs = 0

        for attempt in range(1, max_attempts + 1):
            self._model_context.learning_rate = None
            self._model_context.scheduler = None
            self._model_context.epochs = None

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
            ) = _split_generated_assignments(
                getattr(model, "generated_layers_code", "")
            )

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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
            self._model_context.scheduler = "ReduceLROnPlateau"
            return scheduler

        try:
            class _SchedulerContext:
                def __init__(self, trainer: "GPTTrainer") -> None:
                    self.optimizer = trainer.optimizer
                    self.scheduler: Optional[Any] = None

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
            self.criterion = self.criterion.to(self.device)
            existing = getattr(self.criterion, "__class__", type(self.criterion))
            self._model_context.loss = existing.__name__
            return

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
