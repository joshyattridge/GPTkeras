# kerasgpt

Toolkit for experimenting with GPT-assisted configuration of Keras projects. The package wraps the OpenAI Chat Completions API and provides helpers for drafting model architectures, managing experiments, and logging conversational context when iterating on training runs.

## Installation

```bash
pip install kerasgpt
```

When working from a clone of this repository you can install an editable build instead:

```bash
pip install -e .
```

## Quick start

```python
from kerasgpt import kerasgpt

model = kerasgpt(train_x, train_y, api_key="sk-...")
model.build_model()
model.fit(max_iterations=3)
```
