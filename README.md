# KerasCrafterGPT

KerasCrafterGPT aims to solve the struggles of guessing model configurations and hyperparameters when developing machine learning projects. Instead of manually tuning architectures and settings, you simply provide your training data (`x` and `y`), and this library leverages large language models (LLMs) specifically ChatGPT to automatically generate suitable layers and hyperparameters for your dataset.

The workflow is as follows:

- You provide your training data.
- The library uses ChatGPT to draft a Keras model architecture and select hyperparameters tailored to your data.
- It runs a training session and evaluates the results.
- The training results are then passed back to the LLM, which attempts to further improve the model configuration in subsequent iterations.

Another major challenge in machine learning experimentation is tracking changes to models and understanding how those changes affect results. KerasCrafterGPT addresses this by using the LLM to log and manage conversational context, so you can see how each modification impacts performance over time.

This toolkit wraps the OpenAI Chat Completions API and provides helpers for drafting model architectures, managing experiments, and logging conversational context when iterating on training runs.

## Installation

```bash
pip install KerasCrafterGPT
```

## Quick start

```python
from KerasCrafterGPT import KerasCrafterGPT

model = KerasCrafterGPT(train_x, train_y, api_key="sk-...")
model.fit(max_iterations=3)
```

## Documentation

### KerasCrafterGPT Arguments

```python
KerasCrafterGPT(
	train_x,                      # Required. Training input data (numpy array or tensor)
	train_y,                      # Required. Training target data (numpy array or tensor)
	api_key=None,                 # Required. OpenAI API key for LLM access
	model="gpt-4o-mini",         # Optional. OpenAI model name (default: "gpt-4o-mini")
	history_path="chat_history.jsonl", # Optional. Path to log conversational/model history (default: "chat_history.jsonl")
	continue_from_history=False,  # Optional. If True, resumes conversation/model history from previous runs (default: False)
)
```

## Contributing

Contributions, suggestions, and bug reports are welcome! Feel free to open issues or submit pull requests to help improve KerasCrafterGPT.
