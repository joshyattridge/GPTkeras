"""Main script for training and evaluating the toy classifier."""
import torch
from data import make_toy_dataset
from GPTtorch import GPTTrainer

def main() -> None:
    xs, ys = make_toy_dataset()
    trainer = GPTTrainer(xs, ys)

    trainer.fit(max_iterations=100)

if __name__ == "__main__":
    main()
