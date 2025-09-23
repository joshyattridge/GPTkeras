"""Main script for training and evaluating the toy classifier."""
import torch
from data import make_toy_dataset
from GPTtorch import GPTTrainer

def main() -> None:
    xs, ys = make_toy_dataset()
    trainer = GPTTrainer(xs, ys)

    for i in range(10):
        trainer.fit()

if __name__ == "__main__":
    main()
