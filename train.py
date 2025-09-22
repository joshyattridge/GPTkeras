"""Main script for training and evaluating the toy classifier."""
import torch
from data import Config, make_toy_dataset
from GPTtorch import GPTTrainer

def main() -> None:
    cfg = Config()
    print(f"Using device: {cfg.device}")

    xs, ys = make_toy_dataset(cfg)
    trainer = GPTTrainer(cfg, xs, ys)
    trainer.fit()


    # final testing
    sample = torch.tensor([[2.5, 0.0], [0.0, -4.0]])
    probs = trainer.predict(sample)
    print("\nSample predictions:")
    for point, prob in zip(sample, probs):
        print(f"  point={point.tolist()} -> class_1_prob={prob[1]:.2%}")

if __name__ == "__main__":
    main()
