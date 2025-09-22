"""Main script for training and evaluating the toy classifier."""
import torch
from data import make_toy_dataset
from GPTtorch import GPTTrainer

def main() -> None:
    xs, ys = make_toy_dataset()
    trainer = GPTTrainer(xs, ys)
    trainer.fit()
    trainer.fit()


    # final testing
    val_xs, val_ys = trainer.val_loader.dataset.tensors
    probs = trainer.predict(val_xs)
    preds = torch.argmax(probs, dim=1)
    print("\nValidation predictions:")
    for target, pred, prob in zip(val_ys, preds, probs):
        confidence = prob[pred].item()
        print(
            f"  label={target.item()} predicted={pred.item()} confidence={confidence:.2f}"
        )

if __name__ == "__main__":
    main()
