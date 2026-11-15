"""Independent FP64 reference for the full-architecture smoke test."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from baselines.bp import datasets_and_indices, make_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_ds, test_ds, train_indices, validation_indices, test_indices = (
        datasets_and_indices())
    torch.manual_seed(args.seed)
    model = make_model().double()
    train_indices = train_indices[:9]
    validation_indices = validation_indices[:3]
    test_indices = test_indices[:3]
    order = torch.randperm(9, generator=torch.Generator().manual_seed(args.seed)).tolist()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    train_loss = 0.0
    offset = 0
    for batch_size in (3, 3, 3):
        batch_indices = [train_indices[order[offset + i]]
                         for i in range(batch_size)]
        x = torch.stack([train_ds[index][0].flatten().double()
                         for index in batch_indices])
        y = torch.tensor([train_ds[index][1] for index in batch_indices])
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size
        offset += batch_size

    def evaluate(indices, dataset):
        x = torch.stack([dataset[index][0].flatten().double() for index in indices])
        y = torch.tensor([dataset[index][1] for index in indices])
        with torch.no_grad():
            logits = model(x)
            return (nn.functional.cross_entropy(logits, y).item(),
                    int((logits.argmax(1) == y).sum()))

    validation_loss, validation_correct = evaluate(validation_indices, train_ds)
    test_loss, test_correct = evaluate(test_indices, test_ds)
    args.output.write_text(json.dumps({
        "experiment_id": "SECURE-BP-MATCHED-DIAGNOSTIC-v1",
        "train_loss": train_loss / 9,
        "validation_loss": validation_loss,
        "validation_correct": validation_correct,
        "validation_total": 3,
        "test_loss": test_loss,
        "test_correct": test_correct,
        "test_total": 3,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
