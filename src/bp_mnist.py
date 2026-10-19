import csv
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
EPOCHS = 5
BATCH_SIZE = 128

random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

device = torch.device("cpu")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path.home() / "datasets"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=str(DATA_DIR), train=True,
    download=False, transform=transform
)
test_dataset = datasets.MNIST(
    root=str(DATA_DIR), train=False,
    download=False, transform=transform
)

generator = torch.Generator().manual_seed(SEED)

train_indices = torch.randperm(
    len(train_dataset), generator=generator
)[:5000].tolist()

test_indices = torch.randperm(
    len(test_dataset), generator=generator
)[:1000].tolist()

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    num_workers=0,
)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

results = []

print("Experiment: Plaintext BP convergence test")
print("Device:", device)
print("Training samples:", len(train_subset))
print("Test samples:", len(test_subset))

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.perf_counter()

    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    train_loss = total_loss / total_samples

    model.eval()
    correct = 0
    tested = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels.to(device)
            ).sum().item()
            tested += labels.size(0)

    accuracy = 100.0 * correct / tested
    epoch_seconds = time.perf_counter() - epoch_start

    results.append({
        "seed": SEED,
        "epoch": epoch,
        "train_samples": len(train_subset),
        "test_samples": len(test_subset),
        "train_loss": train_loss,
        "test_accuracy_pct": accuracy,
        "epoch_seconds": epoch_seconds,
    })

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}% | "
        f"Time: {epoch_seconds:.2f}s"
    )

results_file = RESULTS_DIR / "bp_mnist_smoke.csv"

with results_file.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Results saved to:", results_file)
