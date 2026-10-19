
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Reproducible smoke-test settings
SEED = 42
EPOCHS_PER_LAYER = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.01
GOODNESS_THRESHOLD = 2.0

random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

device = torch.device("cpu")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path.home() / "datasets"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def add_label(images, labels):
    """Encode a candidate digit label in the first ten input pixels."""
    x = images.reshape(images.shape[0], -1).clone()
    x[:, :10] = 0.0
    rows = torch.arange(labels.shape[0])
    x[rows, labels] = 1.0
    return x


def normalize_between_layers(activations):
    """Normalize activations before passing them to the next layer."""
    return F.normalize(activations, p=2, dim=1, eps=1e-6)


class ForwardForwardNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Two local-learning layers, with comparable parameter count to BP.
        self.layers = nn.ModuleList([
            nn.Linear(784, 128),
            nn.Linear(128, 128),
        ])

    def predict(self, images, number_of_trained_layers):
        """Classify by testing all ten candidate labels."""
        candidate_scores = []

        for candidate in range(10):
            labels = torch.full(
                (images.shape[0],),
                candidate,
                dtype=torch.long,
            )

            x = add_label(images, labels)
            total_goodness = torch.zeros(images.shape[0])

            for layer in self.layers[:number_of_trained_layers]:
                activations = F.relu(layer(x))

                # Goodness: sum of squared neuron activities.
                total_goodness += activations.square().sum(dim=1)

                x = normalize_between_layers(activations)

            candidate_scores.append(total_goodness)

        scores = torch.stack(candidate_scores, dim=1)
        return scores.argmax(dim=1)


def make_layer_inputs(images, labels, layer_index, model, negative_generator):
    """Construct positive/negative inputs without training earlier layers."""
    negative_offsets = torch.randint(
        1,
        10,
        labels.shape,
        generator=negative_generator,
    )
    negative_labels = (labels + negative_offsets) % 10

    positive = add_label(images, labels)
    negative = add_label(images, negative_labels)

    # Earlier layers are already trained and remain frozen.
    with torch.no_grad():
        for previous_layer in model.layers[:layer_index]:
            positive = normalize_between_layers(
                F.relu(previous_layer(positive))
            )
            negative = normalize_between_layers(
                F.relu(previous_layer(negative))
            )

    return positive, negative


def evaluate(model, test_loader, number_of_trained_layers):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model.predict(
                images,
                number_of_trained_layers,
            )
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# Load the same MNIST dataset and deterministic subsets used by BP.
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform,
)

subset_generator = torch.Generator().manual_seed(SEED)

train_indices = torch.randperm(
    len(train_dataset),
    generator=subset_generator,
)[:5000].tolist()

test_indices = torch.randperm(
    len(test_dataset),
    generator=subset_generator,
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

model = ForwardForwardNet().to(device)
negative_generator = torch.Generator().manual_seed(SEED + 1000)

print("Experiment: Plaintext FF smoke test")
print("Device:", device)
print("Training samples:", len(train_subset))
print("Test samples:", len(test_subset))
print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters()),
)

results = []

# FF trains each layer using its own local goodness objective.
for layer_index, layer in enumerate(model.layers):
    print(f"\nTraining FF layer {layer_index + 1}/{len(model.layers)}")

    optimizer = torch.optim.SGD(
        layer.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(1, EPOCHS_PER_LAYER + 1):
        epoch_start = time.perf_counter()
        model.train()

        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            positive, negative = make_layer_inputs(
                images,
                labels,
                layer_index,
                model,
                negative_generator,
            )

            positive_activations = F.relu(layer(positive))
            negative_activations = F.relu(layer(negative))

            positive_goodness = positive_activations.square().sum(dim=1)
            negative_goodness = negative_activations.square().sum(dim=1)

            # Positive examples should have high goodness;
            # negative examples should have low goodness.
            loss = (
                F.softplus(
                    GOODNESS_THRESHOLD - positive_goodness
                ).mean()
                +
                F.softplus(
                    negative_goodness - GOODNESS_THRESHOLD
                ).mean()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_seconds = time.perf_counter() - epoch_start
        mean_local_loss = total_loss / total_samples

        eval_start = time.perf_counter()
        accuracy = evaluate(
            model,
            test_loader,
            number_of_trained_layers=layer_index + 1,
        )
        inference_seconds = time.perf_counter() - eval_start

        row = {
            "seed": SEED,
            "layer": layer_index + 1,
            "epoch_within_layer": epoch,
            "trained_layers": layer_index + 1,
            "threshold": GOODNESS_THRESHOLD,
            "local_loss": mean_local_loss,
            "test_accuracy_pct": accuracy,
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
        }
        results.append(row)

        print(
            f"Layer {layer_index + 1}, "
            f"Epoch {epoch}/{EPOCHS_PER_LAYER} | "
            f"Local loss: {mean_local_loss:.4f} | "
            f"Test accuracy: {accuracy:.2f}% | "
            f"Train time: {train_seconds:.2f}s"
        )

results_file = RESULTS_DIR / "ff_mnist_smoke.csv"

with results_file.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to:", results_file)
