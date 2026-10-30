"""Reproducible plaintext MNIST BP baseline."""
import csv
import os
import statistics
import sys
import random
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / "datasets"
CANONICAL_OUT = ROOT / "results" / "baselines" / "bp.csv"
OUT = CANONICAL_OUT
SEEDS = (42, 43, 44, 45, 46)
TIMING_REPETITIONS = 3
EPOCHS, BATCH_SIZE, LR = 5, 128, 0.05
TRAIN_N, VALID_N, TEST_N = 5000, 1000, 1000
OLD_TEST_N = 1000
torch.set_num_threads(4)


def parse_output_path(argv):
    if "--output" not in argv:
        return CANONICAL_OUT
    index = argv.index("--output")
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ValueError("--output requires a CSV path")
    return Path(argv[index + 1]).expanduser()


def write_csv_atomic(path, rows):
    fields = sorted({key for row in rows for key in row})
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


BASELINE_RECORD_TYPES = {"epoch", "seed_summary", "aggregate"}


def datasets_and_indices():
    train = datasets.MNIST(DATA, train=True, download=False, transform=transforms.ToTensor())
    test = datasets.MNIST(DATA, train=False, download=False, transform=transforms.ToTensor())
    # Recreate the legacy split RNG sequence: train permutation, then official-test permutation.
    legacy_rng = torch.Generator().manual_seed(42)
    train_perm = torch.randperm(len(train), generator=legacy_rng)
    old_test_perm = torch.randperm(len(test), generator=legacy_rng)
    fresh_test = old_test_perm[OLD_TEST_N:OLD_TEST_N + TEST_N].tolist()
    tr = train_perm[:TRAIN_N].tolist()
    va = train_perm[TRAIN_N:TRAIN_N + VALID_N].tolist()
    assert not set(tr) & set(va)
    return train, test, tr, va, fresh_test


def make_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))


def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss_sum += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            n += y.numel()
    return loss_sum / n, 100.0 * correct / n


def run(train_ds, test_ds, tr, va, te, seed, repetition,
        epochs=EPOCHS, batch_size=BATCH_SIZE):
    random.seed(seed)
    torch.manual_seed(seed)
    model = make_model()
    train_loader = DataLoader(Subset(train_ds, tr), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(Subset(train_ds, va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(test_ds, te), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_epoch, best_state = -1.0, 0, None
    epoch_rows = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * y.size(0)
            count += y.size(0)
        train_loss = total / count
        val_loss, val_acc = evaluate(model, val_loader)
        elapsed = time.perf_counter() - start
        selected = val_acc > best_acc
        if selected:
            best_acc, best_epoch = val_acc, epoch
            best_state = deepcopy(model.state_dict())
        epoch_rows.append(dict(record_type="epoch", seed=seed, repetition=repetition,
            epoch=epoch, train_loss=train_loss, validation_loss=val_loss,
            validation_accuracy_pct=val_acc, epoch_seconds=elapsed,
            selected_best_epoch=selected))
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader)
    summary = dict(record_type="seed_summary", seed=seed, repetition=repetition,
        selected_epoch=best_epoch, best_validation_accuracy_pct=best_acc,
        final_test_loss=test_loss, final_test_accuracy_pct=test_acc,
        training_seconds=sum(r["epoch_seconds"] for r in epoch_rows),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        train_samples=len(tr), validation_samples=len(va), test_samples=len(te))
    return epoch_rows, summary


def reference_indices():
    """Hinton-style full-data reference split, excluding legacy-exposed test indices."""
    rng = torch.Generator().manual_seed(42)
    train_perm = torch.randperm(60000, generator=rng)
    test_perm = torch.randperm(10000, generator=rng)
    tr = train_perm[:50000].tolist()
    va = train_perm[50000:].tolist()
    te = test_perm[OLD_TEST_N:].tolist()
    return tr, va, te


def run_reference_check(output=None):
    """Append a small, source-aligned full-MNIST BP check to canonical evidence."""
    output = output or CANONICAL_OUT
    if not output.is_file():
        raise FileNotFoundError(f"Canonical BP CSV is required: {output}")
    with output.open(newline="") as f:
        existing = list(csv.DictReader(f))
    if any(r.get("reference_id") == "hinton_bp_audit_v2" for r in existing):
        raise RuntimeError("BP reference records already exist in the canonical CSV")
    train, test, _, _, _ = datasets_and_indices()
    tr, va, te = reference_indices()
    reference_seeds = (42, 43, 44)
    rows, summaries = [], []
    for seed in reference_seeds:
        epochs, summary = run(train, test, tr, va, te, seed, 1,
                              epochs=20, batch_size=BATCH_SIZE)
        for row in epochs:
            row.update(record_type="reference_bp_epoch", reference_id="hinton_bp_audit_v2",
                       architecture="784-128-128-10 ReLU", optimizer="SGD",
                       learning_rate=LR, batch_size=BATCH_SIZE, epochs=20,
                       train_indices_seed=42, test_indices="official test permutation positions 1000:10000")
            rows.append(row)
        summary.update(record_type="reference_bp_seed_summary",
                       reference_id="hinton_bp_audit_v2",
                       architecture="784-128-128-10 ReLU", optimizer="SGD",
                       learning_rate=LR, batch_size=BATCH_SIZE, epochs=20,
                       train_indices_seed=42, test_indices="official test permutation positions 1000:10000")
        summaries.append(summary)
        rows.append(summary)
        print(f"BP reference seed={seed}: val={summary['best_validation_accuracy_pct']:.2f}% "
              f"test={summary['final_test_accuracy_pct']:.2f}% "
              f"train={summary['training_seconds']:.2f}s")
    values = [r["final_test_accuracy_pct"] for r in summaries]
    times = [r["training_seconds"] for r in summaries]
    rows.append(dict(record_type="reference_bp_aggregate", reference_id="hinton_bp_audit_v2",
        final_test_accuracy_mean_pct=sum(values) / len(values),
        final_test_accuracy_sd_pct=statistics.stdev(values),
        final_test_accuracy_min_pct=min(values), final_test_accuracy_max_pct=max(values),
        training_seconds_mean=sum(times) / len(times),
        training_seconds_sd=statistics.stdev(times),
        training_seconds_min=min(times), training_seconds_max=max(times),
        seeds="42-44", train_samples=len(tr), validation_samples=len(va), test_samples=len(te),
        reference_note="full MNIST split; 784-128-128-10 ReLU; 20 epochs; test excludes legacy-exposed 1,000"))
    all_rows = existing + rows
    fields = sorted({k for row in all_rows for k in row})
    write_csv_atomic(output, all_rows)
    print(f"Appended {len(rows)} BP reference records to {output}; preserved {len(existing)} prior rows")


def main():
    output = parse_output_path(sys.argv)
    if "--reference-check" in sys.argv:
        run_reference_check(output)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if output.is_file():
        with output.open(newline="") as f:
            existing_rows = list(csv.DictReader(f))
    train, test, tr, va, te = datasets_and_indices()
    assert len(set(te)) == TEST_N
    rows, summaries = [], []
    for repetition in range(1, TIMING_REPETITIONS + 1):
        for seed in SEEDS:
            epochs, summary = run(train, test, tr, va, te, seed, repetition)
            rows.extend(epochs)
            summaries.append(summary)
            print(f"BP rep={repetition} seed={seed}: val={summary['best_validation_accuracy_pct']:.2f}% "
                  f"test={summary['final_test_accuracy_pct']:.2f}% train={summary['training_seconds']:.2f}s")
    values = [next(r["final_test_accuracy_pct"] for r in summaries if r["seed"] == seed)
              for seed in SEEDS]
    times = [r["training_seconds"] for r in summaries]
    rows.extend(summaries)
    rows.append(dict(record_type="aggregate", seed="all", repetition="all",
        final_test_accuracy_pct=sum(values) / len(values),
        test_accuracy_std_pct=torch.tensor(values).std(unbiased=True).item(),
        test_accuracy_min_pct=min(values), test_accuracy_max_pct=max(values),
        training_seconds_mean=sum(times) / len(times),
        training_seconds_std=torch.tensor(times).std(unbiased=True).item(),
        training_seconds_min=min(times), training_seconds_max=max(times),
        repetitions=TIMING_REPETITIONS, seeds="42-46"))
    preserved_historical = [
        row for row in existing_rows
        if row.get("record_type") not in BASELINE_RECORD_TYPES
    ]
    fields = sorted({k for row in rows + preserved_historical for k in row})
    fields = sorted(set(fields) | {"architecture", "optimizer", "learning_rate", "batch_size",
                                   "epochs", "train_indices_seed", "legacy_test_excluded",
                                   "final_test_indices"})
    for row in rows:
        row.update(architecture="784-128-128-10 ReLU", optimizer="SGD",
                   learning_rate=LR, batch_size=BATCH_SIZE, epochs=EPOCHS,
                   train_indices_seed=42, legacy_test_excluded=OLD_TEST_N,
                   final_test_indices="official test permutation positions 1000:2000")
    write_csv_atomic(output, rows + preserved_historical)
    print(f"Wrote {len(rows)} rows to {output}; preserved {len(preserved_historical)} historical rows")


if __name__ == "__main__":
    main()
