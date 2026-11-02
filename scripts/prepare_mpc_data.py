"""Prepare deterministic, stratified MNIST inputs for the bounded MPC job."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def load_records(seed):
    """Select 3 training, validation, and test records per class per party."""
    import torch
    from torchvision import datasets, transforms

    root = Path.home() / "datasets"
    train = datasets.MNIST(root, train=True, download=False,
                           transform=transforms.ToTensor())
    test = datasets.MNIST(root, train=False, download=False,
                          transform=transforms.ToTensor())
    train_by_class = {label: [] for label in range(10)}
    test_by_class = {label: [] for label in range(10)}
    for index, label in enumerate(train.targets.tolist()):
        train_by_class[int(label)].append(index)
    for index, label in enumerate(test.targets.tolist()):
        test_by_class[int(label)].append(index)
    train_rows, validation_rows, test_rows = [], [], []
    indices = {"train": [], "validation": [], "test": []}
    for label in range(10):
        rng = random.Random(seed + label)
        candidates = train_by_class[label][:]
        rng.shuffle(candidates)
        chosen = candidates[:12]
        train_indices = chosen[:9]
        validation_indices = chosen[9:12]
        test_candidates = test_by_class[label][:]
        random.Random(seed + 1000 + label).shuffle(test_candidates)
        test_indices = test_candidates[:3]
        for split, selected, dataset in (
            ("train", train_indices, train),
            ("validation", validation_indices, train),
            ("test", test_indices, test),
        ):
            indices[split].extend(selected)
            for index in selected:
                image, actual_label = dataset[index]
                train_rows_for_split = (train_rows if split == "train" else
                                        validation_rows if split == "validation" else test_rows)
                train_rows_for_split.append(
                    ([float(value) for value in image.reshape(-1)], int(actual_label)))
    # Class-major selection and three-way slices make ownership deterministic
    # and give every party three examples per class in each split.
    return train_rows, validation_rows, test_rows, indices


def write_tokens(streams, owner, values):
    streams[owner].write(" ".join(str(float(value)) for value in values) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train, validation, test, indices = load_records(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    streams = [(args.output_dir / f"Input-P{party}-0").open("w")
               for party in range(3)]
    try:
        # Text input: every scalar is consumed in program order by input_from.
        write_tokens(streams, 0, [((i - j) / 1000.0)
                                  for i in range(784) for j in range(8)])
        write_tokens(streams, 0, [0.0] * 8)
        write_tokens(streams, 0, [((i - j) / 1000.0)
                                  for i in range(8) for j in range(10)])
        write_tokens(streams, 0, [0.0] * 10)

        def emit(rows, repetitions=1):
            for _ in range(repetitions):
                for sample, (features, label) in enumerate(rows):
                    owner = sample % 3
                    write_tokens(streams, owner, features)
                    write_tokens(streams, owner,
                                [float(j == label) for j in range(10)])

        emit(train, 3)
        emit(validation)
        emit(test)
    finally:
        for stream in streams:
            stream.close()
    hashes = {}
    for party in range(3):
        path = args.output_dir / f"Input-P{party}-0"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "seed": args.seed, "ownership": "sample index modulo 3",
        "input_format": "MP-SPDZ text Input-P<party>-0; flattened scalar order",
        "model_input_owner": 0, "parties": 3, "parameter_count": 6370,
        "counts": {"train": 90, "validation": 30, "test": 30},
        "per_party_counts": {"train": [30, 30, 30],
                             "validation": [10, 10, 10],
                             "test": [10, 10, 10]},
        "indices": indices, "sha256": hashes,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
