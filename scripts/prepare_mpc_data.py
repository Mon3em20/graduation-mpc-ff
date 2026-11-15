"""Prepare deterministic, stratified MNIST inputs for bounded MPC jobs."""
import argparse
import json
import random
import sys
from pathlib import Path

REFERENCE_ROOT = Path(__file__).resolve().parent / "reference"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))
from bp_training_reference import (  # noqa: E402
    DATASET_SEED, INITIALIZATION_SEED, ORDER_SEED, initialize, load_mnist,
    parameter_hash, parameter_payload, party_training_indices,
)
from secure_bp_input import input_hashes, write_values


def load_records(seed):
    """Select 3 training, validation, and test records per class per party."""
    train_images, train_labels, test_images, test_labels = load_mnist()
    train_by_class = {label: [] for label in range(10)}
    test_by_class = {label: [] for label in range(10)}
    for index, label in enumerate(train_labels):
        train_by_class[label].append(index)
    for index, label in enumerate(test_labels):
        test_by_class[label].append(index)
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
        for split, selected, source in (
            ("train", train_indices, "train"),
            ("validation", validation_indices, "train"),
            ("test", test_indices, "test"),
        ):
            indices[split].extend(selected)
            for index in selected:
                image = (train_images if source == "train" else test_images)[index]
                actual_label = (train_labels if source == "train" else test_labels)[index]
                train_rows_for_split = (train_rows if split == "train" else
                                        validation_rows if split == "validation" else test_rows)
                train_rows_for_split.append(
                    (image, int(actual_label)))
    # Selection remains class-major so the selected indices are stable; the
    # comparison path orders each party's records independently below.
    return train_rows, validation_rows, test_rows, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DATASET_SEED)
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED)
    parser.add_argument("--initialization-seed", type=int,
                        default=INITIALIZATION_SEED)
    parser.add_argument("--comparison-id", default=None)
    parser.add_argument("--shuffle-training", action="store_true")
    parser.add_argument("--initialization-method",
                        choices=("deterministic", "he_xavier"),
                        default="deterministic")
    args = parser.parse_args()
    train, validation, test, indices = load_records(args.seed)
    selected = {"train": indices["train"], "validation": indices["validation"],
                "test": indices["test"]}
    if args.shuffle_training:
        train_order = [
            party_training_indices(selected, args.order_seed, epoch)
            for epoch in range(3)
        ]
    else:
        train_order = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    streams = [(args.output_dir / f"Input-P{party}-0").open("w")
               for party in range(3)]
    try:
        # Text input: every scalar is consumed in program order by input_from.
        parameters = initialize(args.initialization_method,
                                args.initialization_seed)
        write_values(streams, 0, parameters[0])
        write_values(streams, 0, parameters[1])
        write_values(streams, 0, parameters[2])
        write_values(streams, 0, parameters[3])

        def emit_training(rows, orders):
            by_index = {index: row for index, row in zip(indices["train"], rows)}
            for epoch_order in orders or [
                    {party: [indices["train"][sample]
                             for sample in range(party, len(rows), 3)]
                     for party in range(3)}] * 3:
                for batch in range(30):
                    for owner in range(3):
                        index = epoch_order[owner][batch]
                        features, label = by_index[index]
                        write_values(streams, owner, features)
                        write_values(streams, owner,
                                    [float(j == label) for j in range(10)])

        emit_training(train, train_order)

        def emit_evaluation(rows):
            for sample, (features, label) in enumerate(rows):
                owner = sample % 3
                write_values(streams, owner, features)
                write_values(streams, owner,
                            [float(j == label) for j in range(10)])

        emit_evaluation(validation)
        emit_evaluation(test)
    finally:
        for stream in streams:
            stream.close()
    hashes = input_hashes(args.output_dir)
    metadata = {
        "experiment_id": args.comparison_id or "historical-compatible",
        "configuration": ("configs/bp_comparison.yaml"
                          if args.comparison_id else "configs/bp_mpc.yaml"),
        "dataset_selection_seed": args.seed,
        "batch_order_seed": args.order_seed if args.shuffle_training else None,
        "initialization_seed": args.initialization_seed,
        "initialization_method": args.initialization_method,
        "initial_parameter_sha256": parameter_hash(
            parameters,
            args.initialization_method,
            args.initialization_seed),
        "seed": args.seed, "ownership": "sample index modulo 3",
        "input_format": "MP-SPDZ text Input-P<party>-0; flattened scalar order",
        "model_input_owner": 0, "parties": 3, "parameter_count": 6370,
        "counts": {"train": 90, "validation": 30, "test": 30},
        "per_party_counts": {"train": [30, 30, 30],
                             "validation": [10, 10, 10],
                             "test": [10, 10, 10]},
        "indices": indices, "training_order_by_party_epoch": train_order,
        "sha256": hashes,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "initial_parameters.json").write_text(
        json.dumps({
            "experiment_id": args.comparison_id or "historical-compatible",
            **parameter_payload(parameters,
                                args.initialization_method,
                                args.initialization_seed),
            "sha256": metadata["initial_parameter_sha256"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
