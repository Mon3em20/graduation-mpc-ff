"""Generate immutable inputs for SECURE-BP-MATCHED-v1."""
import argparse
import json
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(ROOT / "src"))
from baselines.bp import datasets_and_indices, make_model
from secure_bp_input import input_hashes, write_values

EXPERIMENT_ID = "SECURE-BP-MATCHED-v1"
ARCHITECTURE = "784-128-128-10"
PARTIES = 3
BATCH_SIZES = [128] * 39 + [8]


def parameters(seed):
    torch.manual_seed(seed)
    model = make_model()
    layers = [model[1], model[3], model[5]]
    return ([value for value in layers[0].weight.detach().flatten().tolist()],
            layers[0].bias.detach().tolist(),
            [value for value in layers[1].weight.detach().flatten().tolist()],
            layers[1].bias.detach().tolist(),
            [value for value in layers[2].weight.detach().flatten().tolist()],
            layers[2].bias.detach().tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initialization-seed", type=int, default=42)
    args = parser.parse_args()
    train_ds, test_ds, train_indices, validation_indices, test_indices = (
        datasets_and_indices())
    params = parameters(args.initialization_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    streams = [(args.output_dir / f"Input-P{party}-0").open("w")
               for party in range(PARTIES)]
    try:
        for values in params:
            write_values(streams, 0, values)

        order_generator = torch.Generator().manual_seed(args.seed)
        for epoch in range(5):
            order = torch.randperm(len(train_indices), generator=order_generator).tolist()
            offset = 0
            for batch_size in BATCH_SIZES:
                for position in order[offset:offset + batch_size]:
                    owner = (offset + order[offset:offset + batch_size].index(position)) % PARTIES
                    image, label = train_ds[train_indices[position]]
                    write_values(streams, owner, image.flatten().tolist())
                    write_values(streams, owner, [float(index == label)
                                                  for index in range(10)])
                offset += batch_size

        for indices, dataset in ((validation_indices, train_ds),
                                 (test_indices, test_ds)):
            for sample, index in enumerate(indices):
                image, label = dataset[index]
                owner = sample % PARTIES
                write_values(streams, owner, image.flatten().tolist())
                write_values(streams, owner,
                             [float(i == label) for i in range(10)])
    finally:
        for stream in streams:
            stream.close()

    hashes = input_hashes(args.output_dir)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "parameter_count": 118282,
        "dataset": "MNIST",
        "split_policy": "src.baselines.bp.datasets_and_indices",
        "split_seed": 42,
        "utility_seed": args.seed,
        "initialization_seed": args.initialization_seed,
        "data_order": "torch.randperm generator, state continued across epochs",
        "batch_sizes": BATCH_SIZES,
        "epochs": 5,
        "learning_rate": 0.05,
        "ownership": "global training position modulo three; evaluation position modulo three",
        "protocol": "Rep3 Ring, semi-honest honest majority, 64-bit ring",
        "fixed_point": "sfix(16,31)",
        "sha256": hashes,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "parameters.json").write_text(
        json.dumps({"architecture": ARCHITECTURE, "seed": args.initialization_seed,
                    "parameters": params}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
