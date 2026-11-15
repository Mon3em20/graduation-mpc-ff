"""Render the shared secure BP template for a compile-time profile."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "mpc/programs/bp/bp_secure_template.mpc.in"
PROFILES = {
    "bounded": {
        "output": "bp_train_refactored.mpc",
        "layer_sizes": [784, 8, 10],
        "counts": [90, 30, 30],
        "batch_sizes": [3] * 30,
        "epochs": 3,
        "learning_rate": 0.05,
    },
    "matched": {
        "output": "bp_matched.mpc",
        "layer_sizes": [784, 128, 128, 10],
        "counts": [5000, 1000, 1000],
        "batch_sizes": [128] * 39 + [8],
        "epochs": 5,
        "learning_rate": 0.05,
    },
    "diagnostic": {
        "output": "bp_matched_diagnostic.mpc",
        "layer_sizes": [784, 128, 128, 10],
        "counts": [9, 3, 3],
        "batch_sizes": [3, 3, 3],
        "epochs": 1,
        "learning_rate": 0.05,
    },
}


def render(profile):
    values = PROFILES[profile]
    source = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "__LAYER_SIZES__": repr(values["layer_sizes"]),
        "__COUNTS__": repr(values["counts"]),
        "__BATCH_SIZES__": repr(values["batch_sizes"]),
        "__EPOCHS__": str(values["epochs"]),
        "__LEARNING_RATE__": repr(values["learning_rate"]),
        "__MAX_BATCH__": str(max(values["batch_sizes"])),
    }
    for marker, replacement in replacements.items():
        source = source.replace(marker, replacement)
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "mpc/programs/bp" / PROFILES[args.profile]["output"]
    output.write_text(render(args.profile), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
