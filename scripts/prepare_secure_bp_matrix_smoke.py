"""Inputs + Python reference numbers for the reduced matrix smoke profile.

Uses the first 264 positions of the exact seed-42 epoch-1 DataLoader order
(2 batches of 128 + 1 of 8), the first 100 validation and first 100 test
samples, and the same seed-42 initialization / [in,out] weight layout as the
full experiment.  Writes only into a new, empty directory outside Player-Data.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "verify"))
sys.path.insert(0, str(ROOT / "src"))
import matrix_bp_reference as R  # noqa: E402
import prepare_secure_bp_matched as P  # noqa: E402
from baselines.bp import datasets_and_indices  # noqa: E402
from run_matrix_reference import arrays  # noqa: E402
from secure_bp_input import input_hashes  # noqa: E402

TRAIN_N, VAL_N, TEST_N = 264, 100, 100


def smoke_plan(mnist_orders, validation, test):
    train = [(g % 3, "train", index) for g, index in enumerate(mnist_orders[0][:TRAIN_N])]
    val = [(s % 3, "validation", index) for s, index in enumerate(validation[:VAL_N])]
    tst = [(s % 3, "test", index) for s, index in enumerate(test[:TEST_N])]
    return train + val + tst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if "Player-Data" in output.parts:
        raise SystemExit("Refusing to write inside Player-Data")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to use non-empty directory: {output}")
    train_ds, test_ds, tr, va, te = datasets_and_indices()
    _, mnist_orders = P.training_orders(tr, 42)
    plan = smoke_plan(mnist_orders, va, te)
    params = P.serialized_parameters(42)
    output.mkdir(parents=True, exist_ok=True)
    streams = [(output / f"Input-P{p}-0").open("w") for p in range(3)]
    try:
        P.write_inputs(streams, params, plan, train_ds, test_ds)
    finally:
        for stream in streams:
            stream.close()
    images, labels = arrays(train_ds, test_ds)
    kwargs = dict(epochs=1, full_batches=2, final_batch=8,
                  validation_n=VAL_N, test_n=TEST_N)
    expected = {"program": "bp_matched_matrix_smoke.mpc",
                "note": "Python reference values; MPC output should agree within "
                        "fixed-point noise (see docs/reports/secure_bp_matched_matrix_v1.md)"}
    flo = R.simulate(params, plan, images, labels, "float", **kwargs)
    expected["float64"] = {"epoch1_train_loss": flo["epoch_losses"][0],
                           "validation_loss_correct": flo["validation"],
                           "test_loss_correct": flo["test"]}
    expected["fixed_point_emulation"] = []
    for seed in (0, 1, 2):
        fx = R.simulate(params, plan, images, labels, "fixed", seed=seed, **kwargs)
        expected["fixed_point_emulation"].append(
            {"rounding_seed": seed, "epoch1_train_loss": fx["epoch_losses"][0],
             "validation_loss_correct": fx["validation"],
             "test_loss_correct": fx["test"]})
    counts = [118282, 0, 0]
    for owner, _, _ in plan:
        counts[owner] += 794
    metadata = {"experiment_id": "SECURE-BP-MATCHED-MATRIX-v1-SMOKE",
                "plan_samples": len(plan), "input_token_counts": counts,
                "sha256": input_hashes(output),
                "weight_layout": "[input, output] row-major",
                "order_source": "first 264 positions of seed-42 DataLoader epoch 1"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output / "expected_python_reference.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(json.dumps({"metadata": metadata, "expected": expected}, indent=2))


if __name__ == "__main__":
    main()
