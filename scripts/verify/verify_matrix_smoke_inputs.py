"""Read-only verification of the SECURE-BP-MATCHED-MATRIX-v1 smoke input bundle.

Parses the three input streams token by token in the order bp_matched_matrix_smoke.mpc
consumes them and checks parameters, ownership, batch boundaries, sample identity
against an independently built baseline DataLoader, token counts and hashes.
Never writes into the bundle.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / sub))
from baselines.bp import datasets_and_indices  # noqa: E402
import prepare_secure_bp_matched as P  # noqa: E402

BATCHES = [(0, 128), (128, 128), (256, 8)]  # (start, rows) exactly as the smoke program
EVAL = 100


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, help="write JSON summary here (not in bundle)")
    args = parser.parse_args()
    b = args.bundle
    train_ds, test_ds, tr, va, te = datasets_and_indices()
    reference = P.baseline_style_orders(tr, 42)[0]
    tokens = [(b / f"Input-P{p}-0").read_text().split() for p in range(3)]
    cur = [0, 0, 0]

    def take(owner, n):
        v = [float(t) for t in tokens[owner][cur[owner]:cur[owner] + n]]
        assert len(v) == n, f"party {owner} ran out of tokens"
        cur[owner] += n
        return v

    checks, per_party = {}, [{"train": 0, "validation": 0, "test": 0}, {"train": 0, "validation": 0, "test": 0}, {"train": 0, "validation": 0, "test": 0}]
    torch_init = P.serialized_parameters(42)
    ok = True
    for values in torch_init:
        ok &= take(0, len(values)) == values
    checks["parameters_equal_seed42_transposed_layout"] = ok
    checks["parameter_tokens"] = sum(len(v) for v in torch_init) == 118282

    def sample(owner, ds, index):
        image, label = ds[index]
        x, y = take(owner, 784), take(owner, 10)
        return x == image.flatten().tolist() and y == [float(i == label) for i in range(10)]

    train_ok = own_ok = True
    pos = 0
    for start, rows in BATCHES:
        for s in range(rows):
            owner = (start + s) % 3          # formula in load_rows()
            train_ok &= sample(owner, train_ds, reference[start + s])
            per_party[owner]["train"] += 1
            own_ok &= owner == (pos % 3)
            pos += 1
    checks["train_samples_match_baseline_epoch1_prefix"] = train_ok
    checks["train_ownership_and_batch_boundaries"] = own_ok and pos == 264
    for split, ds, idx in (("validation", train_ds, va), ("test", test_ds, te)):
        good = True
        for s in range(EVAL):
            owner = (0 + s) % 3              # chunk 0 start 0, evaluation chunk of 100
            good &= sample(owner, ds, idx[s])
            per_party[owner][split] += 1
        checks[f"{split}_first100_order_and_ownership"] = good
    checks["all_tokens_consumed"] = cur == [len(t) for t in tokens]
    counts = [len(t) for t in tokens]
    checks["token_counts"] = counts == [242146, 122276, 122276]
    meta = json.loads((b / "metadata.json").read_text())
    checks["metadata_hashes_match_files"] = all(sha256(b / k) == v for k, v in meta["sha256"].items())
    checks["metadata_counts_match"] = meta["input_token_counts"] == counts
    summary = {"bundle": str(b), "token_counts": counts, "samples_per_party": per_party,
               "sha256": {f"Input-P{p}-0": sha256(b / f"Input-P{p}-0") for p in range(3)},
               "eval_first_validation_indices": va[:3], "eval_last_validation_index": va[EVAL - 1],
               "eval_first_test_indices": te[:3], "eval_last_test_index": te[EVAL - 1],
               "train_first_indices": reference[:3], "train_last_index": reference[263],
               "checks": checks, "all_pass": all(checks.values())}
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
