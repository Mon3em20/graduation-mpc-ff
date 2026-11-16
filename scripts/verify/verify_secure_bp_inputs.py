"""Read-only verification of a generated SECURE-BP-MATCHED-v1 input bundle.

Parses Input-P{0,1,2}-0 token by token and checks them against an independently
constructed plaintext-baseline DataLoader, fresh PyTorch initialization and the
MPC consumption order.  Writes verification_summary.json into the bundle only
when --write-summary is given.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from baselines.bp import datasets_and_indices, make_model  # noqa: E402
import prepare_secure_bp_matched as P  # noqa: E402

EXPECTED_COUNTS = [7266664, 7146794, 7142824]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle
    checks = {}

    train_ds, test_ds, tr, va, te = datasets_and_indices()
    reference_orders = P.baseline_style_orders(tr, args.seed)
    metadata = json.loads((bundle / "metadata.json").read_text())

    tokens = [(bundle / f"Input-P{p}-0").read_text().split() for p in range(3)]
    counts = [len(t) for t in tokens]
    checks["token_counts"] = {"actual": counts, "expected": EXPECTED_COUNTS,
                              "pass": counts == EXPECTED_COUNTS}
    cursors = [0, 0, 0]

    def take(owner, n):
        values = [float(v) for v in tokens[owner][cursors[owner]:cursors[owner] + n]]
        cursors[owner] += n
        return values

    torch.manual_seed(42)
    model = make_model()
    shapes = [(784, 128), (128, 128), (128, 10)]
    init_ok = True
    for layer, (n_in, n_out) in enumerate(shapes):
        linear = model[1 + 2 * layer]
        weight = take(0, n_in * n_out)
        bias = take(0, n_out)
        want_w = linear.weight.detach().t().contiguous().flatten().tolist()
        init_ok &= weight == want_w and bias == linear.bias.detach().tolist()
    checks["initial_parameters_transposed_layout"] = init_ok

    order_ok = ownership_ok = sample_ok = True
    for epoch in range(5):
        seen = []
        for global_position, index in enumerate(reference_orders[epoch]):
            owner = global_position % 3
            image, label = train_ds[index]
            x, y = take(owner, 784), take(owner, 10)
            sample_ok &= (x == [float(v) for v in image.flatten().tolist()]
                          and y == [float(i == label) for i in range(10)])
            seen.append(index)
        order_ok &= sorted(seen) == sorted(tr) and len(set(seen)) == 5000
    checks["five_epoch_samples_match_baseline_order_and_owner"] = order_ok and sample_ok
    for split, indices, ds in (("validation", va, train_ds), ("test", te, test_ds)):
        ok = True
        for sample, index in enumerate(indices):
            owner = sample % 3
            image, label = ds[index]
            ok &= (take(owner, 784) == image.flatten().tolist()
                   and take(owner, 10) == [float(i == label) for i in range(10)])
        checks[f"{split}_order_and_ownership"] = ok
    checks["all_tokens_consumed"] = cursors == counts

    mnist_orders = reference_orders
    hashes = [P.order_hash(order) for order in mnist_orders]
    checks["epoch_hashes_match_metadata"] = (
        hashes == metadata["epoch_order_mnist_index_sha256"])
    checks["input_sha256_match_metadata"] = all(
        sha256(bundle / name) == digest for name, digest in metadata["sha256"].items())

    summary = {"bundle": str(bundle), "seed": args.seed,
               "epoch_order_mnist_index_sha256": hashes,
               "input_sha256": {f"Input-P{p}-0": sha256(bundle / f"Input-P{p}-0")
                                for p in range(3)},
               "checks": checks,
               "all_pass": all(v if isinstance(v, bool) else v["pass"]
                               for v in checks.values())}
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.write_summary:
        (bundle / "verification_summary.json").write_text(text + "\n")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
