"""Compare the matrix-program reference (float64 and fixed-point emulation) with the
plaintext PyTorch baseline training loop.  Pure Python; no MPC is executed."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "verify"))
import matrix_bp_reference as R  # noqa: E402
import prepare_secure_bp_matched as P  # noqa: E402
from baselines.bp import datasets_and_indices, make_model  # noqa: E402


def arrays(train_ds, test_ds):
    def flat(ds):
        return (ds.data.float() / 255).reshape(-1, 784).double().numpy()
    train = flat(train_ds)
    return ({"train": train, "validation": train, "test": flat(test_ds)},
            {"train": train_ds.targets.numpy(), "validation": train_ds.targets.numpy(),
             "test": test_ds.targets.numpy()})


def torch_baseline(train_ds, test_ds, tr, va, te, dtype=torch.float64, seed=42):
    """bp.py's loop (same DataLoader/optimizer), optionally in float64."""
    torch.manual_seed(seed)
    model = make_model().to(dtype)
    loader = DataLoader(Subset(train_ds, tr), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    crit = torch.nn.CrossEntropyLoss()
    losses, snaps = [], []
    for _ in range(5):
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x.to(dtype)), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        losses.append(total / 5000)
        snaps.append([p.detach().clone() for p in model.parameters()])

    def ev(ds, idx):
        ld = DataLoader(Subset(ds, idx), batch_size=100, shuffle=False)
        ls, ok, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in ld:
                z = model(x.to(dtype))
                ls += torch.nn.functional.cross_entropy(z, y, reduction="sum").item()
                ok += (z.argmax(1) == y).sum().item()
                n += len(y)
        return ls / n, ok
    return losses, snaps, ev(train_ds, va), ev(test_ds, te)


def to_serialized(snapshot_tensors):
    out = []
    for i in range(3):
        out.append(snapshot_tensors[2 * i].t().contiguous().flatten().numpy())
        out.append(snapshot_tensors[2 * i + 1].numpy())
    return out


def maxdiff(sim_snapshot, torch_snapshot):
    W, b = sim_snapshot
    ref = to_serialized(torch_snapshot)
    worst = 0.0
    for layer in range(3):
        worst = max(worst, np.abs(W[layer].flatten() - ref[2 * layer]).max(),
                    np.abs(b[layer] - ref[2 * layer + 1]).max())
    return float(worst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    train_ds, test_ds, tr, va, te = datasets_and_indices()
    _, mnist_orders = P.training_orders(tr, 42)
    plan = list(P.input_plan(mnist_orders, va, te))
    images, labels = arrays(train_ds, test_ds)
    params = P.serialized_parameters(42)
    start = time.time()
    t_loss, t_snaps, t_val, t_test = torch_baseline(train_ds, test_ds, tr, va, te)
    f32 = torch_baseline(train_ds, test_ds, tr, va, te, dtype=torch.float32)
    report = {"torch_float64": {"epoch_losses": t_loss, "validation": t_val, "test": t_test},
              "torch_float32_baseline": {"epoch_losses": f32[0], "validation": f32[2],
                                         "test": f32[3]}}
    sim = R.simulate(params, plan, images, labels, "float")
    report["matrix_float64"] = {
        "epoch_losses": sim["epoch_losses"], "validation": sim["validation"],
        "test": sim["test"], "fully_consumed": sim["fully_consumed"],
        "consumed_samples_per_party": sim["consumed"],
        "max_param_diff_vs_torch64_per_epoch": [
            maxdiff(sim["snapshots"][e], t_snaps[e]) for e in range(5)]}
    fixed_runs = []
    for seed in args.fixed_seeds:
        fx = R.simulate(params, plan, images, labels, "fixed", seed=seed)
        fixed_runs.append({
            "rounding_seed": seed, "epoch_losses": fx["epoch_losses"],
            "validation": fx["validation"], "test": fx["test"],
            "max_param_diff_vs_float64_per_epoch": [
                float(max(max(np.abs(fx["snapshots"][e][0][l] - sim["snapshots"][e][0][l]).max(),
                              np.abs(fx["snapshots"][e][1][l] - sim["snapshots"][e][1][l]).max())
                          for l in range(3))) for e in range(5)]})
    report["fixed_point_emulation"] = fixed_runs
    report["seconds"] = time.time() - start
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
