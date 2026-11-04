# MPC programs

MP-SPDZ is an external dependency, not a Python package in this repository.
Set `MP_SPDZ_ROOT` to a pinned MP-SPDZ v0.4.4 checkout at commit
`113c4e47dd94bca9ecab029d2f0abe649bea5c52`.

## Programs

- `programs/bp/bp_tiny.mpc` is a 2-2-1 secret BP correctness prototype. It
  deliberately reveals synthetic forward, loss, gradient, and update
  diagnostics for comparison with `scripts/verify/verify_bp.py`.
- `programs/bp/relu_edge.mpc` checks inactive and active ReLU behavior using
  synthetic inputs. Its revealed values are not private MNIST results.
- `programs/bp/bp_train.mpc` is the bounded end-to-end secure BP program. It
  trains a `784-8-10` model using stable softmax cross-entropy, strict
  derivative `(0 < pre[j])`, secure forward and backward passes, weight and
  bias gradients, SGD updates, and persistent parameters. It consumes three
  party input streams and reveals only aggregate validation/test losses and
  correct counts.

All programs use the three-party semi-honest honest-majority Rep3 Ring
protocol, a 64-bit ring, and `sfix(16,31)`. The full training run uses the
deterministic 90/30/30 MNIST subset, batch size 3, three epochs, and learning
rate `0.05`.

## Input and execution workflow

For `BP-COMP-001`, run `scripts/prepare_mpc_data.py` with an output directory,
`--shuffle-training`, `--initialization-method he_xavier`,
`--order-seed 42042`, and `--initialization-seed 42`. It requires MNIST under
`~/datasets/MNIST`, writes `Input-P0-0`, `Input-P1-0`, and `Input-P2-0`, and
records deterministic indices, epoch order, exact initial parameters, and
hashes in the manifest. Copy those files into the external MP-SPDZ
`Player-Data` directory before execution.

```bash
MP_SPDZ_ROOT="$HOME/MP-SPDZ" \
MP_SPDZ_PYTHON="$HOME/venvs/mp-spdz-ml/bin/python" \
./scripts/run_mpc.sh bp_train.mpc /path/to/unique.log
```

The runner copies only the requested authoritative source into
`Programs/Source`, compiles with `-M -E ring`, and records the selected log.
It does not prepare inputs or modify MP-SPDZ source or keys.

The preserved strict-ReLU evidence is
`results/mpc/bp/training/relu_strict_edge_audit.log`. Its exact diagnostic
source is not tracked in this repository, so that output is evidence of the
recorded execution but is not fully reproducible from repository files alone.
The newly tracked `programs/bp/relu_strict_edge.mpc` and
`scripts/verify/verify_relu_strict.py` reproduce the same public boundary
test without relabeling the historical log.

`BP-COMP-001` uses the same program and protocol with a separate input and
metadata directory under `results/mpc/bp/comparison_baseline/`. Its party
training records are deterministically shuffled by epoch and its exact
He/Xavier initial parameters are stored beside the inputs.
