# Tiny secure BP prototype

`bp_tiny.mpc` is an isolated one-example correctness prototype, not MNIST
training. It uses secret `sfix` inputs and weights, a 2-2-1 network, ReLU in
the hidden layer, linear output, half-squared-error loss, and one SGD update.
The hidden preactivations are 1.5 and 2.5, avoiding a ReLU boundary.

The independent checker is [`verify_bp.py`](../../scripts/verify/verify_bp.py). It checks the deliberately
revealed MPC diagnostics with tolerance `2e-4`, appropriate for the
`sfix.set_precision(16, 31)` configuration used here.

The only revealed values are forward values (`z1`, `h1`, `y_hat`, `loss`),
representative backward values (`d_w1`, `d_b1`, `d_w2`, `d_b2`), and updated
parameters. Inputs, weights, intermediate computations, gradients, and updates
are secret until those controlled diagnostics are revealed. This reveal is
for numerical validation only and is not a privacy-preserving training
workflow.

The verified local runner is:

```bash
env -u DEBUG PATH="$HOME/venvs/mp-spdz-ml/bin:$PATH" \
  Scripts/compile-run.py -M -E ring bp_tiny
```

`-M` preserves memory order as used by the existing Stage A Rep3 smoke tests;
`-E ring` selects the three-party semi-honest Rep3 ring setup.

The preserved `results/mpc/bp/bp_tiny_historical.log` records a historical compilation
of `stage_b_tiny_bp.mpc`, a source name that is not present in this repository.
Its contents therefore are not attributed to `mpc/programs/bp/bp_tiny.mpc`
without a source hash. The stored `bp_tiny.mpc` is the authoritative current
source; the log remains preserved as historical evidence.
