# SECURE-BP-MATCHED-v1

## Status

The matched implementation is added as a distinct experiment and is not a
replacement for the bounded Stage B programs or their results. Its
compile-time source is rendered from the shared implementation described in
[`secure_bp_refactoring.md`](secure_bp_refactoring.md). Full MPC execution is
resource-dependent and has not been claimed here without a measured run.

| Area | Status |
|---|---|
| Implementation | Complete |
| Unit tests | Passed when `tests/test_bp_matched.py` passes |
| Numerical verification | Partial: independent scalar/reference checks are present; MPC aggregate verification requires execution |
| Full dataset execution | Not run by this change |
| Five-seed evaluation | Not run |
| Timing repetitions | Not run |
| Communication measurement | Not run |
| Ready for Secure FF comparison | No |

## Baseline match

The authoritative plaintext baseline uses MNIST 5,000/1,000/1,000 splits,
`784-128-128-10`, ReLU, cross-entropy, SGD at `0.05`, batch size 128, and
five epochs. The matched program uses the same dimensions, parameter count
`118,282`, split helper, PyTorch initialization policy, and continued
`torch.randperm` ordering policy. Its final eight-sample batch is averaged by
eight, matching minibatch-mean loss semantics.

## MPC design

The experiment preserves MP-SPDZ v0.4.4 commit
`113c4e47dd94bca9ecab029d2f0abe649bea5c52`, three-party semi-honest
honest-majority Rep3 Ring, 64-bit ring, and `sfix(16,31)`. Parameters are
input by party 0. Each training or evaluation sample is input by
`global_position % 3`; input files are generated in the exact consumption
order. Features, labels, parameters, activations, gradients, and updates
remain secret. Only per-epoch training loss and aggregate validation/test
loss and correct counts are opened.

The implementation uses affine layers, strict-positive ReLU derivatives,
stable softmax cross-entropy, explicit gradient accumulation, minibatch
averaging, and persistent parameters across all epochs. It does not claim
formal overflow safety or stronger security than the selected protocol.

## Reproduction

```bash
python scripts/prepare_secure_bp_matched.py \
  --output-dir /path/to/MP-SPDZ/Player-Data \
  --seed 42 --initialization-seed 42
MP_SPDZ_ROOT=/path/to/MP-SPDZ \
  scripts/run_mpc.sh bp_matched.mpc results/mpc/bp/matched_v1_seed42.log
```

The generated `metadata.json`, `parameters.json`, input hashes, compiler
output, runtime output, and communication report must be preserved under a
new run directory. Repeat with utility seeds 43--46 and three timing
repetitions only after correctness and resource feasibility are established.

## Limitations and remaining work

This source has not fabricated runtime, communication, numerical-agreement, or
five-seed results. A complete study still requires an independent FP64
full-training reference, fixed-point agreement tolerances based on observed
outputs, MP-SPDZ compilation/execution, resource measurements, and an
aggregate report. The old Stage B files and `BP-COMP-001` artifacts remain
historical evidence and are not reinterpreted as this experiment.
