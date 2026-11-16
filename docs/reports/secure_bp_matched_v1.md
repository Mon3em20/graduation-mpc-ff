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
`118,282`, split helper, PyTorch initialization policy, and the plaintext DataLoader
ordering (see Corrections below). Its final eight-sample batch is averaged by
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

## Corrections before full execution (2026-11-15)

1. **Training order.** The first generator drew `torch.randperm(5000)` once per
   epoch. The plaintext baseline's `DataLoader(shuffle=True, generator=seed)`
   additionally consumes the generator per epoch (a base-seed `random_()` draw,
   then `randperm(n)` plus an empty-remainder `randperm(n)` in
   `RandomSampler.__iter__`, torch 2.14.1), so the orders differed in
   4999-5000 of 5000 positions in every epoch. The generator now iterates one
   index-only `DataLoader` (same batch size, shuffle, generator seed,
   `num_workers=0`) across all five epochs and asserts exact equality with an
   independently built `DataLoader(Subset(...), ...)`.
2. **Weight orientation.** PyTorch stores `[out, in]`; the MPC declares
   `sfix.Matrix(in, out)` read row-major. The generator now writes all three
   weight matrices transposed to `[input, output]`; biases are unchanged and
   initial values/seed are unchanged.
3. **Hygiene.** The generator refuses non-empty output directories and any path
   under `Player-Data`.

The MPC program `bp_matched.mpc` is unchanged: the source audit found its
forward pass, strict-positive ReLU derivative, stable softmax/cross-entropy,
delta recursion, gradient accumulation, `/batch_size` averaging (final batch of
8 uses 8) and persistent parameters consistent with the plaintext baseline;
`tests/test_secure_bp_matched_inputs.py` mirrors that algorithm in float64 and
matches PyTorch SGD to 1e-9 for a full and a partial batch.

Evaluation note: the secure program reports validation/test metrics for the
epoch-5 model only. For seed 42 the plaintext baseline selects epoch 5 as its
best-validation checkpoint (val 76.6%, test 77.2%, epoch-5 train loss
0.9092), so the comparison is like-for-like for this seed only.

Seed-42 corrected bundle (outside Player-Data):
`~/secure-bp-runs/SECURE-BP-MATCHED-v1_seed42_corrected/` with per-party token
counts 7,266,664 / 7,146,794 / 7,142,824 (unchanged) and five epoch-order
hashes in `metadata.json`. Hash rule: SHA-256 over UTF-8 text of the epoch's
5,000 MNIST indices, each as a decimal followed by a newline. Verify with
`scripts/verify/verify_secure_bp_inputs.py`.

## Reproduction

```bash
python scripts/prepare_secure_bp_matched.py \
  --output-dir $HOME/secure-bp-runs/<new-empty-dir> \
  --seed 42 --initialization-seed 42
python scripts/verify/verify_secure_bp_inputs.py $HOME/secure-bp-runs/<new-empty-dir>
```

Copying inputs into MP-SPDZ `Player-Data` and running
`scripts/run_mpc.sh bp_matched.mpc` is a manual step. Preserve generated
metadata, input hashes, compiler output and runtime output under a new run
directory. Repeat with utility seeds 43--46 and timing repetitions only after
correctness and resource feasibility are established. Note that the
scalar-loop `bp_matched_diagnostic` (9 training samples) timed out after 600 s
in an earlier attempt, so full-size feasibility of this program is unproven.

## Scalar vs matrix implementation

`bp_matched.mpc` needs about 2.0e10 VM rounds and 6.6e9 truncations and is not
practically executable. A separate batched-matrix program with the same input
format is documented in [secure_bp_matched_matrix_v1.md](secure_bp_matched_matrix_v1.md),
including compile-only statistics, Python verification and the manual run
commands. The scalar program remains unchanged.

## Limitations and remaining work

This source has not fabricated runtime, communication, numerical-agreement, or
five-seed results. A complete study still requires an independent FP64
full-training reference, fixed-point agreement tolerances based on observed
outputs, MP-SPDZ compilation/execution, resource measurements, and an
aggregate report. The old Stage B files and `BP-COMP-001` artifacts remain
historical evidence and are not reinterpreted as this experiment.
