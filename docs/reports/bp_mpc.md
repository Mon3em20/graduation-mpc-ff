# Bounded secure BP

This artifact is a bounded secure backpropagation run: a `784-8-10` ReLU
network, three epochs, 30 SGD updates per epoch, and MNIST splits of 90
training, 30 validation, and 30 test examples. It uses MP-SPDZ v0.4.4's
three-party semi-honest Rep3 ring protocol and `sfix(16,31)` (fractional
precision 16, total precision 31).

`scripts/prepare_mpc_data.py` selects three training, three validation, and
three test images per MNIST class for each of the three parties (9/3/3
examples per class globally), using seeded per-class permutations. Samples are
class-major and assigned by sample index modulo three, giving each party
30/10/10 examples. It writes text
`Input-P<party>-0` files: flattened scalar values are consumed in program
order, with model initialization owned by party 0. Metadata records source
indices and SHA-256 hashes.

The output layer uses MP-SPDZ's stabilized `ml.softmax`; training uses the
corresponding stable log-sum-exp cross-entropy and its `softmax - one_hot`
gradient. The model has 6,370 trainable parameters. Only split-level loss and
correct-count aggregates are revealed.

Prepare inputs into an MP-SPDZ `Player-Data` directory, then run
`scripts/run_mpc.sh bp_train.mpc` with `MP_SPDZ_ROOT` set. The independent
checker is `scripts/verify/verify_bp_training.py`.

The focused Python tests passed. The original source expanded every example,
gradient, and parameter loop as Python compile-time work; this caused the
earlier compile to reach 9.8 million emitted instructions. The optimized
source uses `for_range_opt` for batch and evaluation loops, `Matrix`/`Array`
persistent memory, and runtime input streams. A definitive compile-only run
completed with exit status 0 in 66.70 seconds, reported 900,000 instructions,
and generated the schedule and bytecode. The compiler reported 37,778 VM
rounds and emitted the expected probabilistic-truncation warning.

The regenerated inputs contain 90 training, 30 validation, and 30 test
examples globally, with balanced ownership across the three parties. The
complete three-party run then completed all 90 updates (three epochs) with
exit status 0. Party 0 reported:

```text
epoch=1 train_loss=113.964
epoch=2 train_loss=24.1647
epoch=3 train_loss=3.45412
validation loss=2.61444 correct=3 total=30
test loss=2.48322 correct=3 total=30
Time = 40.81 seconds
Global data sent = 10045.7 MB (all parties)
```

The aggregate-only verifier passed. The runtime log is preserved in
`results/mpc/bp/training/runtime.log`. The source has no separate one-update
or 30-update execution mode, so those intermediate results are not claimed
as independent runs; the recorded result is the complete 90-update execution.
No logits, gradients, or secret parameters were opened, so a direct
intermediate-value comparison and workload-specific overflow analysis remain
unvalidated. This confirms bounded execution and aggregate diagnostics, but
does not establish full-dataset or production-scale secure BP.

## Post-execution verification addendum

The preserved **A0-original** execution used MP-SPDZ v0.4.4 execution-mode defaults for
the `ring` protocol. In the pinned `Compiler/compilerLib.py`, execution
enables probabilistic truncation and selects `use_split(3)` for the ring
family unless an explicit option is already supplied. The `.mpc` source does
not override either setting. A0-original must therefore be described as using both
runtime defaults, not as a run with both options disabled.

`use_trunc_pr` selects the Rep3-compatible probabilistic fixed-point
truncation path when the operand-width and ring-size conditions permit it.
MP-SPDZ warns that this can leak information about truncation-related values;
it changes the security/privacy analysis and is not only a speed setting.
`use_split(3)` selects three-way local arithmetic-to-binary conversion for
ring non-linear operations such as comparisons and truncation. The argument
is a conversion decomposition count, not the number of parties. These
options must be evaluated separately in any future controlled A/B test.

A separate public synthetic diagnostic was compiled and executed without
changing A0-original. It covered fixed-point input, matrix products, bias addition,
ReLU, stable softmax, cross-entropy, output and hidden gradients, weight and
bias gradients, SGD updates, and persistent state over two consecutive
updates. A dedicated edge fixture found that pinned `ml.relu_prime` returns
`[0, 1, 1]` for `[-1, 0, 1]`, so its zero branch did not implement the
specified strict derivative. The authoritative BP program now uses
`(0 < pre[j])`; the preserved A0-original result is not retroactively relabeled.
Across the non-boundary synthetic values, the largest direct
MPC-versus-FP64 absolute difference was `8.21e-05` for loss; the largest
probability difference was `1.44e-05`, and the largest loss difference was
`9.72e-05`. No branch disagreement or invalid/overflow result occurred.
These are public-fixture results, not disclosures from A0-original.

For `sfix(16,31)`, the pinned source documents a signed nominal range of
approximately `[-16384, 16384)` with quantum `2^-16`. Aggregate reference-side
range analysis over the selected public MNIST records and the three-update
epochs observed maximum absolute values of approximately:

| Quantity | Maximum absolute observed value |
|---|---:|
| Input | 1.0 |
| Hidden pre-activation | 75.28 |
| Hidden activation | 75.28 |
| Logit | 1032.14 |
| Cross-entropy | 624.72 |
| Hidden gradient | 2.382 |
| Weight gradient | 50.13 |
| Parameter update | 1.915 |

The observed FP64 minibatch accumulators were approximately `[-2.675, 0.347]`.
Using three times the largest observed per-example weight-gradient magnitude
gives a conservative three-example accumulator bound of about `150.4` before
cancellation. These are reference-side observations and analytical
workload-specific bounds, not secret MPC-derived openings.

The largest observed value used about 6.3% of the nominal fixed-point range.
This is an observed bounded-run result, not a proof for arbitrary future
training or adversarial inputs. No ring size or precision was changed.

The synthetic diagnostic and range analysis close important evidence gaps but
do not establish a formal leakage proof for probabilistic truncation,
bit-level equality, or absence of overflow outside the measured workload.
These limitations qualify the security claim but are not additional
requirements in the proposal's Stage B gate. Stage B is therefore
**COMPLETE - bounded Secure BP prototype**. Supervisor/security approval is
still required before presenting probabilistic truncation as acceptable for
the project's final privacy claim. The corrected derivative has now been
validated in MPC and in the separate A1-relu-corrected full execution
documented below.

The pinned MP-SPDZ `TruncPrRing` source labels the ordinary non-full-width
path used by the current BP compilation as `CdH10-fixed, Protocol 3.1`. The
2024 Curl paper ([ePrint 2024/1127](https://eprint.iacr.org/2024/1127))
proves stand-alone semi-honest security for the Escudero protocol with
respect to a modified probabilistic-truncation functionality that reveals a
random cutoff point. The paper explains that the cutoff partitions the
secret's low-order fractional residue; it does not claim zero leakage or
universal composability. The MP-SPDZ source is a related
Catrina/de Hoogh-family construction, but the paper is not a formal proof of
every MP-SPDZ branch or arbitrary composition. The candidate policy is
therefore to retain probabilistic truncation with this qualified
stand-alone limitation.

The strict ReLU correction was directly executed under Rep3 Ring with public
fixed-point `[-1, 0, 1]`; exit status was `0`, with
`relu_derivative=[0, 0, 1]`. A corrected full run was subsequently authorized and completed as a separate
run after this security memo. A0-original remains historical and must not be
overwritten.

## Corrected full execution (A1-relu-corrected)

After the qualified truncation decision, the corrected authoritative source
was executed as a separate A1-relu-corrected run with the same protocol, 64-bit ring,
`sfix(16,31)`, architecture, learning rate, batch size, dataset subset,
initialization, and input hashes as A0-original. The strict derivative
`(0 < pre[j])` was used. Compilation and execution both exited with status
`0`; all 90 updates and 3 epochs completed.

```text
epoch=1 train_loss=113.964
epoch=2 train_loss=24.1644
epoch=3 train_loss=3.4539
validation loss=2.61574 correct=3 total=30
test loss=2.483 correct=3 total=30
Time = 6.99069 seconds
Global data sent = 353.975 MB (all parties)
```

The aggregate verifier passed. A1-relu-corrected is close to A0-original but
not bit-identical:
probabilistic truncation is randomized, and A1-relu-corrected also contains the strict
zero-boundary derivative correction. The A0-original log and metadata remain
unchanged. A1-relu-corrected evidence is stored in
`results/mpc/bp/training/corrected_runtime.log` and
`results/mpc/bp/training/corrected_run_metadata.json`.

The A1-relu-corrected communication and runtime figures must not be directly
compared with A0-original as causal performance results. Both reported totals
include preprocessing, but A0-original preserves three party benchmark blocks
while the captured A1-relu-corrected log preserves only one visible party
benchmark block plus a global summary. The values are therefore
**NOT COMPARABLE** for attributing communication or runtime improvement to the
ReLU correction. They remain measured run-specific totals including
preprocessing.

The bounded correctness evidence above is not a formal performance benchmark.
Per-operation MPC profiling belongs to a separate profiling stage and has not
been established by these runs.

## Comparison-ready BP-COMP-001

The separate `BP-COMP-001` run uses the same bounded model, protocol,
precision, selected MNIST indices, and 90/30/30 split, but changes the
comparison policy deliberately: each party's 30 training records are
independently shuffled for each epoch (`order_seed=42042`), and the public
initial parameters use deterministic He/Xavier initialization
(`initialization_seed=42`). A0/A1 are unchanged.

The matched FP64 reference reports final validation `13/30` and test `14/30`,
with test loss `1.5760615`. The secure run reports validation `13/30`, test
`14/30`, validation loss `1.82413`, and test loss `1.57591`. Its runtime was
`8.36998` seconds and global communication was `353.975 MB`, including
preprocessing. These figures are not causal comparisons with A0/A1 and do not
constitute Stage D profiling. Full details and hashes are in
`docs/reports/bp_comparison_readiness.md` and the
`results/mpc/bp/comparison_baseline/BP-COMP-001/` manifest.

## Evidence integrity note

The preserved A0 metadata contains the embedded value
`e01936296ec2b7c771e93a20a46fead24b3d7d2e42826127d67c17b996c2c314` for
`runtime.log`, but the current raw file hashes to
`e01936296ec2b7c771e93a20a46faed24b3d7d2e42826127d67c17b996c2c314`.
The embedded A0 log hash is therefore not verified. The raw log and metadata
are preserved unchanged; this discrepancy does not invalidate the independently
verified A1 log and metadata hashes:

- A0 metadata: `34458feabf2103a2996f160bdb761c84c89c2583f7580667ac4273c902f1a5f3`
- A1 runtime log: `13899e8e4e494a522ab0cdb1782afe630572b4184eba4b5fe3fbfa09735b693c`
- A1 metadata: `df45b8f64afa3a25403ac32b6bf55b1faf6c5df38c9427bf3243d8203a5d43ce`
