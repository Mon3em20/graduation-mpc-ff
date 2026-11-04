# Technical design

The project compares plaintext BP and FF, then evaluates their secure
computation costs under a common MPC setting. This document distinguishes
implemented evidence from planned work.

## Implemented and verified

### Plaintext layer

`src/baselines/bp.py` implements the project `784-128-128-10` ReLU BP
baseline with cross-entropy, SGD, learning rate `0.05`, batch size `128`, and
five epochs on the 5,000/1,000/1,000 MNIST project split.
`src/baselines/ff.py` implements greedy local FF with 128-unit layers, L2
normalization, threshold `2.0`, per-layer SGD at learning rate `0.01`, and
uniform incorrect-label negatives. Benchmark v1 standardizes the two-layer
FF profile and all-layer inference. These plaintext architectures are not the
smaller secure BP architecture.

### Secure BP layer

The implemented secure program is a bounded `784-8-10` BP network with 6,370
parameters. It runs in three-party semi-honest honest-majority Rep3 Ring over
a 64-bit ring using `sfix(16,31)`. The input script selects MNIST globally as
9 training, 3 validation, and 3 test examples per class, yielding 90/30/30;
sample-index modulo three gives each party 30/10/10 examples per split.
Party 0 supplies the initial parameters and all three parties supply their
assigned streams.

The forward pass computes affine hidden activations, ReLU, affine logits, and
MP-SPDZ stabilized softmax. Cross-entropy uses the matching stabilized
log-sum-exp form. The output error is `probabilities - target`; hidden errors,
weight gradients, bias gradients, and manual SGD updates follow from it.
Parameters persist through three epochs of 30 batch-size-3 updates. Only
aggregate split losses and correct counts are opened.

`A0-original` is the historical run using `ml.relu_prime(pre)`.
`A1-relu-corrected` is the corrected complete run using the strict derivative
`(0 < pre[j])`. Both use the retained runtime defaults for probabilistic
truncation and `use_split(3)`, with the qualified security limitation recorded
in the reports.

### Numerical verification

- Tiny BP and ReLU programs reveal synthetic intermediate values and are
  checked by independent Python references.
- The strict ReLU boundary was separately executed in Rep3 MPC with public
  fixed-point `[-1, 0, 1]`; its diagnostic source is not tracked.
- A public synthetic BP fixture exercises forward, loss, backward, gradients,
  updates, and persistent state.
- The preserved matched FP64 comparison shows close aggregate loss and
  accuracy agreement with A1, but is not bit-level equality and is not an
  intermediate-value comparison for the full secure run.
- Reference-side range analysis is empirical for this bounded workload. It
  is not a formal all-workload overflow proof.

### Comparison-ready BP reference

`BP-COMP-001` uses a separate deterministic policy for future BP/FF
comparison: the same selected records and party ownership, independently
shuffled party streams per epoch, and exact He/Xavier initial parameters
shared by the FP64 reference and MPC input manifest. This is a comparison
readiness refinement, not a new stage or a replacement for the historical
A0/A1 evidence.

## Planned components

### Secure FF - Stage C

Secure FF is not implemented. It must provide positive and negative forward
passes, a layer-local goodness objective, secure negative-data generation,
normalization, layer-wise updates, and inference under the common setting.

### Profiling - Stage D

Per-operation runtime, communication, rounds, and preprocessing profiling is
future work for both BP and FF. Compiler counters and total run statistics
are not a Stage D profile.

### MPC-aware modification - Stage E

After both secure implementations are profiled, one measured FF bottleneck
will determine a testable MPC-aware modification. The work requires an
unmodified Secure FF baseline, an ablation, utility results, and
privacy/information-release analysis.

### Optional mechanisms - Stage F

SNNs, gating, selective activation, other datasets, and additional protocols
are optional extensions after mandatory Stages A-E. They are not implemented
in the current repository.
