# Project progress

This report uses the official proposal stages and the preserved repository
evidence as of 2026-11-03.

## Matched Secure BP preparation

**Status: IMPLEMENTED, EXECUTION NOT YET CLAIMED.**

`SECURE-BP-MATCHED-v1` is an additive `784-128-128-10` implementation
matching the plaintext Benchmark v1 configuration. Its source,
configuration, deterministic input generator, and focused tests are separate
from the bounded Stage B `784-8-10` experiments. No runtime, communication,
full-training, or five-seed result is attributed to this experiment until an
actual MP-SPDZ run produces preserved evidence.

The future bounded and matched source profiles now share the compile-time
template in `mpc/programs/bp/bp_secure_template.mpc.in`. The historical
`bp_train.mpc` source remains preserved as a compatibility snapshot.

## Stage A - Environment and plaintext feasibility

**Status: COMPLETE - 2026-11-02.**

Plaintext BP/FF baselines, Benchmark v1 outputs, source-aligned references,
MP-SPDZ feasibility evidence, the secure-operation inventory, and the selected
Rep3 Ring setting are preserved. Stage A does not claim Secure FF, full-dataset
MPC training, or Stage D profiling.

## Stage B - Small end-to-end secure BP prototype

**Status: COMPLETE - bounded Secure BP prototype.**

The proposal gate is supported by `A1-relu-corrected`:

1. Complete input-to-output secure execution: 90 examples, 30 validation
   examples, 30 test examples, three epochs, and 90 SGD updates completed.
2. Runtime measurement: 6.99069 seconds is preserved in A1 metadata.
3. Communication measurement: 353.975 MB global traffic, including
   preprocessing, is preserved in A1 metadata.
4. Agreement with plaintext emulation: matched FP64 aggregate losses and
   correct counts are preserved in the BP audit.

`A0-original` remains the historical pre-correction run. Both runs use
`784-8-10`, 64-bit Rep3 Ring, `sfix(16,31)`, and aggregate-only diagnostics.
Their measurement coverage and schedules differ, so A0/A1 totals are not
causally comparable.

This is bounded evidence, not full-MNIST or production-scale Secure BP. It
does not establish formal arbitrary-workload overflow safety, zero leakage,
universal composability, or Stage D profiling.

### Comparison-readiness refinement

`BP-COMP-001` is a separate bounded reference run, not a new official stage.
It preserves the selected 90/30/30 MNIST indices and party ownership while
using deterministic per-party epoch shuffling and He/Xavier initialization.
Its matched FP64 reference and aggregate secure outputs agree closely
(validation `13/30`, test `14/30`). The five-seed utility and three-repetition
timing requirements for the final BP-versus-FF comparison remain outstanding.

## Stage C - Secure Forward-Forward

**Status: NOT STARTED.**

Secure FF remains future work: positive/negative secure passes, local
goodness objectives, negative-data generation, normalization, layer-wise
updates, and inference are not implemented.

## Stage D - Profiling of BP and FF

**Status: NOT STARTED.**

Per-operation profiling must cover both secure BP and secure FF. Existing
compiler counters, quickstart counters, and total run statistics are not
Stage D profiling.

## Stage E - Evidence-driven MPC-aware FF modification

**Status: NOT STARTED.**

The preliminary plaintext hard-negative comparison is not Stage E completion.
After Stage D identifies a measured bottleneck, one MPC-aware modification,
baseline comparison, ablation, and utility/privacy trade-off analysis are
required.

## Stage F - Additional mechanisms

**Status: NOT STARTED - OPTIONAL.**

SNNs, gating, selective activation, other datasets, and additional protocols
begin only after mandatory Stages A-E.

The final BP-versus-FF performance comparison is a mandatory future
cross-stage outcome, not Stage F.
