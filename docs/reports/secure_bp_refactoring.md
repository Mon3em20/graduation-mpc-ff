# Secure BP refactoring audit

## Audit findings

Before refactoring, `bp_train.mpc` and `bp_matched.mpc` independently
implemented the same affine/ReLU forward pass, stable softmax cross-entropy,
manual reverse-mode gradients, minibatch averaging, SGD updates, persistent
parameters, and aggregate evaluation. The bounded source duplicated the
algorithm for one hidden layer and a fixed three-party batch; the matched
source duplicated it for two hidden layers and a fixed 128/8 batch schedule.
The two input scripts also duplicated token writing, hashing, metadata
construction, and parameter-stream handling.

The duplication was risky because changes to ReLU boundaries, loss
stabilization, gradient ordering, or update averaging could silently diverge
between experiments.

## Refactoring design

[`bp_secure_template.mpc.in`](../../mpc/programs/bp/bp_secure_template.mpc.in)
is now the authoritative MPC implementation. The
[`generate_secure_bp_program.py`](../../scripts/generate_secure_bp_program.py)
renderer substitutes compile-time layer sizes, split counts, batch schedule,
epoch count, and learning rate. It emits independent source snapshots because
MP-SPDZ dimensions and loop bounds are compile-time values; runtime JSON is not
used to select tensor dimensions.

The bounded profile is emitted as `bp_train_refactored.mpc`; the matched
profile is emitted as `bp_matched.mpc`. The historical `bp_train.mpc` source
and all A0, A1, and BP-COMP-001 artifacts remain untouched.

The shared algorithm uses lists of compile-time-sized matrices and arrays:
forward activations are recorded for every layer, deltas are propagated in
reverse order, gradients are accumulated for every affine layer, and each
layer is updated using the actual batch size. The final matched batch therefore
uses eight rather than 128 in its divisor.

## Compatibility risks

- The historical bounded source is preserved rather than silently replaced.
- The refactored bounded profile has the same dimensions, schedule, strict
  positive ReLU derivative, fixed-point setting, and input ownership policy,
  but requires MP-SPDZ compilation before it can be considered execution
  compatible.
- Generated sources are reproducible from the template and profile constants;
  they must be regenerated after template changes.
- Full MPC compilation and execution were not possible in this environment
  because `MP_SPDZ_ROOT` is not configured. No runtime or communication
  result is inferred.

## Validation

The regression tests verify both parameter counts, architecture dimensions,
batch counts, the incomplete matched batch, and presence of the shared
forward/loss/backward/update operations in both rendered profiles.

## Runtime-loop refactoring

The shared template now stores activations, preactivations, deltas, and
gradients in runtime-accessible `sfix.Array` and `sfix.Matrix` containers.
Epochs, full batches, samples, accumulation dimensions, and parameter updates
use `@for_range_opt`; fixed Python loops remain only for the statically known
layer structure and output dimensions. The matched schedule is represented as
39 runtime full-batch iterations followed by a distinct eight-sample update,
with ownership still based on the per-epoch global training position modulo
three.

The loop optimization budget is deliberately conservative (`100`) to limit
compiler memory use. With `-l -M -E ring`, the one-epoch,
784-128-128-10 diagnostic profile compiled successfully under a 4 GiB virtual
memory limit at approximately 0.5 million compiled lines. This is a
compile-only result; no matched five-epoch compilation or MPC execution is
claimed.
