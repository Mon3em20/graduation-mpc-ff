# Secure BP post-execution scientific audit

**Audit scope:** Read-only analysis of the preserved A0-original bounded run
recorded in `results/mpc/bp/training/runtime.log`. The statement below that no
second execution was started was accurate at the time of this historical
section; the later A1-relu-corrected execution is documented separately below.
The original source, inputs, metadata, and runtime log were not overwritten.

## 1. Training-loss aggregation

In `mpc/programs/bp/bp_train.mpc`, `train_epoch()` initializes
`loss_total = sfix(0)` and adds one cross-entropy value for each of the 90
training examples. The optimizer divides gradients by `BATCH` for each
three-example update, but the returned loss is divided only once, at the
print site, by `TRAIN`:

```text
print_ln(..., (train_loss / TRAIN).reveal())
```

Therefore the logged values are **mean loss over all 90 examples**, not sums of
30 minibatch means. The raw log is unchanged. The logged values would be
misinterpreted if divided by 30 again.

## 2. Implementation audit

The inspected program has the following consistent dimensions and operations:

- `w1`: 784 x 8, `b1`: 8; hidden ReLU.
- `w2`: 8 x 10, `b2`: 10; output logits and stabilized softmax.
- One-hot targets are emitted in class-index order `0..9`.
- The output error is `probabilities - target`.
- A0-original hidden backpropagation used `error * w2^T` and
  `ml.relu_prime(pre)`; A1-relu-corrected uses `error * w2^T` and the strict
  derivative `(0 < pre[j])`. The strict derivative was separately verified
  inside Rep3 MPC with public fixed-point inputs `[-1, 0, 1]`.
- Both weight matrices and both bias vectors are updated.
- Each update averages the accumulated three-example gradients by `BATCH`.
- The learning rate is applied once, with `LR = 0.05`.
- The model state is initialized once from party 0 and persists across all
  epochs.
- Validation and test ownership uses `sample % 3`, matching the input writer.

The input preparation script emits features followed by one-hot labels in the
same order consumed by `input_from()`. The selected data are class-major, and
the three repetitions of the training rows provide the same global batch order
for each epoch.

No double batch division, missing bias update, transposition error, or
label-index mismatch was found in the inspected source.

## 3. Matched FP64 reference

A temporary, non-MPC FP64 reference was constructed from the existing
`/tmp/bp-mpc-input/metadata.json` manifest and the same MNIST records,
initialization formulas, class-major order, batch size, learning rate, loss,
and 90-update schedule. It was not substituted with the general Benchmark v1
results.

| Quantity | Secure run | Matched FP64 | Difference |
|---|---:|---:|---:|
| Epoch 1 mean training loss | 113.964 | 113.9644328 | -0.00043 |
| Epoch 2 mean training loss | 24.1647 | 24.1666402 | -0.00194 |
| Epoch 3 mean training loss | 3.45412 | 3.4537260 | +0.00039 |
| Validation mean loss | 2.61444 | 2.6168357 | -0.00240 |
| Validation correct | 3/30 | 3/30 | same |
| Test mean loss | 2.48322 | 2.4815983 | +0.00162 |
| Test correct | 3/30 | 3/30 | same |

The close agreement, especially the identical final accuracies and nearly
identical loss trajectory, is evidence against an MPC-specific implementation
error. It does not prove bit-for-bit equality: MP-SPDZ's fixed-point
arithmetic, secure comparisons, exponential/logarithm approximations, and
probabilistic truncation remain distinct from FP64.

### Interpretation of the 10% accuracy

The low accuracy is not evidence by itself of a broken neural-network
implementation. The matched FP64 model shows the same bounded-regime
behavior. At the final epoch the plaintext training predictions are heavily
concentrated in class 1:

- Training prediction histogram: `[6, 66, 3, 0, 3, 3, 0, 6, 3, 0]`.
- Validation prediction histogram: `[0, 22, 0, 0, 0, 0, 0, 0, 8, 0]`.
- Test prediction histogram: `[0, 24, 0, 0, 0, 0, 0, 0, 6, 0]`.

The final correct-by-true-class counts in the matched reference are:

- Validation: `[3, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.
- Test: `[3, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.

These are **plaintext reference diagnostics**, not secret MPC disclosures.
The existing aggregate-only MPC run did not emit prediction histograms or
per-class counts, so secure histograms cannot be recovered from the preserved
log. Producing them would require a separately documented diagnostic program
and another execution.

## 4. Numerical verification gaps

The evidence separates the main possibilities as follows:

- **Training behavior mismatch:** not observed; secure and FP64 loss
  trajectories agree closely.
- **MPC implementation error:** no source-level error found, and aggregate
  behavior matches FP64.
- **Fixed-point quantization/truncation:** present in principle, but its
  observed aggregate effect is small in this run.
- **Probabilistic-truncation effects:** not isolated from other fixed-point
  effects by the existing run.

The run did not reveal logits, gradients, parameters, or per-example
predictions. Consequently, it does not establish per-operation error bounds,
overflow margins, or bit-level agreement.

## 5. MP-SPDZ suggested options

The pinned MP-SPDZ v0.4.4 source shows that:

- `program.use_trunc_pr = True` selects the Rep3-compatible probabilistic
  truncation path when the ring and operand-width conditions permit it. The
  compiler warns that probabilistic truncation can leak some information; it
  is not merely a performance switch. It changes the truncation protocol and
  can change numerical rounding behavior.
- `program.use_split(3)` selects three-way local
  arithmetic-to-binary share conversion for ring non-linear operations,
  including comparison/truncation paths. The source labels the relevant
  truncation construction as ABY3. It is applicable to the three-party ring
  setting and is intended to reduce conversion cost/rounds, while changing
  the conversion implementation and preprocessing requirements.

The runtime command uses the execution-mode defaults in MP-SPDZ's
`compilerLib.py`: for the `ring` protocol it enables probabilistic truncation,
and for the ring family it enables split-3 unless an explicit split option is
already present. This explains why the runtime recommends both options even
though they are not present in the `.mpc` source. The successful run therefore
already reflects those runtime defaults; it was not a baseline with both
options disabled.

The truncation warning and the documented leakage warning mean
`use_trunc_pr` must not be presented as a free performance optimization.
Security and numerical behavior must be reviewed before changing the
authoritative configuration. The two options may be tested independently, but
they should not be changed together if the goal is causal attribution.

### Controlled A/B design

Preserve the current run as A0-original. For any future execution, keep the same
compiled source logic, input hashes, ring size, precision, dataset, seed,
initial parameters, and 90-update schedule:

1. **A0-original:** current runtime defaults; record runtime, global/per-party
   communication, round counters, losses, and aggregate accuracy.
2. **OPT-split-off:** explicitly disable split conversion while retaining the existing
   truncation behavior, if supported by the selected runner; compare only
   conversion-related cost and output drift.
3. **OPT-truncation-alternative:** retain split-3 and change only probabilistic
   truncation behavior;
   this requires an explicit security review because the alternatives have
   different leakage and rounding properties.
4. Do not combine both changes until `OPT-split-off` and
   `OPT-truncation-alternative` are understood independently.

No A/B execution was performed in this audit.

## 6. Communication and rounds

The preserved runtime reports communication including preprocessing/offline
traffic. Using the global total of 10,045.7 MB:

- Per optimizer update: `10,045.7 / 90 = 111.619 MB`.
- Per epoch: `10,045.7 / 3 = 3,348.567 MB`.
- Per party, using the reported approximately 3,348.6 MB: about
  `37.207 MB/update` and `1,116.2 MB/epoch`.

These are total-communication estimates including preprocessing, not
online-only communication. The preserved log does not provide a separate
offline/online byte breakdown. MP-SPDZ's displayed benchmark line explicitly
states that preprocessing is included; a separate breakdown would require a
different measurement run or lower-level instrumentation.

The compiler's reported 37,778 VM rounds are a static compile-time schedule
estimate. The runtime counters (`~113,750` for parties 0 and 2 and
`~73,452` for party 1) are party-local execution/communication counters.
They are not compatible quantities and must not be summed or directly
equated. The disparity is not evidence of a failed execution.

## 7. Conclusion and next step

The smallest justified conclusion is that the observed 10% accuracy is
consistent with the intentionally tiny model, initialization, and dataset
regime, rather than a demonstrated MPC training bug. At the time of this
historical section, a second 90-update run was **not necessary to establish
the immediate root cause** and had not been started. A later corrected
full run is documented as A1-relu-corrected below.

A future diagnostic run is justified only if per-class secure prediction
histograms or per-operation numerical bounds are required. It should be
separate from the preserved run, use aggregate-only disclosures, and change
one option or diagnostic surface at a time.

The successful original artifacts remain preserved:

- `results/mpc/bp/training/runtime.log`
- `results/mpc/bp/training/metadata.json`
- existing party input files and recorded hashes
- existing source and configuration files

## 8. Final verification addendum

### Runtime security settings

The pinned v0.4.4 `Compiler/compilerLib.py` sets
`program.use_trunc_pr = True` for executable `ring` programs. It also selects
`program.use_split(3)` for the ring protocol family when no explicit split
option is provided. The authoritative BP source does not override either
setting. Consequently, A0-original used both runtime defaults.

`use_trunc_pr` selects probabilistic truncation for compatible fixed-point
operations. The pinned compiler emits a warning that this protocol can leak
some information, so the resulting claim is not equivalent to an
unqualified semi-honest Rep3 privacy claim. The warning concerns information
associated with truncation, rather than revealing the plaintext model or
training records directly; the precise acceptable leakage depends on the
protocol assumptions and the application's privacy objective. The source
does not provide a project-specific proof that this leakage is acceptable.

Setting `sfix.round_nearest = True` selects deterministic round-nearest
truncation for fixed-point operations, but the pinned source implements it
through a more expensive comparison-based path. It changes numerical
semantics and communication and is not automatically a security replacement:
it would require a separately compiled and measured run plus a security
decision. It was not enabled.

`use_split(3)` selects three-way local arithmetic-to-binary conversion used by
ring comparison/truncation paths. It is applicable to three-party Rep3 Ring;
the `3` is the decomposition count, not a party-count declaration. It may
alter preprocessing, communication, and rounds, but the inspected source
shows no change to the stated semi-honest threat model by itself. It is a
separate optimization axis from probabilistic truncation.

**Assessment:** probabilistic truncation is **potentially acceptable only
with explicit supervisor/security approval** for the project's final privacy
claim. A0-original remains a valid measured feasibility baseline, but the audit does
not silently upgrade its security claim.

### Public synthetic numerical verification

A separate public fixture used `sfix(16,31)` and a 3-2-2 network with two
consecutive persistent SGD updates. It exercised the complete forward,
loss, backward, gradient, and update path. A dedicated boundary fixture found
that the pinned `ml.relu_prime(x)` returns `[0, 1, 1]` for
`x = [-1, 0, 1]`; its zero branch is therefore inconsistent with the
specified `1[x > 0]` derivative.

The authoritative BP program was corrected surgically to use `(0 < pre[j])`
for its derivative. This does not alter the preserved A0-original log or metadata.
The correction is relevant only at an exact fixed-point zero pre-activation,
which was not isolated in A0-original, so A0-original remains the historical
result. The later replacement is documented as A1-relu-corrected.

Compared with FP64 after using the strict derivative in the diagnostic
fixture, the maximum direct MPC absolute errors were:

- loss: `9.72e-05`;
- probabilities: `1.44e-05`;
- hidden/output gradients: below `1.65e-05`;
- parameter values after two updates: below `1.59e-05`.

No branch disagreement or invalid result occurred. The diagnostic was public
and did not use A0-original records or parameters.

### Range analysis

The pinned `sfix` implementation uses 16 fractional bits and 31 total bits,
with nominal range approximately `[-16384, 16384)` and quantum `2^-16`.
Aggregate FP64 analysis over the selected workload observed maxima of 1.0
for inputs, 75.28 for hidden pre-activations, 1032.14 for logits, 624.72
for cross-entropy, 2.382 for hidden gradients, 50.13 for weight gradients,
and 1.915 for parameter updates. The largest observed absolute value was
about 6.3% of the nominal range.

This is evidence of margin for the preserved bounded workload, not a formal
bound for arbitrary future data, updates, or adversarial values. It does not
prove the absence of overflow in a different configuration.

### Stage B decision

The successful end-to-end execution, matched FP64 aggregate behavior,
synthetic operation checks, aggregate-only output review, and workload range
analysis establish a defensible bounded Secure BP baseline. Essential
limitations remain: probabilistic-truncation leakage has no project-specific
approval, and no formal all-workload overflow or per-operation security proof
exists. Stage B is therefore **PARTIAL**, not COMPLETE.

## 10. Final closure security memo

### Strict ReLU boundary verification

The corrected expression was tested directly inside pinned MP-SPDZ v0.4.4
Rep3 Ring execution, using public fixed-point inputs `[-1, 0, 1]`. The
separate fixture used `ml.relu` only for the forward output and used the
authoritative strict expression `(0 < value)` for the derivative. The
complete three-party process exited with status `0` and revealed:

```text
pre=[-1, 0, 1]
relu=[0, 0, 1]
relu_derivative=[0, 0, 1]
```

The zero input was represented exactly as fixed-point zero. This directly
verifies the corrected boundary behavior in MPC without using the
inconsistent `ml.relu_prime` helper. The complete evidence is preserved in
`results/mpc/bp/training/relu_strict_edge_audit.log` (SHA-256
`815b56bcf0139436240d690a786398da3355d56626edbc4888e0a46e0bd3836c`).

### Classification of range evidence

The earlier maximum values are not secret values extracted from A0-original. They are
FP64 reference-side observations over the selected public MNIST subset after
the matched update schedule. The synthetic fixture provides separate
fixed-point/MPC observations for a small public network. The following
categories are distinct:

1. **Observed FP64 values:** the workload range table in the report, including
   the approximately `1032.14` maximum logit.
2. **Observed fixed-point/MPC values:** the public synthetic fixture outputs,
   with errors measured against FP64.
3. **Analytical representation bounds:** `sfix(16,31)` nominal range
   approximately `[-16384, 16384)` and quantum `2^-16`.
4. **Conservative workload bounds:** input pixels are in `[0,1]`; an exact
   first-layer dot-product bound from the initialized weights is obtained by
   summing `|x_i w_i|` plus the bias bound, while later-layer bounds require
   bounds on the evolving parameters and activations. The measured reference
   trajectory is substantially tighter than the unconstrained worst-case
   ring domain, but it is not a formal invariant for all future executions.

The range table covers inputs and weights, first-layer pre-activations and
activations, logits, stabilized exponentials and denominators,
cross-entropy, output/hidden gradients, weight/bias gradients, minibatch
accumulators, and parameter updates. The FP64 reference-side minibatch
accumulator observations were approximately `[-2.675, 0.347]`; as a
conservative bound based on three examples and the largest observed
per-example weight-gradient magnitude, a three-example accumulator could
reach about `150.4` before cancellation. The largest observed absolute value
was approximately `1032.14`, or `6.3%` of the nominal range. No A0-original secret
intermediate was opened. This is sufficient to support numerical safety for
the particular bounded run as an empirical engineering conclusion, but not
as a formal all-workload overflow proof.

### Probabilistic truncation and the cited result

The relevant source paths are `Compiler/floatingpoint.py::TruncPrRing`,
`Compiler/non_linear.py::Ring._trunc_pr`, and `Protocols/Replicated.hpp`.
For the normal A0-original fixed-point paths where the truncation input width `k` is
less than the 64-bit ring width, MP-SPDZ uses the branch labeled
`CdH10-fixed, Protocol 3.1`. It masks the secret with a random value,
reconstructs a masked ring value, and derives the truncated result from the
masked value and secret-shared mask components. The full-width `k == 64`
branch is a distinct leaky ring truncation path; the current BP compile
artifacts show the ordinary `k=47` and `k=63` truncation families rather than
an assertion that every possible fixed-point operation has one identical
path.

The cited paper, [Curl: Private LLMs through Wavelet-Encoded Look-Up
Tables](https://eprint.iacr.org/2024/1127), explains the leakage concern
precisely. For an integer `x`, write the low `f` bits as `x^(f)` and the
truncated value as `floor(x / 2^f)`. Ideal probabilistic truncation rounds up
with probability `x^(f)/2^f`; the ordinary ideal functionality hides the
rounding decision. The modified functionality instead samples a random
cutoff `s^(f)` uniformly in `[0, 2^f-1]`, sets the rounding bit to one when
`x^(f) >= s^(f)` (up to the paper's equivalent inequality convention), and
reveals the cutoff point. The cutoff partitions the possible low-order
fractional residues, so repeated observations can estimate the low bits.

The paper proves, in its Theorem 3.1, stand-alone semi-honest simulation
security of the Escudero et al. protocol with respect to that **modified**
functionality. It also explains that the Catrina/de Hoogh family realizes the
same modified functionality after protocol-specific adjustments. This is not
a proof of zero leakage, not a proof of the ordinary hidden-cutoff
functionality, and not a universal-composability theorem. It is a
stand-alone result for the specified protocol/functionality and assumptions.

The MP-SPDZ implementation is materially related: its source explicitly
identifies the non-full-width path as `CdH10-fixed` and uses the same class of
random-mask/reconstruction construction in a three-party ring. The source
does not cite or formally instantiate the paper's theorem, and the paper's
exact parameter notation and protocol variant are not a drop-in proof for
every MP-SPDZ branch or composition. Therefore the defensible thesis
statement is qualified: A0-original uses a published-family stand-alone
probabilistic-truncation construction under semi-honest Rep3 assumptions,
with truncation-related side information and without a UC guarantee. It must
not be advertised as privacy beyond that functionality.

### Deterministic round-nearest alternative

In the pinned source, `sfix.round_nearest = True` routes fixed-point
truncation to `TruncRoundNearest`, which adds a half-unit and invokes a
comparison-based truncation. This avoids the probabilistic truncation
rounding/cutoff behavior, but it is more expensive, changes rounding
semantics, and can increase comparisons, rounds, preprocessing, and
communication. It is not automatically a complete privacy proof for the
whole neural-network program. Adopting it would require a new matched
synthetic comparison, a new full execution, and a fresh numerical/communication
record.

### Engineering decision

**Decision: RETAIN PROBABILISTIC TRUNCATION as the candidate A0-original policy**, with
the following explicit limitation: the security claim is restricted to the
published-family stand-alone semi-honest functionality to the extent the
MP-SPDZ `CdH10-fixed` branch matches that construction; no zero-leakage or UC
claim is made. The separate `use_split(3)` conversion setting is retained and
is not conflated with truncation security.

This decision is sufficient to authorize a corrected engineering run under
the same policy, but it does not mark Stage B complete or replace any
project-level privacy approval that may be required for the thesis claim.

### Corrected full execution A1-relu-corrected

A1-relu-corrected was then executed separately with the corrected strict ReLU derivative and
the same protocol, ring, precision, data, initialization, architecture,
learning rate, batch size, and input hashes as A0-original. Compilation and execution
both exited with status `0`, and all 90 updates completed:

| Metric | A0-original | A1-relu-corrected |
|---|---:|---:|
| Epoch 1 loss | 113.964 | 113.964 |
| Epoch 2 loss | 24.1647 | 24.1644 |
| Epoch 3 loss | 3.45412 | 3.4539 |
| Validation loss | 2.61444 | 2.61574 |
| Validation correct | 3/30 | 3/30 |
| Test loss | 2.48322 | 2.483 |
| Test correct | 3/30 | 3/30 |

A1-relu-corrected passed the aggregate verifier. Its preserved log and metadata are
`results/mpc/bp/training/corrected_runtime.log` and
`results/mpc/bp/training/corrected_run_metadata.json`. A1-relu-corrected is not
expected to be bit-identical to A0-original because probabilistic truncation is randomized and the
strict derivative correction changes the exact-zero branch.

### A0-original/A1-relu-corrected measurement reconciliation

Both raw logs state that their MP-SPDZ benchmarks include preprocessing
(offline-phase) traffic. A0-original contains three visible party benchmark
blocks and reports:

- `3348.61 MB` for party 0;
- `3348.56 MB` for party 1;
- `3348.56 MB` for party 2;
- `10045.7 MB` global traffic.

Those party values approximately reconcile to the A0-original global total.

A1-relu-corrected launches all three party processes and reports a global
total of `353.975 MB`, but its preserved raw log contains only one visible
party benchmark block: party 0 at `98.2271 MB` and approximately `59564`
rounds. It does not contain independent party-1/party-2 benchmark blocks or
independent per-party exit-status lines. The metadata records execution exit
status `0`, but the raw log does not independently establish each party's
exit code.

The compiled schedules also differ: A0-original records `37778` compiler VM
rounds, while A1-relu-corrected records `40508`; A1 additionally records
`4609470` triples, `7424425` probabilistic truncations, and `28380` three-way
splits, while the A0 metadata does not preserve equivalent counts. The
preserved evidence does not establish identical preprocessing reuse,
environmental conditions, or output-capture coverage.

Accordingly, both totals are retained as measured, preprocessing-inclusive
run-specific values, but the A0-original/A1-relu-corrected communication and
runtime comparison is **NOT COMPARABLE** as a causal performance
measurement. No communication or runtime improvement is attributed to the
ReLU correction. Formal per-operation MPC profiling remains separate future
work.

### Final Stage B assessment

- **Mathematical correctness:** supported by source inspection, the strict
  ReLU MPC edge test, public synthetic checks, and matched FP64 aggregates.
- **Corrected full execution:** A1-relu-corrected completed 90 updates and
  three epochs with aggregate verification passing.
- **Numerical verification:** supported for the bounded aggregate workload and
  public synthetic fixtures; not bit-level or formal all-workload proof.
- **Bounded range analysis:** empirically supports the measured workload, not
  arbitrary future executions.
- **Security/truncation:** qualified stand-alone modified-functionality claim;
  no zero-leakage or universal-composability claim, and no proof covering
  every MP-SPDZ execution branch.
- **Communication/runtime quality:** measured per run, but A0-original and
  A1-relu-corrected are not causally comparable because their preserved
  measurement coverage and schedule evidence differ.
- **Reproducibility:** configuration, input hashes, and aggregate outputs are
  preserved; A1 per-party raw benchmark and exit-status coverage is incomplete.

Stage B remains **PARTIAL**. The remaining limitations are claim scope and
measurement quality, not an identified mathematical correctness failure.

## 9. Verification commands and changed files

The public diagnostic was run with the pinned environment and protocol:

```text
cd /home/abdelmonem/MP-SPDZ
/home/abdelmonem/venvs/mp-spdz-ml/bin/python \
  Scripts/compile-run.py -M -E ring synthetic_bp_audit
```

The corrected authoritative program was compile-checked without execution:

```text
cd /home/abdelmonem/MP-SPDZ
/home/abdelmonem/venvs/mp-spdz-ml/bin/python \
  compile.py -M -R 64 bp_train
```

Focused repository tests passed: `3` tests, including the existing synthetic
BP reference test and aggregate-output parser tests. `git diff --check` also
passed. The exact repository changes for this addendum are:

- `mpc/programs/bp/bp_train.mpc`: strict ReLU derivative at zero.
- `docs/reports/bp_mpc.md`: verification and security addendum.
- `docs/reports/bp_mpc_post_execution_audit.md`: final verification findings.
- `docs/reports/progress.md`: bounded Secure BP status and blockers.

The A0-original `runtime.log`, `metadata.json`, party input hashes, and historical
experiment records were preserved. No new full Secure BP run, FF run, or
Stage C experiment was started.
