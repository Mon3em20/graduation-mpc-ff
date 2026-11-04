# BP comparison readiness

## Scope and experiment identifiers

This report prepares a controlled reference for the next Secure FF work. It
does not reopen the completed Stage B gate and does not start Stage C.

- `A0-original`: preserved historical bounded BP execution.
- `A1-relu-corrected`: preserved corrected bounded BP execution.
- `BP-COMP-001`: new comparison-ready bounded reference execution.

The A0/A1 logs, metadata, input copies, Benchmark v1 outputs, and Stage A
records were not overwritten.

## Ordering audit

The historical preparation selects records class-major and assigns global
sample position modulo three to parties. `bp_train.mpc` consumes one record
from each party per update. Therefore the historical schedule has 30 updates
per epoch, but each update contains three records of the same class; the ten
class tuples repeat for every epoch. No selected record is skipped or
duplicated, and each party owns 30 training, 10 validation, and 10 test
records.

`BP-COMP-001` preserves the same selected indices and ownership. It shuffles
each party's 30 training indices independently for each epoch using
`order_seed=42042`, then consumes one record from each party for each of 30
updates. Every epoch consumes all 90 selected records exactly once, and the
schedule differs across epochs without requiring every batch to contain three
different classes. The epoch-specific order is recorded in the comparison
manifest; it is not emitted by the normal MPC program.

## Initialization audit and decision

The historical initializer is public and deterministic:

```text
W1[i,j] = (i - j) / 1000
W2[i,j] = (i - j) / 1000
b1 = b2 = 0
```

Its combined weight range is `[-0.009, 0.783]`, mean `0.38310`, standard
deviation `0.22905`, and variance `0.05246`. On the selected training
records, hidden pre-activations and activations ranged from `12.449` to
`75.271`, while logits ranged from `-3.292` to `2.082`. The earlier bounded
trajectory's large logits are therefore attributable to both its initial
scale and subsequent updates, not to initialization alone.

The selected comparison initializer is deterministic He normal for the hidden
layer and Xavier uniform for the output layer, with zero biases and
`initialization_seed=42`. Its combined weight range is approximately
`[-0.55587, 0.57299]`, mean `-0.00102`, standard deviation `0.06087`, and
variance `0.00370`. On the selected training set, initial hidden
pre-activations ranged from `-2.386` to `1.109`, logits from `-0.643` to
`0.662`, and no initial maximum softmax probability exceeded `0.99`.
Hidden-neuron weight-vector cosine similarities ranged from `-0.060` to
`0.109`, so the vectors are not near-duplicates.

The four controlled plaintext diagnostics used the same 90/30/30 split,
architecture, batch size, learning rate, three epochs, and initialization
seed:

| Config | Order | Initialization | Final validation | Test |
|---|---|---|---:|---:|
| C0 | historical class-major | deterministic | 3/30 | 3/30 |
| C1 | shuffled per party | deterministic | 6/30 | 4/30 |
| C2 | historical class-major | He/Xavier | 6/30 | 6/30 |
| C3 | shuffled per party | He/Xavier | 13/30 | 14/30 |

C3 was selected using validation behavior and scale diagnostics. The test
column is reported only after that selection and was not used for choosing
the policy. This is a bounded diagnostic, not a hyperparameter search or a
claim of generalization.

## Reproducible artifacts

The shared policy and FP64 reference are implemented in
`scripts/reference/bp_training_reference.py`. The preparation script and the
reference use the same raw MNIST indices, per-party epoch order, labels,
architecture, and exact initial parameter values.

`results/mpc/bp/comparison_baseline/BP-COMP-001/` contains:

- `Input-P0-0`, `Input-P1-0`, and `Input-P2-0`;
- `metadata.json` with selection, ordering, ownership, and input hashes;
- `initial_parameters.json` with the exact parameter values and hash;
- `fp64_reference.json` with epoch metrics, test aggregates, and initialization
  diagnostics;
- `runtime.log` and `run_metadata.json`;
- `relu_strict_edge.log` from the newly tracked diagnostic source.

The FP64 reference reports:

```text
epoch 1: train loss 2.2489326, validation loss 2.1652213, validation 7/30
epoch 2: train loss 1.9589519, validation loss 1.9583528, validation 12/30
epoch 3: train loss 1.6812161, validation loss 1.8242255, validation 13/30
test: loss 1.5760615, correct 14/30
```

The deterministic `2^-16` round-to-nearest public fixed-point emulation
reports:

```text
epoch 1: train loss 2.2489315, validation loss 2.1652078, validation 7/30
epoch 2: train loss 1.9589559, validation loss 1.9583320, validation 12/30
epoch 3: train loss 1.6811851, validation loss 1.8242150, validation 13/30
test: loss 1.5759578, correct 14/30
```

This emulation is not MP-SPDZ's probabilistic truncation protocol. Its largest
loss difference from FP64 is `0.000104` on the test aggregate, and correct
counts match.

The fixed-point secure aggregates are:

```text
epoch 1 train loss 2.24889
epoch 2 train loss 1.95888
epoch 3 train loss 1.68115
validation loss 1.82413 correct 13/30
test loss 1.57591 correct 14/30
```

The largest displayed secure-versus-FP64 aggregate loss difference is below
`0.00016`; correct counts match. Secure-versus-emulation differences are
consistent with fixed-point approximation and probabilistic truncation, but
the emulation is not a protocol-identical reproduction. This is aggregate-level
agreement, not bit-level equality; no individual MPC values were opened.

## Runtime and communication

`BP-COMP-001` compiled successfully with 900,000 reported instructions and
40,508 compiler VM rounds. The preserved execution reports exit status `0`,
runtime `8.36998` seconds, `98.2271 MB` for the visible party-0 block,
approximately 59,564 party-0 rounds, and `353.975 MB` global communication.
The totals include preprocessing. Compile duration was not directly recorded.
These are bounded comparison-reference measurements, not Stage D
per-operation profiling and not a causal comparison with A0/A1.

The run emitted MP-SPDZ's warning that the computation domain is too small for
low-round truncation and retained the existing probabilistic-truncation and
`use_split(3)` policy. No zero-leakage, UC, or arbitrary-workload overflow
claim is made.

## Reproducible strict ReLU evidence

`mpc/programs/bp/relu_strict_edge.mpc` is a newly tracked diagnostic; it is
not claimed to be the missing source that generated the historical
`relu_strict_edge_audit.log`. Its new output is:

```text
pre=[-1, 0, 1]
relu=[0, 0, 1]
relu_derivative=[0, 0, 1]
```

`scripts/verify/verify_relu_strict.py` passes on the new log. The historical
log remains preserved separately.

## Remaining boundaries

The five-seed utility and three-repetition timing requirements for the final
BP-versus-FF comparison remain outstanding. `BP-COMP-001` is one controlled
reference run, not that repeated-trial comparison. Stage B remains complete,
Stage C Secure FF remains not started, and Stages D-F remain unchanged.
