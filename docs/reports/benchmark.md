# Standardized plaintext BP--FF benchmark

## 1. Experimental objective

This benchmark establishes a controlled plaintext reference before secure
MPC implementations are compared. It measures the utility, training work,
timing, inference cost, and process memory of the existing MNIST
Backpropagation (BP) and Forward-Forward (FF) implementations. It is not an
MPC benchmark and provides no privacy guarantee.

The experiment is a new standardized benchmark. It is distinct from the
historical Stage A CSVs and from the larger source-aligned reference
experiments.

## 2. Experimental environment

The benchmark ran on the repository's Linux/WSL CPU environment:

- Python 3.14.4
- PyTorch 2.14.1+cpu
- torchvision 0.29.1+cpu
- Matplotlib 3.11.2
- CPU device, four PyTorch threads
- MNIST already available under `~/datasets/MNIST`

The runner records process-level maximum resident set size with
`resource.getrusage(RUSAGE_SELF).ru_maxrss`, but because all runs share one
process this is a cumulative high-water mark. It is retained as raw metadata
only and is not used for a BP--FF memory comparison.

## 3. Models and configurations

The project-baseline BP model is `784-128-128-10` with ReLU activations,
cross-entropy, SGD at learning rate 0.05, batch size 128, and five epochs.
It has 118,282 trainable parameters.

The project-baseline FF model is `784-128-128` with two local ReLU layers,
L2 normalization between layers, squared goodness, threshold 2.0, local
softplus loss, per-layer SGD at learning rate 0.01, batch size 128, and five
epochs per layer. It has 116,992 trainable parameters. The difference is
1,290 parameters, or 1.09% relative to BP. FF uses uniform random incorrect
labels, and each layer receives positive and negative presentations.

These settings are the standardized two-layer **Benchmark v1**. They must not
be merged with the separate bounded source-aligned FF reference: that
experiment uses four 128-unit layers and 60 epochs per layer. The reference
experiment is retained to assess convergence behavior under a source-aligned
reduced model; it is not a Benchmark v1 result or an exact reproduction of
Hinton's four 2,000-unit-layer experiment.

The original FF baseline evaluates both `all_layers` and
`second_layer_only` candidate-scoring variants and selects the better variant
on validation. Benchmark v1 intentionally defines a fixed `all_layers`
inference profile so every standardized run uses the same full-network
scoring rule; this preserves the existing corrected v1 results rather than
silently substituting the baseline's adaptive variant selection. The baseline
module remains capable of validation-based variant selection for its original
workflow.

BP uses global backpropagation and one optimizer update per minibatch. FF
trains layer by layer and performs the complete positive pass, negative pass,
negative-label generation, goodness calculation, local loss, backward pass,
and optimizer update for each layer.

## 4. Dataset and evaluation protocol

Both methods use the same deterministic split reconstructed from the existing
baseline code:

- 5,000 official MNIST training examples for training
- 1,000 official training examples for validation
- 1,000 official test examples for final evaluation
- Split generator seed 42
- Utility seeds 42--46
- Batch size 128

The first 1,000 positions of the seeded official-test permutation, which were
previously exposed by the legacy experiments, are excluded. The fresh held-out
test set is positions 1,000--1,999. Test accuracy is evaluated only after
validation checkpoint selection. Thresholds and checkpoints are never selected
from test accuracy.

The benchmark uses three timing repetitions for each of the five independent
utility seeds. Repetitions are runtime observations, not additional
independent utility samples.

## 5. Timing methodology

Historical BP epoch timings included validation, while historical FF training
timings excluded validation and test inference. Those historical values are
therefore not directly comparable and remain unchanged.

The new runner, [scripts/benchmark.py](../../scripts/benchmark.py), records
separate boundaries:

- Dataset loading and DataLoader preparation
- Model initialization
- Training-only time
- Validation time
- Selected-checkpoint inference time
- Total measured experiment wall-clock time

BP training time includes forward computation, cross-entropy, backward
calculation, and optimizer update. FF training time includes both positive and
negative presentations, negative-label generation, layer processing, goodness,
local loss, backward calculation, and optimizer update. BP progress is reported
as epochs; FF progress is reported as layer-epochs and is not treated as an
equivalent unit.

The benchmark records optimizer updates and presentation counts. BP has
25,000 training presentations per run (5,000 examples × 5 epochs). FF has
100,000 presentations per run (5,000 examples × 5 epochs × 2 layers × 2
positive/negative passes), comprising 50,000 positive and 50,000 negative
presentations. Dataset loading is measured once globally per invocation;
DataLoader construction is measured separately within each run. Corrected
run-level training time is the sum of epoch/layer-epoch training fields and
excludes validation.

## 6. Utility results

The standardized benchmark produced 15 runs per method: five seeds and three
timing repetitions. Results below aggregate all timing repetitions for the
reported descriptive runtime statistics; test-accuracy variability is shown
over the same recorded runs, while the independent-seed interpretation is
based on the five utility seeds.

| Method | Mean test accuracy | SD | Range | Mean training time (s) | Median (s) | Mean inference (s) |
|---|---:|---:|---:|---:|---:|---:|
| BP | 75.96% | 2.89 | 71.7--78.4% | 1.47 | 1.45 | 0.053 |
| FF | 59.76% | 3.18 | 55.5--62.9% | 3.12 | 3.12 | 0.077 |

BP and FF losses are method-specific and are not compared as if they were the
same objective. BP records cross-entropy; FF records local softplus goodness
loss and positive/negative goodness diagnostics. The epoch- and layer-level
curves are available in `epochs.csv`, and the validation-accuracy figure is
in `figures/accuracy.png`.

## 7. Efficiency results

| Method | Mean time/update (s) | Mean time/presentation (s) | Updates/run | Training presentations/run | Peak RSS |
|---|---:|---:|---:|---:|---:|
| BP | 0.00734 | 0.0000587 | 200 | 25,000 | unavailable |
| FF | 0.00779 | 0.0000312 | 400 | 100,000 | unavailable |

Time per optimizer update is not a claim that FF updates are equivalent to BP
updates: FF updates operate on local layer objectives and the complete FF run
uses twice as many minibatch optimizer updates and four times as many
presentations. Similarly, time per
original training sample uses all required presentations in the denominator
and does not remove FF's negative work.

FF uses more updates and presentations, and its mean update time is slightly
higher than BP under the corrected training-only boundary. In this experiment
BP has lower total training time and higher test accuracy. The measured
inference procedure is also slower for FF because
its candidate-label scoring evaluates the FF model's label candidates rather
than using one ordinary BP forward pass.

## 8. Matched-accuracy analysis

Common validation targets were derived from the overlap of observed validation
accuracy ranges, rounded to five-percentage-point levels. The resulting
validation-only targets were 55%, 60%, and 65%. No test accuracy was used to
choose these thresholds.

At 55%, both methods reached the target for all 15 seed/repetition runs. At
60%, BP reached the target for all 15 and FF reached it for 9. At 65%, BP
reached the target for all 15 and FF reached it for 3. The independent-seed
utility interpretation remains five seeds; matched timing rows keep all three
repetitions separate. FF rows marked `not_reached` are explicit, not
estimated.

For FF, a valid target checkpoint is available only after layer 2, when the
full two-layer inference procedure is defined. Cumulative FF time includes all
five layer-1 epochs before the layer-2 target epoch. Seed/repetition results
are in `matched_accuracy.csv`, aggregate values are in `matched_summary.csv`;
the plot is
`figures/time_to_target.png`.

## 9. Statistical analysis

There are five independent utility seeds (42--46) and three timing repetitions
per seed. Accuracy mean, sample standard deviation, minimum, and maximum use
one outcome per seed. Runtime means, standard deviations, medians, minima, and
maxima use the 15 timing runs and are not independent utility replicates. No
inferential significance test is claimed.

## 10. Discussion

Under the project-baseline configuration, BP converged to substantially higher
held-out accuracy and required less total training-only time. FF's local
optimizer updates were individually shorter, but FF required layer-wise
training, positive and negative presentations, and 400 updates per run versus
200 for BP. FF also reached the higher shared targets less consistently.

These results do not establish that FF is intrinsically inefficient or that BP
will be cheaper under MPC. They establish the plaintext work decomposition that
later secure experiments must preserve. In particular, an MPC comparison must
not omit FF's negative-label generation, negative pass, or candidate-label
inference.

## 11. Limitations

- The benchmark uses the small 5,000-example project baseline, not the
  source-aligned 50,000-example reference configurations.
- Both models are CPU-only and relatively narrow.
- BP and FF optimize different objectives and have different learning
  mechanisms.
- Parameter counts are close but not identical.
- Hyperparameters are the existing project settings, not a jointly tuned
  optimum.
- Independent peak-memory comparison is unavailable. The previous RSS values
  were a process-wide high-water mark collected in one process and are not
  interpreted as per-run or equal BP/FF memory measurements.
- The standardized run does not provide energy measurements; energy is not an
  approved measured proposal metric.
- The benchmark does not establish scaling behavior or secure communication
  cost.

## 12. Relevance to secure MPC

Plaintext benchmarking is required to separate algorithmic utility and work
from protocol overhead. The measured operation counts, layer-wise structure,
inference procedure, and validation protocol provide the baseline for later
secure BP and secure FF programs. Plaintext results do not establish privacy,
security, communication cost, or MPC runtime.

## 13. Correction changelog

The original v1 artifacts included BP validation time in run-level training
time, reported 40 rather than 200 BP updates per run, combined timing
repetitions in matched-target calculations, used duplicated timing runs for
utility statistics, and plotted an invalid memory comparison. The corrected
artifacts:

- derive BP run time and update counts from epoch rows;
- preserve seed and repetition in matched targets and include sequential
  layer-1 time for FF layer-2 checkpoints;
- use one test outcome per seed for utility statistics;
- remove memory figures and method-level RSS claims;
- rename the denominator to `mean_time_per_presentation_seconds`;
- regenerate all affected figures and summaries from raw records.

Benchmark v1 intentionally uses FF threshold 2.0, matching the source default
and the v1 configuration. Historical threshold-3.0 validation-selected
records remain preserved separately. An additional threshold-3.0 standardized
experiment is not necessary to correct v1 and was not run.

## 14. Conclusion

The new standardized benchmark provides a reproducible controlled comparison
of the existing project baselines. BP achieved higher utility and lower total
training time in the measured CPU setting. FF performed more local work and
had weaker target reachability under this fixed configuration, although its
individual updates were shorter. The evidence supports carrying both methods
forward to secure evaluation without claiming a predetermined winner.

## Reproduction

From the repository root:

```bash
source "$HOME/venvs/mp-spdz-ml/bin/activate"
python scripts/benchmark.py --method all
python -m unittest discover -s tests -v
```

Use `--method bp`, `--method ff`, or `--quick` for focused validation. Raw
outputs are in [results/benchmarks/plaintext](../../results/benchmarks/plaintext).
Historical baseline files under `results/baselines/` and `results/archive/`
are not overwritten.
