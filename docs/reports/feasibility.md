# Stage A feasibility report

**Final status: COMPLETE (2026-11-02).**

This report closes Stage A against the proposal-defined feasibility gate. It
does not claim secure BP or FF training, exact reproduction of published
results, numerical safety for a future full neural-network MPC workload, or
Stage D per-operation profiling.

## Final Stage A acceptance checklist

| Proposal criterion | Evidence and disposition |
|---|---|
| Install and pin the MPC framework | MP-SPDZ `v0.4.4`, commit `113c4e47dd94bca9ecab029d2f0abe649bea5c52`; recorded below. |
| Verify that the framework builds and executes its documented example | The three-party ring ML quickstart completed and returned `[0, 1]`; see the quickstart evidence below. |
| Implement and validate plaintext BP and FF baselines | Corrected standardized Benchmark v1 and source-aligned bounded BP/FF references are preserved under `results/`. |
| Establish reproducible reference accuracy and convergence behavior | BP 20-epoch and FF 60-epoch-per-layer records are preserved; the bounded reduced-model evidence is accepted as the project reference gate. |
| Inventory required secure operations | The operation inventory and support classifications are recorded in this report. |
| Map FF operations to MP-SPDZ primitives | Fixed-point arithmetic, ReLU/comparison, normalization, softplus, softmax/cross-entropy, and secret-label handling are mapped from isolated tests. |
| Identify direct support versus custom fixed-point requirements | Direct/API support, isolated execution, and future custom implementation requirements are explicitly separated below. |
| Identify compile-time and runtime counters | VM rounds, communication estimates, preprocessing-related counts, and runtime output are recorded; no Stage D profiling is claimed. |
| Select protocol and threat model from feasibility evidence | MP-SPDZ Rep3 Ring, three parties, semi-honest, honest majority, at most one corrupted party, observed 64-bit ring. |

All proposal-defined Stage A criteria are supported. The known truncation
warnings and the absence of secure project training are documented limitations
of the selected configuration and are not additional Stage A gate criteria.

This report records measured evidence and explicitly separates it from
secure-operation assumptions. No project BP/FF secure implementation or
full-dataset secure experiment is claimed.

## Audit against the actual proposal

The attached proposal, [`docs/proposal.pdf`](../proposal.pdf), sets the Stage A gate as: the framework builds and runs; plaintext BP and FF reproduce published small-scale behavior to establish reference accuracy/convergence; and an inventory identifies unsupported or costly operations (Table 2 and §3.6). Section 3.3 adds that Stage A must map FF operations to MP-SPDZ primitives, determine whether its ML library can express the required layers or whether direct fixed-point code is needed, and confirm which runtime/compile-time counters are available. The proposal explicitly assigns per-operation cost breakdown to **Stage D** (§3.3, §3.6, §3.7). Accordingly, this report distinguishes source/API support and likely cost from measured secure cost; no Stage D profiling is claimed.

The plaintext CSVs establish reproducible project baselines, but they do not
establish paper-matched reference behavior. The proposal does not state one
numeric target/configuration for both baselines. Hinton [16] gives a broad BP
reference for permutation-invariant MNIST and a specific much wider FF result.
The project's reduced models remain bounded feasibility references, not exact
reproductions. No paper accuracy is treated as a guaranteed threshold for a
changed model.

### Source-aligned reference configurations

The proposal's Stage A criterion is reference accuracy/convergence for the plaintext baselines, not necessarily bit-for-bit reproduction. Hinton's [paper, §3.1](https://arxiv.org/html/2212.13345v1) does not specify one unique architecture or complete hyperparameter recipe for its BP comparison; it reports that fully connected ReLU nets with a few hidden layers typically reach about 1.4% error in roughly 20 epochs. The BP reference check therefore keeps the project's 784-128-128-10 network/SGD implementation, but uses 50k training and 10k validation examples for 20 epochs. It uses cross-entropy, plain SGD at 0.05, batch size 128, and no scheduler or explicit regularizer; these are project choices because Hinton's BP discussion does not supply one exact recipe. It evaluates on 9k fresh official-test examples because the first 1k of the seeded test permutation were previously exposed. This is a source-aligned check, not an exact Hinton reproduction.

For supervised FF, Hinton's [§3.3](https://arxiv.org/html/2212.13345v1) reports 4×2,000 ReLU units, label-coded first ten pixels, layer-wise goodness, normalization between layers, 60 epochs, hard negatives chosen from a neutral-label pass, and inference by summing goodness over layers 2–4 for each candidate label; the paper reports 1.36% test error. That model has about 13.576 million parameters including biases (about 51.8 MiB of FP32 weights). Scaling dense training multiply work from the measured threshold-3 FF run (3.431 s for 5k examples × 10 layer-epochs with 116,992 parameters) gives a rough **26.6 CPU-hours per seed** for 50k examples × 4 layers × 60 epochs, before the paper's hard-negative selection overhead. A nominal working-set floor is about **119 MiB** for weights, the 60k uint8 MNIST images (~47 MiB), largest active-layer gradients (~15.3 MiB), and one batch of hidden activations (~4 MiB); Python/framework workspaces add overhead, and peak memory was not measured. The full Hinton width is therefore not run blindly.

The bounded FF reference check uses four 128-unit layers, 50k/10k
train/validation, 60 epochs/layer, batch 128, threshold 2.0, SGD at 0.01,
and 9k fresh test examples. It replaces the first ten flattened pixels with a
one-hot label; pairs each correct-label positive with an incorrect-label
negative; computes pre-normalization squared-ReLU activity as goodness;
optimizes the local softplus objective; L2-normalizes activations between
layers; and freezes earlier layers while training each later layer. Inference
scores each candidate by summing goodness from layers 2–4. These mechanics
match the central supervised FF description. The important deviations are
uniform random incorrect labels instead of neutral-pass hard negatives and
128 units instead of 2,000. This is source-aligned bounded evidence, not an
exact reproduction. The 150,016-parameter model has about 0.57 MiB of FP32
weights; its accuracy gap is reported below and is not attributed to one
cause.

The BP audit reran three seeds (42–44) and restored all 20-epoch records under `hinton_bp_audit_v2`; the earlier summary-only ID remains. Best validation accuracy occurred at epoch 20 for all seeds. Mean train loss fell from **1.106 at epoch 1** to **0.071 at epoch 20**, mean validation loss from **0.439 to 0.114**, while mean validation accuracy rose from **87.86% to 96.58%**. The network learns consistently and is not obviously broken, but curves had not plateaued by epoch 20. Test accuracies on 9k clean examples were **96.89%, 97.01%, 97.18%** (mean **97.03%**, sample SD **0.15 points**, range 96.89–97.18%; mean error 2.97%). The gap from Hinton's approximate 1.4% typical observation could plausibly reflect the narrower two-hidden-layer, 128-wide model and unspecified initialization/training details; these results do not isolate the cause. A single 30-epoch run per same seed/configuration is a bounded follow-up if a plateau check is needed, not a broad search. Training sums were 69.62–70.05 s (mean 69.87 s); these include validation timers.

The first FF run evaluated test only at the final layer-4 epoch; it did not checkpoint on validation. The corrected `hinton_ff_checkpoint_v2` run evaluates layer-2-through-4 candidate goodness on validation after each layer-4 epoch, selects the layer-4 state using validation only, and reports fixed-60 and selected-checkpoint test scores separately. All three seeds selected epoch 1: mean best validation accuracy **74.06%** (SD **0.76 points**) and selected-checkpoint test accuracy **75.96%** (SD **0.78**, range **75.30–76.82%**). The fixed-60 result was mean validation **71.43%** and test **73.34%** (SD **1.62**, range **72.06–75.16%**), matching the prior run. Layer-4 validation means were 74.06% at epoch 1, 72.30% at epoch 10, 71.97% at epoch 20, and 71.43% at epoch 60. Training sums averaged **778.99 s** (SD **8.19 s**, range **773.70–788.43 s**), excluding validation and test. The checkpoint correction recovers about 2.62 points of mean test accuracy but does not close the large gap from Hinton's result. The code audit found no bug in label overwrite, paired positive/negative construction, pre-normalization squared goodness, local softplus loss, L2 normalization, greedy freezing, or layers-2-through-4 inference. Width and negative sampling remain material differences; width alone is not asserted as the explanation.


### Controlled hard-negative comparison

Experiment ID: `hinton_ff_negative_comparison_v1`.

The matched comparison used the same four-layer, 128-unit network,
seeds 42–44, data split, threshold, optimizer, and 60 epochs per layer.
Both arms trained the same auxiliary neutral-label softmax predictor.
The uniform-random control ignored its predicted distribution when
sampling negative labels; the hard-negative arm sampled an incorrect
label from the predictor's probabilities after masking the true class.
Both arms used the same uniform warm-up for layer 1; the hard-negative
selector was used for layers 2–4. This is a source-aligned adaptation,
not an exact reproduction of Hinton's full-width experiment.

| Metric | Uniform-random control | Neutral-softmax hard negatives |
|---|---:|---:|
| Best-checkpoint test accuracy, mean ± sample SD | 75.96% ± 0.78 points | 75.01% ± 0.92 points |
| Best-checkpoint test range | 75.30–76.82% | 74.14–75.98% |
| Fixed-epoch-60 test accuracy, mean ± sample SD | 73.34% ± 1.62 points | 71.97% ± 1.94 points |
| Mean best validation accuracy | 74.06% | 72.97% |
| Mean training time per run | 846.25 s | 854.54 s |
| Mean selected layer-4 epoch | 1 | 1 |

All three seeds selected epoch 1 in both arms. The hard-negative
variant's selected-checkpoint test accuracy was lower in each matched
seed: 74.92% vs 75.30% (seed 42), 74.14% vs 75.74% (seed 43), and
75.98% vs 76.82% (seed 44). Its mean selected-checkpoint score was
about 0.94 percentage points lower, and mean runtime was about
8.30 seconds higher.

The control in this matched comparison includes the auxiliary
predictor's training overhead, so its runtime should not be compared
directly with the earlier random-negative experiment that had no
predictor head. The results show that this source-aligned hard-negative
adaptation did not improve utility for the current reduced-width setup;
three seeds do not establish that hard negatives are generally worse.
Neither arm reproduces Hinton's reported accuracy. The historical
random-negative results and archive are retained.

## Environment and reproducibility

- Workspace commit at audit: `e5f8182890088c56d0f53cd570de503b6e04c7ef`; prior local edits were consolidated into the current canonical scripts/results. The verified final snapshot and manifest are at `/tmp/stage-a-safety/stage-a-consolidated-snapshot.tar.gz` (contents listed in its manifest; gzip integrity check passed).
- Ubuntu WSL; Python `3.14.4`; PyTorch `2.14.1+cpu`; torchvision `0.29.1+cpu`; four CPU threads. Environment: `~/venvs/mp-spdz-ml`.
- MNIST files already present in `~/datasets/MNIST/raw`; downloaded data is outside the repository.
- MP-SPDZ commit `113c4e47dd94bca9ecab029d2f0abe649bea5c52`, tag `v0.4.4`; repository has local Player-Data artifacts. `Scripts/mascot.sh tutorial` completed as a **two-party MASCOT** example: expected values matched; output reported about 311.962 MB global traffic, 1.35124 s including preprocessing, and about 1,139 rounds. MASCOT is a malicious, dishonest-majority protocol, not the proposal’s candidate three-party semi-honest setting. This is protocol/tutorial evidence, not ML-training evidence.
- `doc/ml-quickstart.rst` was run successfully at the pinned commit using its documented three-party `ring` command. The first attempt failed with `ModuleNotFoundError: No module named 'numpy'`; using the existing `~/venvs/mp-spdz-ml` interpreter fixed that. Compilation then failed with `clang++: error: no such file or directory: 'release'` because the inherited `DEBUG=release` environment variable was passed to the compiler; unsetting it for the command fixed compilation. The first runtime attempt reported `Cannot access Player-Data/P0.pem` (and P1/P2 equivalents); after the documented `Scripts/setup-ssl.sh 3` setup, the secure run completed all 100 epochs and returned `[0, 1]`. The exact successful command and output are preserved in [the quickstart log](../../results/archive/evidence/stage_a_mp_spdz_ml_quickstart.log), SHA-256 `738e7c80f1e0728aec190a8257be93f8de403b7b331b5c790fe032479fa876f6`. The run reported 44,121 VM rounds, 2.03405 MB estimated online communication, 7.27021 s runtime, and 2.01811 MB global traffic. MP-SPDZ warns the domain is too small for low-round truncation (103 bits would be needed); this is recorded as a limitation, not a failed run.
- A separate primitive smoke program was compiled and run with three-party semi-honest Rep3 over a 64-bit ring, using `-M` to preserve memory order. It passed fixed-point 2×2 matrix multiplication, fixed-point multiplication/truncation, ReLU and comparison, inverse-square-root normalization, `ml.lse_0` softplus, and secure wrong-label wraparound for secret labels/offsets. Exact output is preserved in [the primitive log](../../results/archive/evidence/stage_a_mp_spdz_primitives.log), SHA-256 `62440a63b311d0eaddc3ef36808e7db212a36bd19e55d6e2dc30f648ffe75345`. This is compatibility evidence for these concrete operations under the selected protocol, not end-to-end BP/FF or Stage D operation profiling. MP-SPDZ's module header still warns that its training library is work in progress. The full primitive source was a temporary MP-SPDZ program outside this repository.
- A second isolated Rep3 program tested 3-class `ml.softmax` and the same stable log-sum-exp cross-entropy formula used by MP-SPDZ `ml.MultiOutput`, plus secret random wrong-label offsets. For logits `[2,1,-1]` and secret one-hot target `[1,0,0]`, it returned probabilities `[0.705368, 0.259491, 0.0351257]`, probability sum `0.999985`, and cross-entropy `0.348938`. It also compiled/executed secret label inputs `[0,4,9]`, rejection-sampled offsets `[2,2,7]`, and compare/subtract wraparound results `[2,6,6]`; all three sampler-success/range/wrong-label checks passed. The exact output is in [the remaining MPC log](../../results/archive/evidence/stage_a_mp_spdz_remaining_mpc.log), SHA-256 `408d7cd9af0825bc9ee8511ca27054e8d91e5b2171f352cb8ad06149185cf840`. `sint.get_random_int(4)` uses MP-SPDZ's secret uniform 4-bit source (`Compiler/types.py`/`Compiler/comparison.py`); rejecting values 9–15 and adding 1 gives an exactly uniform offset in 1..9 conditional on acceptance. A fixed 64 draws avoid secret-dependent loop length; probability all 64 are rejected is `(7/16)^64 ≈ 1.1e-23`. The test reveals only a success/range predicate and the test outputs for correctness; production must abort/retry on failure and must not substitute a biased offset. This is not a statistical RNG audit or a cost profile.
- The successful quickstart and primitive program establish coarse compile-time estimates and end-to-end runtime/communication output. MP-SPDZ docs clarify that VM rounds are not exact network rounds. No per-operation runtime counters have been measured for this project.
- Exact successful invocations: `Scripts/setup-ssl.sh 3`; then `env -u DEBUG PATH="$HOME/venvs/mp-spdz-ml/bin:$PATH" Scripts/compile-run.py -E ring foo`. Isolated checks used `env -u DEBUG PATH="$HOME/venvs/mp-spdz-ml/bin:$PATH" Scripts/compile-run.py -M -E ring stage_a_primitives` and `env -u DEBUG PATH="$HOME/venvs/mp-spdz-ml/bin:$PATH" Scripts/compile-run.py -M -E ring stage_a_remaining`. Complete outputs are in the linked evidence logs. Quickstart source is `doc/ml-quickstart.rst`; additional test sources remain under `~/MP-SPDZ/Programs/Source/` in the installed checkout. The tests verify the specific inputs above, not end-to-end project training.
- Exact logged reference commands: `set -o pipefail && PYTHONUNBUFFERED=1 $HOME/venvs/mp-spdz-ml/bin/python src/baselines/bp.py --reference-check --output results/baselines/bp.csv 2>&1 | tee results/archive/evidence/stage_a_bp_reference_audit_v2.log`; and `set -o pipefail && PYTHONUNBUFFERED=1 $HOME/venvs/mp-spdz-ml/bin/python src/baselines/ff.py --reference-check --output results/baselines/ff.csv 2>&1 | tee results/archive/evidence/stage_a_ff_reference_checkpoint_v2.log`. The BP audit rerun restored its complete curve after the earlier baseline rewrite had dropped the first reference epoch rows; its output is in [the BP audit log](../../results/archive/evidence/stage_a_bp_reference_audit_v2.log), SHA-256 `3d5e7f659acbe3fc31c7db126fbad5b30d7d892fc5ce3152ae5713961347c807`. The FF validation-checkpoint rerun is in [the FF checkpoint log](../../results/archive/evidence/stage_a_ff_reference_checkpoint_v2.log), SHA-256 `82ca2af31a5b8e3de0abe06fe251ddaa7c8ab3b497d6559fd1d1aba7d88f5025`. Both append tagged records to their canonical CSV; the new FF ID separately preserves fixed-60 and validation-selected test results.
- Reproduce baselines after activating the environment using `python src/baselines/bp.py` and `python src/baselines/ff.py`. Syntax compilation succeeded with `python -m py_compile src/baselines/bp.py src/baselines/ff.py`.

## Final protocol and threat-model decision

The selected protocol for subsequent secure implementation work is:

| Item | Final Stage A decision |
|---|---|
| Framework | MP-SPDZ `v0.4.4`, commit `113c4e47dd94bca9ecab029d2f0abe649bea5c52` |
| Protocol | Three-party Rep3 Ring |
| Parties | 3 |
| Security model | Semi-honest |
| Adversary assumption | Honest majority; at most one corrupted party |
| Observed ring size | 64 bits |
| Fixed point observed | `sfix.set_precision(16, 31)` |
| Truncation observed | Probabilistic truncation; no deterministic rounding enabled |

This is a protocol-selection decision based on existing feasibility evidence.
It does not mean that secure BP or FF training has been implemented or
validated. The two-party MASCOT tutorial remains separate historical
framework evidence and is not the selected project protocol.

## Secure-operation compatibility inventory

| Required operation | Evidence/status | Stage A classification |
|---|---|---|
| Fixed-point multiplication and truncation | `sfix` primitive execution and tiny BP logs | Direct framework/API support; workload range validation is future implementation work |
| Matrix multiplication and dot products | 2x2 fixed-point primitive output | Direct arithmetic support; scale-up cost not measured |
| ReLU and comparisons | Active/inactive ReLU checks | Direct `ml.relu`/comparison support |
| L2 normalization and inverse square root | Primitive output `inverse_sqrt` and normalized values | Available primitive support; numerical behavior is only isolated-test evidence |
| Softplus / `ml.lse_0` | Primitive output | Available API support |
| Softmax and cross-entropy | Three-class isolated test | Available API/composition support |
| Secret labels and incorrect-label generation | Secret-label wraparound and rejection-sampling test | Available low-level secret integer/comparison support; production sampling must preserve the tested unbiased construction |
| Layer-wise FF local objectives and freezing | No project FF MPC program | Requires project-specific fixed-point program composition; not implemented in Stage A |
| Full secure BP/FF training | No project training program | Outside the Stage A gate |

The inventory distinguishes API/source support from isolated execution. The
isolated tests do not establish end-to-end secure neural-network training.
Where the MP-SPDZ ML API cannot express the project layer/update path directly,
subsequent work will require custom fixed-point composition rather than a claim
that the framework already provides the complete model.

## Available counters and known limitations

The preserved compile/runtime logs expose integer inputs, triples, simple
multiplications, probabilistic truncations, bit triples, opens, VM rounds,
estimated online communication, total communication, and wall-clock runtime.
The quickstart, for example, reports 44,121 VM rounds, 2.03405 MB estimated
online communication, 7.27021 seconds, and 2.01811 MB global traffic. These
are whole-program feasibility observations. MP-SPDZ documents that VM rounds
are not exact network-round measurements, and no per-operation Stage D profile
has been performed.

The observed 64-bit ring and `sfix.set_precision(16,31)` generated two classes
of low-round truncation warning:

- `47 + 40 = 87` bits for the tiny BP fixed-point multiplication path.
- `63 + 40 = 103` bits for the broader primitive/ML paths.

The 47- and 63-bit values are the largest truncation input widths observed in
the respective schedules; 40 is the configured statistical truncation margin.
The existing computations completed. These messages mean that the low-round
truncation optimization was unavailable in the observed 64-bit domain; they
are not execution failures. The runs used Rep3 probabilistic truncation, not
deterministic nearest rounding.

Stage A intentionally does not remediate these warnings. Ring-size changes,
precision changes, deterministic rounding, workload-specific overflow and
repeated-update validation, and full BP/FF numerical stability belong with the
actual secure implementation and are not Stage A closure requirements.

## Closure decision

Stage A is **COMPLETE as of 2026-11-02**. The project accepts the corrected
standardized two-layer Benchmark v1 and the bounded source-aligned BP/FF
reference evidence under the reduced computational configuration. The evidence
does not claim exact paper reproduction: the source-aligned FF check uses four
128-unit layers rather than four 2,000-unit layers, uniform random incorrect
labels rather than neutral-pass hard negatives, and reports a material
accuracy gap. The BP 20-epoch convergence evidence is sufficient for this
feasibility decision; no optional 30-epoch extension is required.

## Data, model, and evaluation protocol

The legacy code used a single `torch.Generator(seed=42)` to draw the 60,000-example training permutation, then a 10,000-example test permutation, selecting the first 1,000 test indices. The revised scripts reconstruct that sequence and exclude those previously exposed 1,000 test examples. They use positions 1,000–1,999 of the same official-test permutation as a fresh shared final-test subset. The official training permutation supplies 5,000 train and the following 1,000 validation examples; assertions check disjoint train/validation indices. Test data is from the separate official test split. No validation/tuning decision uses the fresh test subset.

BP uses `784-128-128-10`, ReLU, 118,282 parameters, cross-entropy, SGD `lr=0.05`, batch 128, five epochs. The checkpoint is chosen by validation accuracy; test is evaluated once after selection. FF uses `784-128-128` local layers, ReLU, L2 normalization between layers, 116,992 parameters, batch 128, five greedy epochs/layer, and per-layer SGD `lr=0.01`. The original FF baseline used threshold 2.0; after a validation-only grid, the revised configuration uses threshold 3.0. Parameter counts differ by about 1.1%. FF trains positive/correct-label and negative/wrong-label passes with local softplus losses over squared-activation goodness. Wrong labels are sampled uniformly from the nine incorrect classes using a nonzero random offset; this is a reasonable simple negative construction, not a hard-negative scheme. Prior layers are run under `no_grad` and are not updated during later-layer training. Candidate-label inference evaluates cumulative goodness across both layers and, as a separately reported ablation, second-layer goodness only. Validation selects checkpoint and inference variant; final test reports both variants and the validation-selected variant.

Utility uses five seeds (42–46); timing uses three repetitions per seed. Repetitions reset the same seed/configuration, so they characterize runtime variability, not independent utility variation. CSV `training_seconds` aggregates are **per-run sums**, not seconds per epoch. BP sums five `epoch_seconds` values; its timer includes that epoch’s validation pass. FF sums ten `train_seconds` values (five local-training epochs for each of two layers); it excludes validation and test inference. Consequently these per-run timings have different scopes and must not be read as a like-for-like speed comparison. The individual epoch times remain in the epoch rows. Neither value is full process wall time including data loading and final testing.

## Plaintext results and historical records

The BP canonical CSV contains its 75 project-baseline epoch/15 summary/1 aggregate rows, the retained earlier reference summaries, and 60 complete `hinton_bp_audit_v2` epoch rows with three summaries and one aggregate. The earlier reference summaries remain but their epoch rows were lost in a subsequent baseline rewrite; the v2 audit restores complete per-epoch evidence. The current FF canonical CSV contains 3,453 data rows: the 557 baseline/diagnostic/selected rows, 1,440 `hinton_ff_checkpoint_v2` reference rows, 1,440 controlled negative-comparison rows, and their summaries/aggregates. The separate `results/archive/ff_results1.csv` retains the 557-row baseline-only output. The original threshold-2.0 FF baseline CSV remains at `results/archive/` with SHA-256 `818f28b3ad6101a4d61f75cb38208480b5628995fe1d58f34e12ff97f46bd2aa`. Reference and comparison records are retained in the canonical CSV; no duplicate algorithm scripts or result files were created. Prior preliminary smoke results used the old 1,000 official-test examples and are not final-test evidence; those indices were reconstructed and excluded from the reference test subsets too.

The verified baseline and revised runs (five distinct seeds, three timing repetitions per seed) are summarized below; the canonical CSV retains both configurations, all 15 run summaries per configuration, and validation-only diagnostics. Utility aggregates use one deterministic score per distinct seed; timing aggregates use all 15 runs. Repetitions repeat the same seed and configuration, so they characterize runtime variability and are not independent accuracy trials.

- BP best validation accuracy: **76.64% ± 1.76 points**, range **74.70–78.50%**; final-test accuracy: **75.96% ± 2.89 points** (sample SD across five seeds), range **71.70–78.40%**. Its 15 per-run `training_seconds` values are the sum of five epoch timers, each including validation: mean **1.830 s**, sample SD **0.121 s**, range **1.706–2.193 s**.
- FF baseline (threshold 2.0): validation **61.64% ± 3.22**, test **59.76% ± 3.18** (sample SD over five seeds), test range **55.50–62.90%**. Second-layer-only test accuracy averaged **6.32% ± 1.93**. Its 15 per-run times sum the ten local-training epoch timers (two layers × five epochs), excluding validation/test: mean **3.658 s**, sample SD **0.338 s**, range **3.163–4.216 s**.
- FF revised (threshold 3.0 selected on validation): validation **62.78% ± 3.49**, test **60.64% ± 3.27**, test range **56.90–64.50%**. Second-layer-only test accuracy averaged **4.60% ± 2.06**. Its 15 per-run ten-epoch training sums are mean **3.431 s**, sample SD **0.207 s**, range **3.085–3.804 s**. All five seeds selected cumulative all-layer goodness. The revised utility gain is modest, and the second-layer-only ablation remains poor. Because BP and FF timers include different work, these totals do not support a speedup comparison; per-epoch values are recorded separately in the CSVs.

These figures were recalculated from the canonical CSVs: utility uses one summary per distinct seed (five seeds), while timing uses all 15 repetitions. I summed the underlying epoch timers by seed/repetition; all 45 per-run sums (15 each for BP, FF threshold 2.0, and FF threshold 3.0) exactly match their seed-summary `training_seconds` values. Repeated summaries have the same per-seed utility. The previous-draft totals of 1.884 s for BP and 3.310 s for FF do not match the canonical run rows and are superseded by 1.830 s (BP), 3.658 s (FF threshold 2.0), and 3.431 s (FF threshold 3.0). The CSV aggregate rows agree to the displayed precision.

### Validation-only threshold diagnostic

Before changing the canonical output, the original result file was byte-for-byte copied to `results/archive/ff_baseline_before_threshold_diagnostics.csv` and verified with `cmp` and SHA-256. The diagnostic trained the same 784-128-128 network for the same five epochs/layer on the same fixed training/validation split for thresholds **0.5, 1.5, 2.0, and 3.0**, using seeds 42–46. No test DataLoader was created in these diagnostic runs. Each threshold’s criterion was mean best validation accuracy over the five seeds; inference variant and checkpoint were also selected on validation.

| Threshold | Mean best validation accuracy | Sample SD | Range | Validation-selected inference |
|---:|---:|---:|---:|---|
| 0.5 | 52.96% | 2.31 | 50.5–55.5% | all-layer, all 5 seeds |
| 1.5 | 60.80% | 2.88 | 56.5–63.4% | all-layer, all 5 seeds |
| 2.0 | 61.64% | 3.22 | 57.6–65.1% | all-layer, all 5 seeds |
| 3.0 | 62.78% | 3.49 | 59.2–66.9% | all-layer, all 5 seeds |

Threshold 3.0 was highest; its per-seed validation accuracies were 62.3%, 59.2%, 66.9%, 59.7%, and 65.8%, higher than threshold 2.0 for each seed. All selected all-layer inference and layer-2 epoch 1. After this choice was fixed, the selected configuration was evaluated on the held-out test subset across the five seeds, with three repeated timing runs.

This is a small validation search over four values and five seeds, not a broad hyperparameter optimization. The test result was not part of the threshold decision. The archive preserves the prior complete baseline CSV; the canonical CSV marks its rows as `baseline_*`, records the diagnostic rows separately, and stores the revised run as `epoch`, `seed_summary`, and `aggregate`.

Measured parameter counts: BP **118,282**, FF **116,992** (difference **1.09%** relative to BP).

At threshold 2.0, across layer-2 training epochs/seeds, positive/negative goodness averaged **1.531/1.537** (gap **−0.006**); 19.48% of positives were above the threshold and 78.87% of negatives below. At threshold 3.0 the corresponding means were **2.364/2.378** (gap **−0.014**), with 24.8% of positives above and 72.2% of negatives below. Raising the threshold increased the all-layer validation score slightly, but did not improve positive/negative separation; second-layer-only validation averaged only **4.74%** at its best epoch for threshold 3.0 and fell to about **3.5–3.6%** later. Thus the revised threshold improves the selected validation utility metric modestly while leaving the local layer-2 objective behavior unresolved. It is not evidence that threshold adjustment solved the FF representation problem.

Validation accuracy peaked at layer-2 epoch 1 for all five seeds at thresholds 1.5, 2.0, and 3.0, then declined for the cumulative goodness method. Extending training beyond the existing five epochs is therefore not justified by these observations; longer training was not tested.

The earlier threshold-2.0 smoke result had layer-1 goodness near 1.86/1.84 initially and roughly 2.04/1.85 by epoch 5; layer 2 began near 0.436/0.436. The current controlled runs confirm that cumulative goodness is dominated by useful first-layer label information while the second layer fails to separate positive from negative examples. It is not claimed to reproduce Hinton’s reported experiments: this is a small, two-dense-layer adaptation with uniform wrong-label sampling, L2 normalization, and SGD. Hinton’s paper motivates two forward passes, per-layer goodness objectives, and squared-activity goodness; see [the original paper](https://arxiv.org/abs/2212.13345). Its supervised MNIST example reports four 2,000-unit hidden layers trained for 60 epochs and 1.36% test error, using hard negatives selected from a neutral-label pass and inference by accumulating goodness from all but the first hidden layer. That is a reference description, not a pass/fail target for this different model. Keller and Sun’s [paper](https://proceedings.mlr.press/v162/keller22a.html) uses a different convolutional model and fixed-point emulator; its accuracy does not transfer as a target to this plaintext MLP. The exact label corruption, normalization, architecture, and candidate inference here are project-specific adaptations.

## Stage A MP-SPDZ operation inventory and compatibility checks (not profiling)

The inventory was cross-checked against the installed MP-SPDZ commit above: protocol matrix in `~/MP-SPDZ/README.md`; arithmetic ML APIs and their caveat in `Compiler/ml.py`; fixed-point implementation in `Compiler/types.py`; nonlinear routines in `Compiler/mpc_math.py`; and ML/counter examples in `doc/ml-quickstart.rst` and `doc/troubleshooting.rst`. Source inspection is distinguished from the compatibility checks below, which ran under the selected protocol. No per-operation cost measurements are reported.

The repository protocol table lists arithmetic semi-honest honest-majority
candidates. The quickstart and primitive checks used **Rep3 ring with three
local parties**, matching the proposal's candidate party count and
semi-honest honest-majority model. Stage A selects this configuration based on
the successful feasibility evidence. The earlier MASCOT tutorial used a
different security model and remains separate evidence.

| Required operation | What the installed source/docs show | Stage A support assessment | Cost or unresolved point for Stage D |
|---|---|---|---|
| Dense matrix products | `ml.Dense`, fixed-point matrices, and the documented ML quickstart are present. The primitive run computed a secret 2×2 product with expected values. | **Compiled and ran** under selected 3-party semi-honest Rep3 ring for this minimal case. | No per-operation Stage D measurement. This does not establish project-scale feasibility. |
| ReLU and comparisons | `ml.relu` uses secret comparison/selection; quickstart and primitive checks executed comparison paths. | **Compiled and ran** for tested inputs under selected Rep3 ring. | Boundary/range coverage and full-model behavior remain untested; profiling belongs to Stage D. |
| Backward products and SGD | The quickstart trained its documented logistic-regression model for 100 epochs; `ml.SGD` and dense gradients are used. | **Framework training path verified** for logistic regression. Project BP architecture/training remains untested. FF has no dedicated training method in `ml`; local FF gradients/updates still need custom layer logic or direct `sfix` code. | No project BP or FF training. Per-operation Stage D measurement is deferred. |
| Softmax/cross-entropy | The isolated test used secret logits/target, `ml.softmax`, and MP-SPDZ's stable log-sum-exp cross-entropy formula from `ml.MultiOutput`. | **Compiled and ran** under selected Rep3; probability sum was 0.999985 and cross-entropy 0.348938 for the known input. | The 64-bit ring emitted the low-round truncation-domain advisory; numeric range/precision beyond the known test vector remains unverified. Per-operation profiling belongs to Stage D. |
| Squared-activation goodness and reduction | Fixed-point products/reductions are available; the primitive smoke program computed squared norm as part of normalization. | **Squared sum compiled and ran** as a component of the normalization check; standalone FF goodness/loss integration remains untested. | Count secure squares and reduction work during Stage D. |
| FF local softplus objective | `ml.lse_0` implements log-sum-exp-zero / softplus from fixed-point exp/log; an approximation also exists. | **Exact `ml.lse_0` compiled and ran** for a test input. FF loss/gradient integration remains unimplemented and untested. | Exact vs approximate nonlinear cost and utility comparison are Stage D/E questions. |
| L2 normalization between FF layers | `mpc_math.InvertSqrt` and `ml.LayerNorm` are present; `LayerNorm` is not equivalent to this baseline’s L2 normalization. | **Sum-of-squares + inverse square root + scaling compiled and ran** on a nonzero test vector. Exact project epsilon/range behavior remains untested. | Division/inverse-square-root/truncation cost and numerical error are unmeasured. |
| Fixed-point multiplication, truncation and conversions | `sfix` is implemented (default 16 fractional, 31-bit precision); the primitive run used products and truncation. | **Multiplication/truncation compiled and ran** under the selected Rep3 ring. Binary ML APIs are unsupported by `ml.Optimizer`. | Quantization error and alternate-domain conversions remain untested. |
| Secret labels and negative-example generation | `sint.get_random_int(k)` yields a secure uniform k-bit integer; `sint % c` only implements powers of two. The isolated check uses 64-draw rejection from `[0,15]` to `[0,8]`, maps to `[1,9]`, then compares/subtracts ten for labels in `[0,9]`. | **Secure randomness, conditional uniformity, secret labels, range, and wrong-label construction compiled and ran** for test inputs. The cap failure probability is about `1.1e-23` per sample; the test fails the aggregate check rather than substituting a biased value. This is not a statistical RNG audit. | A 64-draw fixed bound is correctness-oriented and potentially costly; do not profile it here. Decide and profile the production sampler in Stage D. |
| Candidate-label inference | Existing ML APIs compose forward layers; the FF inference loop is public over ten labels. | **Likely expressible** as repeated secure forwards; no secure run. | It repeats the network per label (or requires an auxiliary classifier); include inference separately in Stage D. |

**Unsupported status:** no required arithmetic operation has been proven unsupported for an arithmetic candidate. The confirmed library boundary is that MP-SPDZ’s high-level ML optimizer does not support binary circuits, and its own header says training is work in progress with only consecutive-layer training tested. FF is not a built-in algorithm. This means custom FF code and arithmetic protocol verification are needed; it does not justify calling the listed primitives unsupported.

**Counters:** documentation and the quickstart/primitive outputs show coarse compile-time estimates (e.g. VM rounds, triples, matrix-multiplication counts) and runtime time/communication output. The quickstart reported 44,121 VM rounds; the arithmetic primitive program 192; the softmax/sampler program 371. The separate MASCOT tutorial printed about 1,139 rounds. MP-SPDZ documentation warns VM rounds are not exact network rounds and protocol calls may expand. These are whole-program smoke outputs; no per-operation timers/counts for BP/FF have been collected. Stage D will profile operations after a project secure program exists; Stage A is limited to identifying exposed counters and mapping operations.

## Gate checklist and remaining work

- MP-SPDZ example execution: **COMPLETE**; the documented ML quickstart securely trained logistic regression under selected three-party Rep3 ring. This is framework validation, not a project BP/FF secure baseline.
- Plaintext BP reference behavior: **COMPLETE as a project-sized, source-aligned reference check; not a paper reproduction**. The complete 20-epoch curves show declining training/validation loss and improving validation accuracy on all three seeds. Test accuracy is 97.03% mean on 9k clean examples; the paper's ~1.4% typical error is descriptive, not a guaranteed target for this narrower and incompletely specified configuration.
- Plaintext FF reference behavior: **COMPLETE as bounded source-aligned evidence**; the reduced-width random-versus-hard-negative comparison is complete, but neither variant reproduces Hinton's reported accuracy. The deviation and negative result are documented; no additional experiment is required for Stage A.
- MPC operation mapping and compatibility checks: **COMPLETE for the Stage A inventory**; the operation map distinguishes source-only claims from verified Rep3 primitives, and matrix products, ReLU/comparison, fixed-point truncation, inverse-square-root normalization, softplus, multiclass softmax/cross-entropy, and private uniform nonzero wrong-label construction compiled and ran in isolated checks. Candidate-label inference remains API-composable but was not directly run. The selected protocol is documented; no per-operation cost profiling is required until Stage D.
- Reproducibility and canonical artifact integrity: **COMPLETE**; both reference configurations have tagged records in the two canonical CSVs, prior rows remain present, the archived threshold-2.0 CSV hash is unchanged, and exact outputs are retained in `results/archive/evidence/`.

## Remaining Stage A blockers and closure evidence

1. **No Stage A blockers remain.** The bounded 4×128 FF comparison is accepted for the project reference gate despite its documented gap from Hinton's full-width experiment.
2. **BP convergence plateau:** the optional 30-epoch extension is explicitly not required; the 20-epoch convergence evidence is sufficient for this feasibility decision.

The controlled hard-negative comparison addressed the major negative-sampling
difference at reduced width, but it did not improve accuracy or close the gap
to Hinton's reported result. Stage A is **COMPLETE (2026-11-02)** because the
proposal gate requires bounded reference evidence and an operation feasibility
inventory, not exact paper reproduction or secure training. No further FF
experiment or full-width run is required. No Stage D per-operation cost has
been measured or claimed.
