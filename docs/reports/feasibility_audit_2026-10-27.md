# Stage A audit — 2026-10-27

## Scope and safety

This audit inspected the repository source, all result CSVs, the archive, the
Stage A report, README, and available evidence logs. No training, benchmark, or
expensive experiment was run. Historical CSVs were not edited, renamed,
deleted, merged, or regenerated. The ordinary FF output and the separate
`*_results1.csv` paths remain distinct.

Before source or report edits, verified backups were created at
`/tmp/stage-a-safety-20261027T103250Z/`. The backup manifest was checked with
`sha256sum -c`; it contains the pre-edit versions of `src/baselines/ff.py`,
`src/baselines/bp.py`, and `docs/reports/feasibility.md`.

## Changes made

- **VERIFIED:** [`src/baselines/ff.py`](../src/baselines/ff.py) now writes through an
  atomic temporary-file plus `os.replace` helper, flushing and fsyncing before
  replacement. Failed replacement leaves the original output intact.
- **VERIFIED:** The ordinary FF path now retains existing records outside its
  explicitly regenerated baseline/diagnostic record types. This preserves
  `reference_ff_*`, `negative_comparison_*`, and other historical records.
- **VERIFIED:** `neutral_softmax_hard` now rejects a missing
  `auxiliary_predictor=True` before model/data work. Unknown negative strategies
  are also rejected early.
- **VERIFIED:** [`tests/test_ff_persistence.py`](../tests/test_ff_persistence.py)
  uses temporary synthetic CSVs only. It covers unrelated-record preservation,
  separate output paths, failed-write safety, and invalid configurations.
- **VERIFIED:** README wording now reflects that the hard-negative comparison
  is recorded rather than still pending. The Stage A report remains
  explicitly in progress and retains the pending reference-gate decision.

## CSV integrity

Hashes were captured before edits and after tests/documentation edits. They are
identical:

| CSV | Before SHA-256 | After SHA-256 | Result |
|---|---|---|---|
| `results/baselines/bp.csv` | `a7e8c2b0e31daebd9fc46088925dfd1c72fa650cbac256264d9129a1683fdb67` | `a7e8c2b0e31daebd9fc46088925dfd1c72fa650cbac256264d9129a1683fdb67` | VERIFIED |
| `results/archive/bp_results1.csv` | `d8d7e9acec5eabcb1c9786193667ed345aa210eb1f233e679a6a2cb0910dfed5` | `d8d7e9acec5eabcb1c9786193667ed345aa210eb1f233e679a6a2cb0910dfed5` | VERIFIED |
| `results/baselines/ff.csv` | `7f0567993363bff64317f29e1b397442ffd960d4ebec2f6211c25a41bd81eb70` | `7f0567993363bff64317f29e1b397442ffd960d4ebec2f6211c25a41bd81eb70` | VERIFIED |
| `results/archive/ff_results1.csv` | `e12157e910a870795326566438a717a745b1e8e3a35b76d3aa7734d43c8bb54e` | `e12157e910a870795326566438a717a745b1e8e3a35b76d3aa7734d43c8bb54e` | VERIFIED |
| `results/archive/ff_baseline_before_threshold_diagnostics.csv` | `818f28b3ad6101a4d61f75cb38208480b5628995fe1d58f34e12ff97f46bd2aa` | `818f28b3ad6101a4d61f75cb38208480b5628995fe1d58f34e12ff97f46bd2aa` | VERIFIED |

The canonical FF CSV currently contains 3,453 data rows, including 1,440
`reference_ff_epoch`, 1,440 `negative_comparison_epoch`, and 500 threshold/
baseline/revised epoch rows, six reference summaries, six negative-comparison
summaries, and their aggregate/selection records. The separate
`ff_mnist_results1.csv` contains 557 data rows and has no reference or
negative-comparison records. This difference is intentional and both files
remain separate.

## Evidence audit

| Finding | Classification | Evidence |
|---|---|---|
| BP reference audit has three seeds, 20 epochs, and 60 per-epoch rows | VERIFIED | `results/baselines/bp.csv`; `results/archive/evidence/stage_a_bp_reference_audit_v2.log` |
| BP reference test scores are 96.89%, 97.01%, 97.18% | VERIFIED | CSV seed summaries and audit log |
| FF checkpoint reference has three seeds and validation-selected checkpoints | VERIFIED | `hinton_ff_checkpoint_v2` rows and `stage_a_ff_reference_checkpoint_v2.log` |
| Controlled hard-negative comparison has matched three-seed arms | VERIFIED | `hinton_ff_negative_comparison_v1` rows and `stage_a_ff_negative_comparison_v1.log` |
| Hard negatives improved the reduced-width reference | MISMATCH | Hard-negative best-checkpoint scores (74.92%, 74.14%, 75.98%) are below the uniform control (75.30%, 75.74%, 76.82%) for every seed |
| MP-SPDZ quickstart and primitive checks prove secure project BP/FF training | NOT VERIFIABLE | Logs prove compatibility checks and a quickstart, not project secure training |
| Report's stated historical counts describe the current canonical FF CSV | MISMATCH | Current CSV has 3,453 rows and includes negative-comparison records; older report prose describes an earlier 557+720+720 composition |
| Full-width four-layer, 2,000-unit FF reference is required now | PENDING SUPERVISOR | No experiment was run; the project still lacks an explicit acceptance decision |

The available evidence supports keeping Stage A **IN PROGRESS**. It does not
support declaring paper reproduction, secure BP/FF training, or Stage A
completion.

## Validation executed

- `$HOME/venvs/mp-spdz-ml/bin/python -m unittest discover -s tests -v` —
  **4 tests passed**.
- `$HOME/venvs/mp-spdz-ml/bin/python -m py_compile src/baselines/ff.py
  tests/test_ff_persistence.py` — **passed**.
- An initial test attempt with the repository `.venv` failed because that
  environment has no PyTorch; the documented project interpreter was then used
  successfully. No dependency was installed and no training command was run.

## Decision still required

The supervisor must decide whether to accept the reduced-width four-layer,
128-unit FF reference as the Stage A plaintext reference, and must specify the
required accuracy tolerance (or explicitly require a different bounded
reference). Until that decision is recorded, Stage A remains in progress and
Stage B must not start.
