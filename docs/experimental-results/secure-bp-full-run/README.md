# Secure BP full-run evidence

This directory documents the completed seed-42 Secure Backpropagation (Secure
BP) matched matrix experiment. It is an evidence record, not an execution
script.

## Experiment and evidence locations

- Source: `mpc/programs/bp/bp_matched_matrix.mpc`
- Configuration: `configs/bp_matched.yaml`
- Input bundle: `~/secure-bp-runs/SECURE-BP-MATCHED-v1_seed42_corrected/`
- Full-run log: `~/secure-bp-runs/SECURE-BP-MATCHED-MATRIX-v1_full_seed42_run_20261115_234946/full.log`
- Pre-run Player-Data backup: `~/secure-bp-runs/player-data-backup_before_full_20261115_234946`
- Plain BP source and recorded results: `src/baselines/bp.py` and
  `results/baselines/bp.csv`

## Verified

- The Secure BP source SHA-256 matches the expected digest.
- All three input-file SHA-256 values match the expected digests.
- The input verification summary reports `all_pass: true`, including token
  consumption, ordering, ownership, parameter layout, and metadata hashes.
- The log contains five training-loss lines, validation and test metrics, MP-SPDZ
  time, communication, rounds, memory, the truncation-domain warning, and exit
  status 0.
- The backup exists and contains 33 files totaling 40,492,581 bytes. Its
  contents were not read.
- The Plain BP implementation and seed-42 rows in `results/baselines/bp.csv`
  provide repository evidence for the reference configuration and values.

## Not established by this audit

This is one run with seed 42. It does not establish performance across seeds,
statistical confidence, or a formal security proof. The low-round truncation
domain warning remains an implementation limitation requiring further analysis.
The exact scope relationship between MP-SPDZ's reported time and the
wall-clock measurement is also not normalized beyond what the log states.

## Future controlled reproduction

In a separately controlled environment, pin the documented MP-SPDZ commit and
configuration, verify the source and input hashes before execution, preserve
the Player-Data state, run the three Rep3 Ring parties through the project's
approved MP-SPDZ procedure, and archive the complete log and resource
measurement. Compare the resulting metrics and hashes with this directory.
Do not reproduce the experiment in place or overwrite this evidence.

