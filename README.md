# Biologically Plausible Secure MPC for Privacy-Preserving AI Training I

This graduation project compares plaintext Backpropagation (BP) and Forward-Forward (FF) on MNIST before evaluating secure training in later stages. Stage A covers reproducible baselines, framework feasibility, and an MPC operation inventory.

## Baseline scripts and environment

The only active implementations are `src/bp_mnist.py` and `src/ff_mnist.py`. They use the existing CPU environment and downloaded MNIST under `~/datasets/MNIST`.

```bash
source ~/venvs/mp-spdz-ml/bin/activate
python src/bp_mnist.py
python src/ff_mnist.py
```

Results go to `results/bp_mnist_results.csv` and `results/ff_mnist_results.csv`. The FF script first runs a four-value threshold diagnostic on validation data only, then evaluates the selected configuration on the final test set. The preserved threshold-2.0 baseline is in `results/archive/ff_mnist_baseline_before_threshold_diagnostics.csv`; threshold 3.0 is the validation-selected configuration in the canonical CSV. Both scripts use the same deterministic 5,000/1,000 official-training split and 1,000-example final-test subset. Previously used test indices are excluded; see [the Stage A report](docs/stage_a_report.md).

## Current status

Stage A is **IN PROGRESS**. The documented MP-SPDZ ML quickstart and isolated arithmetic primitive checks have run under a provisional three-party Rep3 ring setup; this is not secure project training. Multi-seed plaintext baselines, the validation-only FF threshold diagnostic, and named full-data source-aligned reference checks are recorded in the canonical CSVs. The reference checks do not claim paper reproduction: BP averages 97.03% test accuracy; the reduced-width FF check averages 73.34%. See the report for Hinton's reference behavior, configuration differences, and remaining gates. No secure BP/FF training or Stage B work has started.
