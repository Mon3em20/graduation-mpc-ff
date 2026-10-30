# Biologically Plausible Secure MPC for Privacy-Preserving AI Training

This research repository compares plaintext Backpropagation (BP) and
Forward-Forward (FF) under a common MNIST methodology, then investigates
secure training and one MPC-aware FF modification. The research question is:
how do secure BP and secure FF compare in utility, computation, communication,
and protocol overhead under the same MPC setting?

## Repository

```text
configs/              Reference experiment and MPC settings
docs/                 Proposal, design, decisions, and progress reports
src/baselines/        Plaintext BP and FF implementations
mpc/programs/bp/      Implemented Tiny BP MPC microtests
scripts/               MPC runner and independent verifiers
tests/                Python regression and numerical-reference tests
results/baselines/    Canonical BP and FF CSVs
results/mpc/          Preserved Tiny BP execution logs
results/archive/      Historical CSVs and raw feasibility evidence
```

## Setup

The Python dependencies are listed in `requirements.txt`. MP-SPDZ is built
separately and must not be installed as a pip dependency. The baseline scripts
expect MNIST at `~/datasets/MNIST` and use a CPU PyTorch environment.

```bash
python -m pip install -r requirements.txt
```

## Plaintext baselines

These commands run the existing 5,000/1,000/1,000 project baselines and write
to the canonical CSVs. They may take several minutes and download nothing.

```bash
python src/baselines/bp.py
python src/baselines/ff.py
```

The source-aligned 20/60-epoch checks are explicit opt-in commands:
`python src/baselines/bp.py --reference-check` and
`python src/baselines/ff.py --reference-check`. Historical records are
preserved in the canonical files; no claim of paper reproduction is made.

## MPC

Set `MP_SPDZ_ROOT` to an external MP-SPDZ checkout and run an authoritative
source through the safe runner:

```bash
MP_SPDZ_ROOT="$HOME/MP-SPDZ" ./scripts/run_mpc.sh bp_tiny.mpc
MP_SPDZ_ROOT="$HOME/MP-SPDZ" ./scripts/run_mpc.sh relu_edge.mpc
```

The runner uses three-party semi-honest Rep3 Ring with the verified `-M -E
ring` flags and stores logs in `results/mpc/bp/`. This is a provisional
configuration. The programs are isolated fixed-point correctness checks, not
secure MNIST BP or FF training.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/verify/verify_bp.py results/mpc/bp/bp_tiny.log
python scripts/verify/verify_relu.py results/mpc/bp/relu_edge.log
```

The two checkers compare deliberately revealed diagnostics against independent
plaintext references with tolerance `2e-4`. Results and raw feasibility logs
are retained under `results/`; `results/benchmarks/` is reserved for future
project-scale profiling.

## Current status and limitations

Stage A plaintext/framework feasibility is **In Progress**. Tiny BP numerical
microtests are complete, but they do not prove end-to-end secure training.
Secure BP/FF training and per-operation Stage D profiling are **Not Started**.
The Rep3 Ring choice remains provisional pending supervisor confirmation.
MASCOT evidence is a separate tutorial and is not project training evidence.
