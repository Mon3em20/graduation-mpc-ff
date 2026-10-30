# Project progress

Progress is mapped to proposal stages while the repository remains organized
by function.

## Stage A — Plaintext and framework feasibility

- **Objective:** establish reproducible BP/FF references and map MPC support.
- **Deliverables:** baselines, source-aligned checks, MP-SPDZ feasibility logs.
- **Verified work:** canonical CSVs, reference records, quickstart and
  primitive evidence are preserved in `results/archive/`.
- **Status:** In Progress.
- **Remaining/blockers:** supervisor decision on the bounded FF reference and
  final protocol; no secure project training.

## Stage B — Tiny secure BP

- **Objective:** validate fixed-point BP mechanics and ReLU edge behavior.
- **Deliverables:** isolated 2-2-1 programs and independent verification.
- **Verified work:** preserved outputs pass the two checkers.
- **Status:** Completed for microtests; not end-to-end training.
- **Remaining/blockers:** implement and validate secure MNIST BP.

## Stage C — Secure training

- **Objective:** implement secure BP and FF training under the common setting.
- **Deliverables:** scalable programs and utility evidence.
- **Status:** Not Started.
- **Remaining/blockers:** protocol confirmation and implementation.

## Stage D — Cost profiling

- **Objective:** measure communication, rounds, preprocessing, and runtime by operation.
- **Verified work:** framework counters exist in feasibility evidence.
- **Status:** Not Started.
- **Remaining/blockers:** project-scale BP/FF programs are required; quickstart counters
  are not Stage D profiling.

## Stage E — MPC-aware FF modification

- **Objective:** investigate one evidence-driven FF modification.
- **Verified work:** controlled hard-negative comparison is recorded.
- **Status:** In Progress.
- **Remaining/blockers:** select and evaluate a modification under measured MPC costs.

## Stage F — Final comparison

- **Objective:** compare BP and FF fairly and report conclusions.
- **Status:** Not Started.
- **Remaining/blockers:** stages C–E and final supervisor decisions.
