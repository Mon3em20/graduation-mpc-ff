# Project progress

Progress is mapped to proposal stages while the repository remains organized
by function.

## Stage A — Plaintext and framework feasibility

- **Objective:** establish reproducible BP/FF references and map MPC support.
- **Deliverables:** baselines, source-aligned checks, MP-SPDZ feasibility logs.
- **Verified work:** canonical CSVs, reference records, quickstart and
  primitive evidence are preserved in `results/archive/`.
- **Status:** **Complete (2026-11-02).**
- **Closure:** the bounded Benchmark v1 and source-aligned BP/FF references
  are accepted; the secure-operation inventory, primitive mapping, counters,
  and final Rep3 Ring threat model are documented. This does not claim secure
  project training or Stage D profiling.

## Stage B — Tiny secure BP

- **Objective:** validate fixed-point BP mechanics and ReLU edge behavior.
- **Deliverables:** isolated 2-2-1 programs and independent verification.
- **Verified work:** preserved outputs pass the two checkers.
- **Status:** Bounded Secure BP A1-relu-corrected is complete as an engineering run;
  Stage B remains **PARTIAL** for the final security claim.
- **Verified work:** the corrected strict ReLU boundary was executed inside
  pinned three-party Rep3 MPC with `[-1, 0, 1]` and returned
  `[0, 0, 1]`; the corrected 90-update A1-relu-corrected run completed and
  passed aggregate verification. A0-original remains preserved as historical
  evidence.
- **Remaining/blockers:** the probabilistic-truncation claim is limited to
  the qualified stand-alone modified functionality described in the security
  memo; no zero-leakage or UC claim is made. Formal all-workload overflow
  proof is also unavailable.

## Stage C — Secure training

- **Objective:** implement secure BP and FF training under the common setting.
- **Deliverables:** scalable programs and utility evidence.
- **Status:** Bounded Secure BP prototype verified; stage remains incomplete.
- **Verified work:** the 784-8-10 three-party Rep3 BP run completed 90 updates,
  matched the bounded FP64 reference at aggregate level, and passed public
  synthetic operation checks. The measured result remains a bounded feasibility
  artifact, not full-dataset training or production-scale validation.
- **Remaining/blockers:** explicit approval of probabilistic-truncation
  leakage, broader numerical/overflow guarantees, and subsequent project
  scope decisions. A corrected A1-relu-corrected full run is now preserved
  separately from A0-original and passed aggregate verification. These bounded
  runs are not formal performance benchmarks; per-operation MPC profiling
  remains a separate future activity.

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
