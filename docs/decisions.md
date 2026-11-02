# Research decisions

| Topic | Evidence-based decision | Status |
|---|---|---|
| Dataset | MNIST with deterministic seeded train/validation/test partitions | Supported by baseline code |
| Plaintext baseline | 5,000/1,000/1,000 project baseline plus separately recorded source-aligned references | Supported |
| MPC framework | MP-SPDZ v0.4.4 at commit `113c4e47dd94bca9ecab029d2f0abe649bea5c52` | Evidence recorded |
| Selected protocol | Three-party semi-honest honest-majority Rep3 Ring; at most one corrupted party; 64-bit ring observed | Selected from Stage A feasibility evidence |
| Fixed point | `sfix.set_precision(16, 31)` in Tiny BP programs | Implemented microtest setting |
| Probabilistic truncation | Retain MP-SPDZ runtime probabilistic truncation as the candidate bounded-BP policy; limit the claim to the related stand-alone semi-honest modified functionality, with no zero-leakage or UC claim; keep `use_split(3)` separate | Qualified decision; A1-relu-corrected run recorded |
| Benchmark v1 | BP `784-128-128-10`, five epochs, lr 0.05; FF `784-128-128`, two local layers, five epochs/layer, threshold 2.0, uniform negatives, all-layers inference | Standardized project baseline |
| FF reference | Four 128-unit layers, 60 epochs/layer, threshold 2.0, uniform negatives | Bounded source-aligned adaptation; not exact paper reproduction |
| Hard negatives | Neutral-label auxiliary predictor comparison is retained as a controlled experiment | Recorded, not general proof |
| Profiling | Stage D must measure per-operation BP/FF cost; quickstart counters are not profiling | Pending |

MASCOT appears only in separate tutorial evidence and is not the selected
project training protocol. Stage A is **COMPLETE (2026-11-02)**. This closure
does not claim secure BP/FF training, full-dataset MPC experiments, or Stage D
per-operation profiling.
