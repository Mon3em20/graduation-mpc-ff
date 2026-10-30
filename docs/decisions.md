# Research decisions

| Topic | Evidence-based decision | Status |
|---|---|---|
| Dataset | MNIST with deterministic seeded train/validation/test partitions | Supported by baseline code |
| Plaintext baseline | 5,000/1,000/1,000 project baseline plus separately recorded source-aligned references | Supported |
| MPC framework | MP-SPDZ v0.4.4 at commit `113c4e47dd94bca9ecab029d2f0abe649bea5c52` | Evidence recorded |
| Candidate protocol | Three-party semi-honest honest-majority Rep3 Ring | Provisional; supervisor confirmation pending |
| Fixed point | `sfix.set_precision(16, 31)` in Tiny BP programs | Implemented microtest setting |
| FF reference | Four 128-unit layers, 60 epochs/layer, threshold 2.0, uniform negatives | Recorded reference adaptation |
| Hard negatives | Neutral-label auxiliary predictor comparison is retained as a controlled experiment | Recorded, not general proof |
| Profiling | Stage D must measure per-operation BP/FF cost; quickstart counters are not profiling | Pending |

MASCOT appears only in separate tutorial evidence and is not claimed as the
project training protocol.
