# Research decisions

| Topic | Decision and evidence | Status |
|---|---|---|
| Dataset | MNIST. Plaintext Benchmark v1 uses 5,000/1,000/1,000; bounded MPC BP uses 90/30/30 with deterministic stratified selection. | Selected |
| Plaintext BP | Project baseline is `784-128-128-10`, five epochs, batch size 128, learning rate 0.05. | Implemented |
| Plaintext FF | Source baseline uses 128-unit local layers, threshold 2.0, L2 normalization, learning rate 0.01, batch size 128, five epochs per layer, and uniform incorrect-label negatives; hard negatives are a comparison. | Implemented |
| MPC framework | MP-SPDZ v0.4.4, commit `113c4e47dd94bca9ecab029d2f0abe649bea5c52`. | Pinned |
| MPC protocol | Three-party semi-honest honest-majority Rep3 Ring. | Selected |
| Ring | 64-bit ring. | Selected |
| Fixed point | `sfix(16,31)` is used by Tiny tests and the full `784-8-10` BP implementation. | Implemented |
| Probabilistic truncation | A0-original and A1-relu-corrected use the retained runtime default `use_trunc_pr=True`. It carries a qualified stand-alone modified-functionality limitation; no zero-leakage or UC claim is made. | Qualified |
| Split conversion | `use_split(3)` is documented separately from truncation and is not treated as a security proof. | Retained |
| Secure BP baseline | A1-relu-corrected is the complete bounded 90-update Secure BP run; A0-original is historical evidence. | Stage B complete |
| Comparison-ready BP | BP-COMP-001 preserves the 90/30/30 indices and ownership, uses deterministic per-party epoch shuffling (`42042`), and exact He/Xavier initialization (`42`). | Recorded reference |
| Secure FF | End-to-end Secure FF is not implemented. | Stage C not started |
| Profiling | Stage D must profile per-operation BP and FF costs; compiler counters and total run statistics do not satisfy it. | Not started |
| Hard negatives | The preliminary plaintext comparison is evidence for Stage A, not automatically Stage E. | Preliminary |
| Stage E | Select one measured FF bottleneck, implement one MPC-aware modification, compare with unmodified Secure FF, and perform an ablation. | Not started |
| Stage F | SNN, gating, selective activation, other datasets, and extra protocols are optional after Stages A-E. | Optional, not started |

MASCOT appears only in separate tutorial evidence and is not the selected
project training protocol. The final BP-versus-FF comparison is a mandatory
cross-stage outcome, not Stage F. No supervisor approval is inferred by these
decisions.
