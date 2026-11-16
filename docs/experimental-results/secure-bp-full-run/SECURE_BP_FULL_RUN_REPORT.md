# Secure BP matched matrix full-run report

## 1. Executive Summary

This report records one completed Secure Backpropagation (Secure BP) full run
using the batched-matrix program
`mpc/programs/bp/bp_matched_matrix.mpc`. The run used MNIST, seed 42, a
784-128-128-10 network, five epochs, and 5,000/1,000/1,000
training/validation/test samples. Secure BP reached training loss 0.908951 in
epoch 5, validation accuracy 76.7% (767/1,000), and test accuracy 77.2%
(772/1,000). The log reports 933.263 seconds of MP-SPDZ time, 15:48.40
elapsed wall-clock time, 1,497.67 MB sent by party 0, and 5,224.49 MB global
data sent across all parties.

The source and input hashes match the expected values, the input verification
summary passes, and the process exited with status 0. The run emitted a
low-round truncation domain warning; completion is not evidence that this
warning is resolved. No claim is made about multi-seed generalization or
Secure BP outperforming Plain BP.

## 2. Research Objective

The objective was to verify that the batched-matrix Secure BP implementation
can train and evaluate the matched MNIST model under MPC while preserving the
intended plaintext training configuration and reporting its computational and
communication cost.

## 3. Experimental Environment

The MPC framework was MP-SPDZ v0.4.4 at commit
`113c4e47dd94bca9ecab029d2f0abe649bea5c52`. The source configuration records a
64-bit ring, three local parties, and the Rep3 Ring protocol. The full-run log
records compilation and party execution on localhost. The repository working
tree was already dirty before this audit; existing changes were not modified.

## 4. Dataset and Data Splits

The dataset was MNIST. The input metadata records split seed 42, initialization
seed 42, utility seed 42, and 5,000 training, 1,000 validation, and 1,000 test
samples. Training uses 40 batches: 39 batches of 128 and a final batch of 8.
The documented split policy is `src.baselines.bp.datasets_and_indices`.
Training order is based on a persistent seeded PyTorch DataLoader, and input
ownership is position modulo three. The bundle metadata and verification
summary report matching order, ownership, token consumption, and input hashes.

## 5. Neural Network Architecture

The network has input dimension 784, hidden layers of 128 and 128 units, and
10 output classes. Hidden activations are ReLU. The configuration records
118,282 trainable parameters, SGD learning rate 0.05, batch size 128, and five
epochs.

## 6. Secure BP Algorithm and Matrix Implementation

The program processes each batch with MP-SPDZ native matrix products. Weights
use input-by-output row-major layout, with the input bundle documenting the
corresponding transpose from PyTorch's output-by-input layout. Stable softmax
and cross-entropy remain element-wise operations where required. The update
scales the gradient by the learning rate and the exact inverse of the actual
batch size, including the final batch of eight.

## 7. MPC Protocol and Threat Model

The run uses Rep3 Ring with three parties and a semi-honest honest-majority
threat model. This means the protocol's protection is interpreted under the
selected protocol assumptions; it does not constitute a claim of malicious
security, arbitrary collusion resistance, or a security proof for the whole
experimental system.

## 8. Fixed-Point Representation

The recorded fixed-point type is `sfix(16,31)` in a 64-bit ring. Secure
intermediate values are represented using the MP-SPDZ fixed-point and
truncation mechanisms. Quantization and probabilistic truncation can introduce
numerical differences from floating-point Plain BP.

## 9. Experimental Procedure

The completed run compiled the matched matrix source, loaded the verified
three-party input bundle, executed five training epochs, and evaluated the
epoch-5 model on validation and test samples. The audit inspected the existing
log and artifacts only; it did not execute MPC, compilation, benchmarking, or
party processes.

## 10. Training Results

| Epoch | Secure BP training loss |
|---:|---:|
| 1 | 2.27463 |
| 2 | 2.16141 |
| 3 | 1.84132 |
| 4 | 1.30545 |
| 5 | 0.908951 |

The decreasing loss is consistent with successful optimization for this
single seed and configuration.

## 11. Validation and Test Results

Validation loss was 0.787674, with 767 correct predictions out of 1,000
(76.7%). Test loss was 0.77771, with 772 correct predictions out of 1,000
(77.2%). These are aggregate revealed metrics; individual inputs, labels, and
model parameters were not reported in the log.

## 12. Runtime and Communication Analysis

MP-SPDZ reported `Time = 933.263 seconds`, including preprocessing according
to the log. The external resource measurement reported 15:48.40 elapsed
wall-clock time. These measurements must not be treated as identical scopes.
The log reports 1,497.67 MB sent by party 0 in approximately 8,928,348 rounds
and 5,224.49 MB as global data sent across all parties. The latter is aggregate
communication, not per-party communication, and neither volume is a network
latency measurement. Maximum resident set size was 768,248 KB.

## 13. Numerical Comparison Against Plain BP

The repository contains the Plain BP implementation in `src/baselines/bp.py`
and seed-42 rows in `results/baselines/bp.csv`. That source uses the same
5,000/1,000/1,000 project split, architecture, five epochs, batch size, SGD
learning rate, and seed-42 initialization/data ordering policy. Its recorded
seed-42 training losses are 2.2747, 2.1616, 1.8416, 1.3058, and 0.9092, and
its final test result is 772/1,000 (77.2%). The source also records validation
loss 0.7881155 and validation accuracy 76.6% for epoch 5.

This is a numerical and configuration comparison for one seed, not a claim of
superiority. The Secure BP loss values differ at the displayed precision, as
expected from fixed-point and secure-operation approximations. The baseline
CSV is repository evidence, but its original execution was not rerun during
this audit.

## 14. Truncation Warning and Technical Limitations

The log states: “The computation domain is too small for low-round truncation;
it would need to have at least 103 bits.” The program nevertheless completed
with exit status 0. This warning is documented as an implementation
observation and limitation, not as a fatal error and not as resolved. Its
effect on numerical reliability and security assumptions requires separate
analysis.

Other limitations are the single seed, the absence of repeated-run variance
estimates, and the different measurement scopes for reported MP-SPDZ time and
elapsed wall-clock time. The audit does not establish formal security
properties beyond the selected Rep3 Ring protocol and semi-honest
honest-majority assumptions.

## 15. Reproducibility and Integrity Evidence

The program SHA-256 is
`8134dd8d17e4338cab489c79823d1b9138487e7aab599c3c637dd6ee505583fd`.
Input hashes and paths are listed in `integrity_manifest.txt`. The input
bundle's `verification_summary.json` reports `all_pass: true`. The full-run
log is 5,911 bytes and ends with exit status 0. The pre-run Player-Data
backup contains 33 files totaling 40,492,581 bytes; secret-bearing contents
were not inspected or copied into this report.

## 16. Threats to Validity

External validity is limited because this is one seed and one local
three-party execution. Internal numerical validity is affected by fixed-point
representation, probabilistic truncation, and the unresolved domain warning.
Performance validity is limited by the distinction between MP-SPDZ reported
time, wall-clock time, preprocessing scope, local execution, and network
conditions. Baseline fairness depends on the recorded configuration match and
does not replace independent reruns under identical controlled conditions.

## 17. Conclusion

The audited Secure BP full run completed successfully and produced the
documented training, validation, test, communication, runtime, and memory
measurements. Source and input integrity checks passed. The results are
suitable for thesis documentation as a single-run case study, provided the
truncation warning, timing scopes, single-seed design, and baseline evidence
limitations are stated explicitly.

