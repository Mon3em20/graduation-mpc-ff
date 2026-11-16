# Secure BP Matrix Prototype Numerical Verification Report

## 1. Overview and Execution Context

This document provides the numerical verification analysis for the Secure BP Matrix Prototype (`bp_matrix_prototype.mpc`), comparing intermediate and updated secure fixed-point values against an authoritative double-precision floating-point (FP64) reference.

- **Authoritative Prototype Source**: `mpc/programs/bp/bp_matrix_prototype.mpc`
- **Diagnostic Program Source**: [bp_matrix_numerical_audit.mpc](../../mpc/programs/bp/bp_matrix_numerical_audit.mpc)
- **Input Directory**: `~/secure-bp-runs/matrix_proto_seed42/inputs/`
  - `Input-P0-0` (`d30e32c1e3ac3b6df5a790f1b5060edfafdd28ab6c4cb35188a88f528940900c`)
  - `Input-P1-0` (`7150269c114364f2af55f881c4ec67de0a4e93800a9fe8af2d5597d0ba8ab275`)
  - `Input-P2-0` (`9de5bdefd7805693afdc1240e8b78ac294f035b8ccef1bd36ff5f06d025b209b`)
- **FP64 Reference Artifacts**:
  - Full tensor state: `/tmp/matrix_proto_fp64_reference_seed42.pt`
  - Reference summary: `/tmp/matrix_proto_fp64_summary_seed42.json`
- **Diagnostic MPC Execution Log**: [matrix_numerical_audit_seed42.log](../../results/mpc/bp/matrix_numerical_audit_seed42.log)
- **Comparison Output**: [matrix_numerical_audit_seed42.json](../../results/mpc/bp/matrix_numerical_audit_seed42.json)
- **Environment & Arithmetic**:
  - Architecture: 784-128-128-10 (batch size = 3)
  - Protocol: 3-party semi-honest Rep3 Ring (64-bit ring)
  - Fixed-point representation: `sfix(16, 31)` with quantum resolution $q = 2^{-16} \approx 1.52588 \times 10^{-5}$
  - Compiler optimization flags: `-M -E ring`

All execution was carried out with full isolation. The existing MP-SPDZ `Player-Data` state was completely backed up, verified, and restored with bit-level hash integrity upon audit completion.

---

## 2. Compilation and Execution Integrity

The diagnostic source `bp_matrix_numerical_audit.mpc` was constructed by making the minimal necessary non-computational addition: revealing intermediate and updated values at player 0 at the end of the batch step, without altering the arithmetic pipeline, matrix layout, or update order.

- **Compilation Status**: Clean success (0 compilation errors or warnings).
- **Compilation Invariant Verification**:
  - Triples: 882,775 (exact match with original prototype)
  - Matrix Multiplications: 8 calls across forward, backward, and parameter gradients (exact match)
  - Probabilistic Truncations: 239,254 (exact match)
  - 3-Way Splits: 1,797 (exact match)
  - Bit Triples: 208,860 (exact match)
  - Integer Dot Products: 119,582 (exact match)
  - VM Rounds: 1,112 (vs 1,095; difference solely attributable to clear reveal rounds)
- **MPC Runtime Execution**: Clean success (0.49 seconds total runtime, 20.29 MB global communication across 3 parties).
- **Player-Data Restoration**: Verified identical before and after execution across all 34 entries and symlinks.

---

## 3. High-Level Numerical Agreement

| Metric | MPC Diagnostic | FP64 Reference | Status / Absolute Diff |
| :--- | :--- | :--- | :--- |
| **Prediction Sample 0** | `4` | `4` | **EXACT MATCH** |
| **Prediction Sample 1** | `3` | `3` | **EXACT MATCH** |
| **Prediction Sample 2** | `3` | `3` | **EXACT MATCH** |
| **Batch Correct Count** | `0 / 3` | `0 / 3` | **EXACT MATCH** |
| **Mean Cross-Entropy Loss** | `2.3213348` | `2.3213721` | Abs diff: $3.733 \times 10^{-5}$ |
| **Sample 0 Loss** | `2.3071289` | `2.3071681` | Abs diff: $3.919 \times 10^{-5}$ |
| **Sample 1 Loss** | `2.3198242` | `2.3199099` | Abs diff: $8.568 \times 10^{-5}$ |
| **Sample 2 Loss** | `2.3370514` | `2.3370384` | Abs diff: $1.296 \times 10^{-5}$ |
| **Tensor Shapes (24 / 24)** | All Matched | All Matched | **EXACT MATCH** |

All predictions and classification accuracy matched the FP64 reference exactly. Per-sample losses and mean loss match within $8.6 \times 10^{-5}$, well within the expected precision bounds of 16-bit fixed point.

---

## 4. Element-by-Element Tensor Error Metrics

The table below reports measured errors for every tensor audited across the 238,964 numerical elements evaluated.

### Error Policies Documented
- **Max Absolute Error**: $\max_i |y_{mpc}[i] - y_{ref}[i]|$
- **Mean Absolute Error**: $\frac{1}{N} \sum_i |y_{mpc}[i] - y_{ref}[i]|$
- **RMSE**: $\sqrt{\frac{1}{N} \sum_i (y_{mpc}[i] - y_{ref}[i])^2}$
- **Relative L2 Error**: $\|y_{mpc} - y_{ref}\|_2 / \|y_{ref}\|_2$ (Frobenius norm ratio)
- **Relative Error ($\epsilon = q$)**: $\max_i \frac{|y_{mpc}[i] - y_{ref}[i]|}{|y_{ref}[i]| + 2^{-16}}$ (normalized by physical quantum $q = 1.526 \times 10^{-5}$)

| Stage | Tensor | Elements | Max Abs Error | Mean Abs Error | RMSE | Relative L2 Error | Rel Err ($\epsilon=q$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Forward** | `Z1` | 384 | $2.2099 \times 10^{-4}$ | $3.8706 \times 10^{-5}$ | $5.1994 \times 10^{-5}$ | $2.0830 \times 10^{-4}$ | $7.1270 \times 10^{-2}$ |
| **Forward** | `H1` | 384 | $2.2099 \times 10^{-4}$ | $2.1089 \times 10^{-5}$ | $3.9825 \times 10^{-5}$ | $2.0209 \times 10^{-4}$ | $2.5363 \times 10^{-2}$ |
| **Forward** | `Z2` | 384 | $8.6785 \times 10^{-5}$ | $1.9808 \times 10^{-5}$ | $2.5843 \times 10^{-5}$ | $2.1951 \times 10^{-4}$ | $3.6143 \times 10^{-1}$ |
| **Forward** | `H2` | 384 | $8.6605 \times 10^{-5}$ | $9.8479 \times 10^{-6}$ | $1.8398 \times 10^{-5}$ | $2.3270 \times 10^{-4}$ | $2.0720 \times 10^{-2}$ |
| **Forward** | `Z3` | 30 | $2.8213 \times 10^{-5}$ | $1.0117 \times 10^{-5}$ | $1.2427 \times 10^{-5}$ | $1.7045 \times 10^{-4}$ | $2.0173 \times 10^{-3}$ |
| **Forward** | `P` | 30 | $2.8187 \times 10^{-5}$ | $9.0151 \times 10^{-6}$ | $1.1323 \times 10^{-5}$ | $1.1295 \times 10^{-4}$ | $3.4548 \times 10^{-4}$ |
| **Forward** | `loss_each` | 3 | $8.4995 \times 10^{-5}$ | $6.7832 \times 10^{-5}$ | $6.8990 \times 10^{-5}$ | $2.9719 \times 10^{-5}$ | $3.6839 \times 10^{-5}$ |
| **Forward** | `mean_loss` | 1 | $3.7332 \times 10^{-5}$ | $3.7332 \times 10^{-5}$ | $3.7332 \times 10^{-5}$ | $1.6082 \times 10^{-5}$ | $1.6082 \times 10^{-5}$ |
| **Backward** | `D3` | 30 | $2.8187 \times 10^{-5}$ | $9.0151 \times 10^{-6}$ | $1.1323 \times 10^{-5}$ | $3.7655 \times 10^{-5}$ | $3.4548 \times 10^{-4}$ |
| **Backward** | `D2` | 384 | $2.0723 \times 10^{-5}$ | $3.0843 \times 10^{-6}$ | $5.3034 \times 10^{-6}$ | $1.6608 \times 10^{-4}$ | $4.0671 \times 10^{-3}$ |
| **Backward** | `D1` | 384 | $1.8650 \times 10^{-5}$ | $3.0558 \times 10^{-6}$ | $5.1745 \times 10^{-6}$ | $4.0979 \times 10^{-4}$ | $2.5999 \times 10^{-2}$ |
| **Backward** | `dW3` | 1,280 | $1.0493 \times 10^{-4}$ | $6.5185 \times 10^{-6}$ | $1.2124 \times 10^{-5}$ | $2.3872 \times 10^{-4}$ | $1.1236 \times 10^{-1}$ |
| **Backward** | `db3` | 10 | $1.4993 \times 10^{-5}$ | $1.0130 \times 10^{-5}$ | $1.0690 \times 10^{-5}$ | $1.6640 \times 10^{-5}$ | $4.6262 \times 10^{-5}$ |
| **Backward** | `dW2` | 16,384 | $3.9325 \times 10^{-5}$ | $2.9383 \times 10^{-6}$ | $5.3465 \times 10^{-6}$ | $3.9599 \times 10^{-4}$ | $1.0000 \times 10^{0}$ |
| **Backward** | `db2` | 128 | $3.4089 \times 10^{-5}$ | $5.8667 \times 10^{-6}$ | $8.7214 \times 10^{-6}$ | $1.2342 \times 10^{-4}$ | $4.6914 \times 10^{-3}$ |
| **Backward** | `dW1` | 100,352 | $5.1867 \times 10^{-5}$ | $2.0155 \times 10^{-6}$ | $4.9506 \times 10^{-6}$ | $4.9589 \times 10^{-4}$ | $7.9513 \times 10^{-1}$ |
| **Backward** | `db1` | 128 | $3.8990 \times 10^{-5}$ | $6.2138 \times 10^{-6}$ | $9.1341 \times 10^{-6}$ | $3.4359 \times 10^{-4}$ | $2.3970 \times 10^{-2}$ |
| **SGD Update** | `W1_new` | 100,352 | $2.2904 \times 10^{-5}$ | $4.4905 \times 10^{-6}$ | $5.5126 \times 10^{-6}$ | $2.6711 \times 10^{-4}$ | $1.0000 \times 10^{0}$ |
| **SGD Update** | `b1_new` | 128 | $1.5925 \times 10^{-5}$ | $5.0610 \times 10^{-6}$ | $6.2680 \times 10^{-6}$ | $3.0084 \times 10^{-4}$ | $9.9184 \times 10^{-2}$ |
| **SGD Update** | `W2_new` | 16,384 | $2.3798 \times 10^{-5}$ | $4.9367 \times 10^{-6}$ | $6.1658 \times 10^{-6}$ | $1.2112 \times 10^{-4}$ | $5.1078 \times 10^{-1}$ |
| **SGD Update** | `b2_new` | 128 | $1.7678 \times 10^{-5}$ | $5.9038 \times 10^{-6}$ | $7.2475 \times 10^{-6}$ | $1.4331 \times 10^{-4}$ | $3.0426 \times 10^{-2}$ |
| **SGD Update** | `W3_new` | 1,280 | $2.5232 \times 10^{-5}$ | $5.6814 \times 10^{-6}$ | $7.0940 \times 10^{-6}$ | $1.4112 \times 10^{-4}$ | $2.4918 \times 10^{-1}$ |
| **SGD Update** | `b3_new` | 10 | $4.9413 \times 10^{-5}$ | $1.6045 \times 10^{-5}$ | $2.0168 \times 10^{-5}$ | $4.4047 \times 10^{-4}$ | $3.1812 \times 10^{-3}$ |

---

## 5. Numerical Observations and Analysis of Discrepancies

1. **Global Maximum Absolute Error**:
   The worst-case absolute error across all 238,964 elements is **$2.2099 \times 10^{-4}$** (occurring in $Z1$ and $H1$ affine layer dot products with 784 inputs). This error corresponds to approximately **14.5 fixed-point quanta** ($14.5 \times 2^{-16}$). In a 784-dimensional inner product with probabilistic truncation, accumulating an error of $\sim 14$ LSBs is entirely consistent with theoretical error bounds for fixed-point dot product truncation in MP-SPDZ.

2. **Mean Absolute Error & RMSE**:
   Across all weight tensors and weight gradients (`W1_new`, `dW1`, `W2_new`, `dW2`, `W3_new`, `dW3`), the mean absolute error is consistently **between $2.0 \times 10^{-6}$ and $6.5 \times 10^{-6}$**, which is **less than half of one fixed-point LSB** ($0.5 \times 2^{-16} \approx 7.6 \times 10^{-6}$).

3. **Relative L2 (Frobenius) Norm Error**:
   Across every single audited tensor, the relative L2 error is **strictly bounded below $0.05\%$** ($5 \times 10^{-4}$):
   - Worst case: `dW1` with relative L2 error of $4.9589 \times 10^{-4}$ ($0.0496\%$).
   - Post-update weights: `W1_new` ($0.0267\%$), `W2_new` ($0.0121\%$), `W3_new` ($0.0141\%$).
   - Loss: `mean_loss` ($0.0016\%$).

4. **Near-Zero Denominator Analysis in Elementwise Relative Error**:
   When evaluating element-wise relative error with an infinitesimal safeguard ($\epsilon = 10^{-8}$), certain elements in `W1_new` and `dW2` exhibit large relative error quotients ($\approx 184$). Inspection reveals that these elements have expected values of order $10^{-8}$ in FP64 (e.g., $7.32 \times 10^{-8}$), whereas in 16-bit fixed point, numbers smaller than $1.526 \times 10^{-5}$ quantize to $0$ or $\pm 2^{-16}$. The absolute discrepancy is exactly one LSB ($1.53 \times 10^{-5}$), but dividing by $10^{-8}$ produces an artificial ratio. When normalized by the physical resolution of the ring ($\epsilon = 2^{-16}$), the maximum relative error is $1.0$ (i.e. strictly $\le 1$ LSB discrepancy at near-zero values).

---

## 6. Verification Status

### Has the Matrix Prototype Passed Numerical Verification?
**YES.**
The Secure BP Matrix Prototype has successfully passed element-by-element numerical audit:
- All 24 forward, backward, prediction, loss, and post-update tensors were confirmed to match in names, dimensions, and matrix orientation.
- Exact categorical prediction agreement (`[4, 3, 3]`) was observed.
- Mean cross-entropy loss matched within $3.73 \times 10^{-5}$.
- Maximum absolute error across the full 118,282 parameters and intermediate activations was bounded by $2.21 \times 10^{-4}$.
- Relative L2 norm error across all tensors was strictly bounded below $0.05\%$.

### What Remains Unresolved?
While the numerical equivalence of the isolated single-batch matrix prototype against the FP64 reference is fully established, the following scope boundaries remain open:
1. **Multi-Batch / Multi-Epoch Training**: The prototype was audited on a single batch of size 3 (seed 42). Compounding quantization noise across multiple epochs has not yet been measured.
2. **Batch Scaling**: The audit tested $N=3$; larger batch sizes ($N=128$, $N=256$) will require validation of memory limits and compilation scaling.
3. **Stage B Milestone Gate**: In accordance with project policy, this single-batch diagnostic alone does not complete Stage B, which requires multi-batch convergence and full benchmark characterization against the agreed baseline.
