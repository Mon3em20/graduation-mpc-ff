# Secure BP matrix prototype

## Native API findings

The pinned MP-SPDZ v0.4.4 `Dense` implementation stores `X` as
`[N, d, d_in]`, `W` as `[d_in, d_out]`, and `Y` as `[N, d, d_out]`.
`Dense.compute_f_input()` flattens the batch dimensions and calls
`X_sub.direct_mul(W, indices=...)`; selected dataset rows are gathered into
sequential batch output rows before bias addition. `Dense.backward()` computes
`f_schur_Y * W^T` with `direct_mul_trans()`.

`DenseBase.backward_params()` computes `B^T A` with
`B.direct_trans_mul(A, reduce=False, indices=...)`. The result is stored as
`unreduced_sfix` and reduced once with `reduce_after_mul()` after the batch
accumulation. Bias gradients are summed over the batch before the optimizer
update.

The native `SGD` defaults are not compatible with this experiment: momentum is
`0.9`, learning rate is `0.01`, and learning-rate decay is enabled. The
prototype therefore performs its own vanilla SGD update at `0.05`.

## Weight layout

PyTorch `Linear.weight.flatten()` is ordered `[out_features, in_features]`.
MP-SPDZ `Dense.W` is `[in_features, out_features]`. The official existing
matched input generator writes the PyTorch flatten order directly and does not
transpose it. That order is not compatible with `Dense.W`.

The isolated prototype generator transposes each weight matrix before writing:

```text
[out0_in0, out1_in0, out0_in1, out1_in1, ...]
```

Existing official input artifacts are intentionally unchanged.

## Prototype

[bp_matrix_prototype.mpc](../../mpc/programs/bp/bp_matrix_prototype.mpc) is an
isolated batch-size-three prototype. It uses:

- `X.direct_mul(W)` for forward affine layers.
- `direct_mul_trans(W)` for hidden-layer backward propagation.
- `direct_trans_mul(..., reduce=False)` for batch weight gradients.
- One `reduce_after_mul()` after each accumulated gradient.
- Explicit bias summation and vanilla SGD at `0.05`.
- Strict-positive ReLU derivatives and stable softmax cross-entropy.

[prepare_secure_bp_matrix_prototype.py](../../scripts/prepare_secure_bp_matrix_prototype.py)
writes isolated transposed-weight inputs. It does not modify official matched
inputs.

## Validation

Focused Python layout and matrix-algebra tests pass. The prototype compiles
with the pinned compiler using `-l -M -E ring`:

- Exit code: `0`
- Duration: `1.90` seconds
- Maximum RSS: `83,076 KB`
- VM matrix multiplications: `8`
- Integer triples: `882,775`
- Probabilistic truncations: `239,254`
- VM rounds: `1,095`

These are compile-time/resource measurements only. No prototype MPC execution
has been performed, so runtime and communication improvements are not yet
claimed. Numerical equivalence beyond the focused Python layout/algebra tests
requires an isolated prototype execution.
