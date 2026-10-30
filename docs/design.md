# Technical design

The project compares plaintext MNIST Backpropagation (BP) and
Forward-Forward (FF) under a common experimental methodology before extending
the comparison to secure computation.

## Implemented components

`src/baselines/bp.py` implements a deterministic 784-128-128-10 ReLU network
with cross-entropy and SGD. `src/baselines/ff.py` implements greedy local
goodness training, L2 normalization, label coding, and the recorded
uniform-random and auxiliary-predictor hard-negative variants.

The MPC layer currently contains numerical microtests only. Secret `sfix`
inputs, weights, products, ReLU comparisons, loss, gradients, and updates are
computed in MP-SPDZ; selected diagnostics are deliberately revealed for
comparison with an independent plaintext reference.

## Planned secure comparison

The intended common setting is three-party honest-majority semi-honest Rep3
Ring with fixed-point arithmetic. BP will require secure dense products,
activation comparisons, loss, gradients, and updates. FF additionally requires
local squared goodness, reductions, normalization, and negative-label
selection. Inputs are secret-shared and only explicitly selected diagnostics or
final metrics may be opened.

The experiment flow is: freeze a plaintext configuration, prepare identical
data partitions, compile the corresponding MPC program, execute all parties,
collect communication/round counters, verify numerical outputs, and compare
utility and cost. The present microtests validate arithmetic mechanics only;
they do not establish secure MNIST training or Stage D profiling.
