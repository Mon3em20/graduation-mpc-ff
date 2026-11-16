# Thesis insertion: Secure BP full-run experiment

The Secure Backpropagation experiment evaluated a batched-matrix neural
network under the Rep3 Ring protocol of MP-SPDZ. The experiment used three
parties under a semi-honest honest-majority threat model. The model followed a
784-128-128-10 architecture with ReLU hidden activations and stable softmax
cross-entropy. MNIST was partitioned into 5,000 training samples, 1,000
validation samples, and 1,000 test samples. Training used SGD with learning
rate 0.05, batch size 128, a final batch of eight, five epochs, and seed 42.
The secure computation used a 64-bit ring and the fixed-point type
`sfix(16,31)`.

The training loss decreased from 2.27463 in epoch 1 to 0.908951 in epoch 5.
The final model achieved a validation loss of 0.787674 and 767 correct
predictions out of 1,000 (76.7%). On the test set, it achieved a loss of
0.77771 and 772 correct predictions out of 1,000 (77.2%). These results
represent one execution and should not be interpreted as evidence of
generalization across random seeds.

The log reports an MP-SPDZ time of 933.263 seconds, while the enclosing
measurement reports 15 minutes 48.40 seconds of elapsed wall-clock time.
Because these measurements may cover different phases, they are reported
separately rather than equated. Party 0 sent 1,497.67 MB in approximately
8,928,348 rounds, and global communication across all parties was 5,224.49
MB. The global figure is aggregate communication, not per-party traffic, and
communication volume is not equivalent to network latency. Maximum resident
set size was 768,248 KB.

The repository's Plain BP reference for seed 42 recorded training losses of
2.2747, 2.1616, 1.8416, 1.3058, and 0.9092, with final test accuracy
772/1,000 (77.2%). The Secure BP and Plain BP configurations are source-aligned
for the project split, architecture, optimizer, learning rate, batch size,
epoch count, and seed. The comparison is descriptive only; it does not show
that Secure BP outperforms Plain BP, and the Plain BP execution was not rerun
as part of this audit.

The run completed with exit status 0 but emitted a warning that the computation
domain was too small for low-round truncation and would require at least 103
bits. This warning is retained as a technical limitation rather than treated
as a fatal error or a resolved issue. Fixed-point arithmetic and probabilistic
truncation can also introduce numerical differences relative to plaintext
floating-point training. Additional seeds, repeated runs, and a focused
analysis of the truncation-domain warning are required before making
statistical or broader performance claims.

