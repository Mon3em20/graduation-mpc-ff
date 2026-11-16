"""Offline Python reference for bp_matched_matrix.mpc (no MPC is run).

``simulate`` mirrors the program step by step: the same input-stream consumption
order (party queues, owner = (batch_start + sample) % 3), the same batched
matrix formulation, strict-positive ReLU masks, stable softmax cross entropy,
the same SGD step ``(0.05 * grad) * (1/B)`` and the same per-epoch / evaluation
reporting.  Two arithmetic modes:

* ``float``: exact float64 (algorithmic equivalence with PyTorch SGD).
* ``fixed``: emulation of sfix(16,31) -- 16 fractional bits, products summed
  exactly then truncated once with *probabilistic rounding*, cfix constants
  quantized, non-linear functions evaluated in float and rounded to the grid.
  This is an EMULATION used to estimate error accumulation; it does not prove
  MP-SPDZ's behaviour (its exp/log/division approximations are not modelled).
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

F = 16
SCALE = 1 << F
SIZES = [784, 128, 128, 10]
FULL_BATCH, FINAL_BATCH, FULL_BATCHES, EPOCHS, EVAL_CHUNK = 128, 8, 39, 5, 100


class Arith:
    def __init__(self, mode, seed=0):
        assert mode in ("float", "fixed")
        self.fixed = mode == "fixed"
        self.rng = np.random.default_rng(seed)

    def q(self, x):
        """Value -> representable value (nearest grid point in fixed mode)."""
        return np.rint(x * SCALE) / SCALE if self.fixed else x

    def const(self, c):
        return self.q(c)

    def mm(self, a, b):
        """Matrix product: exact accumulation, one probabilistic truncation."""
        if not self.fixed:
            return a @ b
        raw = (a * SCALE) @ (b * SCALE)  # integers < 2**53: exact in float64
        return np.floor(raw / SCALE + self.rng.random(raw.shape)) / SCALE

    def mul_const(self, a, c):
        if not self.fixed:
            return a * c
        raw = (a * SCALE) * (c * SCALE)
        return np.floor(raw / SCALE + self.rng.random(raw.shape)) / SCALE


def softmax_loss(az, Z3, Y):
    logits = az.q(Z3)
    shift = logits.max(axis=1, keepdims=True)
    exps = np.exp(logits - shift)
    total = exps.sum(axis=1, keepdims=True)
    P = az.q(exps / total)
    loss = -((Y * logits).sum(axis=1) - (shift[:, 0] + np.log(total[:, 0])))
    return P, loss, logits


class PartyStreams:
    """Per-party input queues; ``take_sample`` consumes exactly like input_from."""

    def __init__(self, params, plan, images, labels):
        self.params = params
        self.queues = [deque(), deque(), deque()]
        self.expected_split = {}
        for owner, split, index in plan:
            self.queues[owner].append((split, index))
        self.images, self.labels = images, labels
        self.consumed = [0, 0, 0]

    def take_sample(self, owner, split):
        got_split, index = self.queues[owner].popleft()
        assert got_split == split, f"party {owner}: expected {split}, stream has {got_split}"
        self.consumed[owner] += 1
        x = self.images[split][index]
        y = np.zeros(10)
        y[self.labels[split][index]] = 1.0
        return x, y, index

    def empty(self):
        return all(not q for q in self.queues)


def load_rows(streams, rows, start, split, trace=None):
    X = np.zeros((rows, 784))
    Y = np.zeros((rows, 10))
    for sample in range(rows):
        x, y, index = streams.take_sample((start + sample) % 3, split)
        X[sample], Y[sample] = x, y
        if trace is not None:
            trace.append(index)
    return X, Y


def forward(az, W, b, X):
    Z1 = az.mm(X, W[0]) + b[0]
    M1 = (Z1 > 0).astype(float)
    H1 = Z1 * M1
    Z2 = az.mm(H1, W[1]) + b[1]
    M2 = (Z2 > 0).astype(float)
    H2 = Z2 * M2
    Z3 = az.mm(H2, W[2]) + b[2]
    return (Z1, Z2, Z3), (M1, M2), (X, H1, H2)


def run_batch(az, W, b, X, Y, lr, rows):
    (Z1, Z2, Z3), (M1, M2), (X, H1, H2) = forward(az, W, b, X)
    P, loss, _ = softmax_loss(az, Z3, Y)
    D3 = P - Y
    D2 = az.mm(D3, W[2].T) * M2
    D1 = az.mm(D2, W[1].T) * M1
    grads_w = [az.mm(X.T, D1), az.mm(H1.T, D2), az.mm(H2.T, D3)]
    grads_b = [D1.sum(0), D2.sum(0), D3.sum(0)]
    inverse = az.const(1.0 / rows)
    lrc = az.const(lr)
    for layer in range(3):
        W[layer] = W[layer] - az.mul_const(az.mul_const(grads_w[layer], lrc), inverse)
        b[layer] = b[layer] - az.mul_const(az.mul_const(grads_b[layer], lrc), inverse)
    return loss.sum()


def evaluate(az, W, b, streams, split, count):
    loss_total, correct = 0.0, 0
    for chunk in range(count // EVAL_CHUNK):
        X, Y = load_rows(streams, EVAL_CHUNK, chunk * EVAL_CHUNK, split)
        (_, _, Z3), _, _ = forward(az, W, b, X)
        _, loss, logits = softmax_loss(az, Z3, Y)
        loss_total += loss.sum()
        correct += int((logits.argmax(1) == Y.argmax(1)).sum())
    return loss_total / count, correct


def simulate(serialized_params, plan, images, labels, mode="float", seed=0, lr=0.05,
             epochs=EPOCHS, full_batches=FULL_BATCHES, final_batch=FINAL_BATCH,
             validation_n=1000, test_n=1000, trace=None):
    """Run the matrix program. ``serialized_params`` use the MPC order
    (W[in,out], b, ...). Returns dict with per-epoch loss, final eval and params."""
    az = Arith(mode, seed)
    p = [np.array(v, dtype=float) for v in serialized_params]
    W = [az.q(p[0].reshape(784, 128)), az.q(p[2].reshape(128, 128)),
         az.q(p[4].reshape(128, 10))]
    b = [az.q(p[1]), az.q(p[3]), az.q(p[5])]
    streams = PartyStreams(serialized_params, plan, images, labels)
    epoch_losses, snapshots = [], []
    for epoch in range(epochs):
        total = 0.0
        for batch in range(full_batches):
            X, Y = load_rows(streams, FULL_BATCH, batch * FULL_BATCH, "train", trace)
            total += run_batch(az, W, b, X, Y, lr, FULL_BATCH)
        X, Y = load_rows(streams, final_batch, full_batches * FULL_BATCH, "train", trace)
        total += run_batch(az, W, b, X, Y, lr, final_batch)
        epoch_losses.append(total / (full_batches * FULL_BATCH + final_batch))
        snapshots.append(([w.copy() for w in W], [x.copy() for x in b]))
    val = evaluate(az, W, b, streams, "validation", validation_n)
    test = evaluate(az, W, b, streams, "test", test_n)
    return dict(epoch_losses=epoch_losses, validation=val, test=test, W=W, b=b,
                snapshots=snapshots, fully_consumed=streams.empty(),
                consumed=streams.consumed)
