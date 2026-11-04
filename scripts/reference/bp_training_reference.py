"""FP64 reference and shared data policy for BP-COMP-001.

This module intentionally uses only the Python standard library so that the
ordering and initialization policy can be tested without importing PyTorch.
"""
import hashlib
import json
import math
import random
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MNIST_ROOT = Path.home() / "datasets" / "MNIST" / "raw"
TRAIN_COUNT, VALIDATION_COUNT, TEST_COUNT = 90, 30, 30
BATCH_SIZE, EPOCHS, LEARNING_RATE = 3, 3, 0.05
PARTIES = 3
DATASET_SEED, ORDER_SEED, INITIALIZATION_SEED = 42, 42042, 42
FIXED_POINT_FRACTIONAL_BITS = 16


class _Images:
    def __init__(self, payload, rows, columns):
        self._payload = payload
        self._size = rows * columns

    def __getitem__(self, index):
        start = index * self._size
        return [value / 255.0 for value in
                self._payload[start:start + self._size]]


def _read_idx(path):
    with path.open("rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2051:
            raise ValueError(f"unexpected image magic in {path}: {magic}")
        rows, columns = struct.unpack(">II", stream.read(8))
        payload = stream.read()
    expected = count * rows * columns
    if len(payload) != expected:
        raise ValueError(f"image payload length mismatch in {path}")
    return _Images(payload, rows, columns)


def _read_labels(path):
    with path.open("rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected label magic in {path}: {magic}")
        payload = list(stream.read())
    if len(payload) != count:
        raise ValueError(f"label payload length mismatch in {path}")
    return payload


def load_mnist():
    return (
        _read_idx(MNIST_ROOT / "train-images-idx3-ubyte"),
        _read_labels(MNIST_ROOT / "train-labels-idx1-ubyte"),
        _read_idx(MNIST_ROOT / "t10k-images-idx3-ubyte"),
        _read_labels(MNIST_ROOT / "t10k-labels-idx1-ubyte"),
    )


def select_indices(seed=DATASET_SEED):
    """Match the existing per-class selection without changing its indices."""
    train_images, train_labels, test_images, test_labels = load_mnist()
    del train_images, test_images
    train_by_class = {label: [] for label in range(10)}
    test_by_class = {label: [] for label in range(10)}
    for index, label in enumerate(train_labels):
        train_by_class[label].append(index)
    for index, label in enumerate(test_labels):
        test_by_class[label].append(index)
    selected = {"train": [], "validation": [], "test": []}
    for label in range(10):
        candidates = train_by_class[label][:]
        random.Random(seed + label).shuffle(candidates)
        selected["train"].extend(candidates[:9])
        selected["validation"].extend(candidates[9:12])
        candidates = test_by_class[label][:]
        random.Random(seed + 1000 + label).shuffle(candidates)
        selected["test"].extend(candidates[:3])
    return selected


def party_training_indices(selected, order_seed=ORDER_SEED, epoch=0):
    """Return one ordered list of 30 source indices for each party."""
    by_party = {party: [] for party in range(PARTIES)}
    for position, index in enumerate(selected["train"]):
        by_party[position % PARTIES].append(index)
    for party in range(PARTIES):
        random.Random(order_seed + epoch * PARTIES + party).shuffle(
            by_party[party])
    return by_party


def training_batches(selected, order_seed=ORDER_SEED, epoch=0):
    by_party = party_training_indices(selected, order_seed, epoch)
    return [tuple(by_party[party][batch] for party in range(PARTIES))
            for batch in range(TRAIN_COUNT // BATCH_SIZE)]


def old_training_batches(selected):
    return [tuple(selected["train"][batch * PARTIES + party]
                  for party in range(PARTIES))
            for batch in range(TRAIN_COUNT // BATCH_SIZE)]


def _he_normal(rng, fan_in):
    values = []
    while len(values) < 784 * 8:
        u1 = max(rng.random(), 1e-15)
        u2 = rng.random()
        radius = math.sqrt(-2.0 * math.log(u1))
        values.extend((radius * math.cos(2.0 * math.pi * u2),
                       radius * math.sin(2.0 * math.pi * u2)))
    scale = math.sqrt(2.0 / fan_in)
    return [value * scale for value in values[:784 * 8]]


def initialize(method="he_xavier", seed=INITIALIZATION_SEED):
    """Return flattened W1, b1, W2, b2 using an explicit policy."""
    if method == "deterministic":
        w1 = [(i - j) / 1000.0 for i in range(784) for j in range(8)]
        w2 = [(i - j) / 1000.0 for i in range(8) for j in range(10)]
    elif method == "he_xavier":
        rng = random.Random(seed)
        w1 = _he_normal(rng, 784)
        limit = math.sqrt(6.0 / (8 + 10))
        w2 = [rng.uniform(-limit, limit) for _ in range(8 * 10)]
    else:
        raise ValueError(f"unsupported initialization method: {method}")
    return w1, [0.0] * 8, w2, [0.0] * 10


def parameter_payload(parameters, method, seed):
    w1, b1, w2, b2 = parameters
    return {
        "method": method,
        "seed": seed,
        "architecture": "784-8-10",
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
    }


def parameter_hash(parameters, method, seed):
    payload = json.dumps(parameter_payload(parameters, method, seed),
                          sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _quantize(value):
    scale = 2 ** FIXED_POINT_FRACTIONAL_BITS
    return round(value * scale) / scale


def _forward(parameters, features, quantized=False):
    w1, b1, w2, b2 = parameters
    pre = [sum(_quantize(w1[i * 8 + j] * features[i])
               if quantized else w1[i * 8 + j] * features[i]
               for i in range(784)) + b1[j]
           for j in range(8)]
    if quantized:
        pre = [_quantize(value) for value in pre]
    hidden = [max(0.0, value) for value in pre]
    logits = [sum(_quantize(w2[j * 10 + k] * hidden[j])
                  if quantized else w2[j * 10 + k] * hidden[j]
                  for j in range(8)) + b2[k]
              for k in range(10)]
    if quantized:
        logits = [_quantize(value) for value in logits]
    shift = max(logits)
    exp_logits = [math.exp(value - shift) for value in logits]
    denominator = sum(exp_logits)
    probabilities = [value / denominator for value in exp_logits]
    if quantized:
        probabilities = [_quantize(value) for value in probabilities]
    return pre, hidden, logits, probabilities


def _step(parameters, batch, quantized=False):
    w1, b1, w2, b2 = parameters
    dw1 = [0.0] * (784 * 8)
    db1 = [0.0] * 8
    dw2 = [0.0] * (8 * 10)
    db2 = [0.0] * 10
    loss = 0.0
    correct = 0
    for features, label in batch:
        pre, hidden, logits, probabilities = _forward(
            parameters, features, quantized)
        loss += math.log(sum(math.exp(value - max(logits))
                             for value in logits)) + max(logits) - logits[label]
        correct += int(max(range(10), key=probabilities.__getitem__) == label)
        error = [probabilities[k] - (k == label) for k in range(10)]
        dpre = [
            sum(_quantize(error[k] * w2[j * 10 + k])
                if quantized else error[k] * w2[j * 10 + k]
                for k in range(10))
            * (pre[j] > 0.0)
            for j in range(8)
        ]
        for j in range(8):
            db1[j] += dpre[j]
            for i in range(784):
                dw1[i * 8 + j] += (_quantize(dpre[j] * features[i])
                                    if quantized else dpre[j] * features[i])
        for k in range(10):
            db2[k] += error[k]
            for j in range(8):
                dw2[j * 10 + k] += (_quantize(error[k] * hidden[j])
                                     if quantized else error[k] * hidden[j])
    scale = LEARNING_RATE / BATCH_SIZE
    for index in range(len(w1)):
        w1[index] -= scale * dw1[index]
    for index in range(len(b1)):
        b1[index] -= scale * db1[index]
    for index in range(len(w2)):
        w2[index] -= scale * dw2[index]
    for index in range(len(b2)):
        b2[index] -= scale * db2[index]
    if quantized:
        for values in parameters:
            for index, value in enumerate(values):
                values[index] = _quantize(value)
    return loss, correct


def train(parameters, records, order_seed=ORDER_SEED, shuffled=True,
          quantized=False):
    selected = records["selected"]
    train_images, train_labels, test_images, test_labels = load_mnist()
    epoch_rows = []
    for epoch in range(EPOCHS):
        batches = (training_batches(selected, order_seed, epoch)
                   if shuffled else old_training_batches(selected))
        total_loss = total_correct = 0
        for batch_indices in batches:
            batch = [(train_images[index], train_labels[index])
                     for index in batch_indices]
            loss, correct = _step(parameters, batch, quantized)
            total_loss += loss
            total_correct += correct
        validation = evaluate(parameters, selected["validation"],
                              train_images, train_labels, quantized=quantized)
        epoch_rows.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / TRAIN_COUNT,
            "train_accuracy": total_correct / TRAIN_COUNT,
            "validation_loss": validation["loss"],
            "validation_correct": validation["correct"],
        })
    test = evaluate(parameters, selected["test"], test_images, test_labels,
                    quantized=quantized)
    return epoch_rows, test


def evaluate(parameters, indices, images, labels, quantized=False):
    loss = correct = 0.0
    for index in indices:
        _, _, logits, probabilities = _forward(
            parameters, images[index], quantized)
        shift = max(logits)
        loss += math.log(sum(math.exp(value - shift) for value in logits)) + shift - logits[labels[index]]
        correct += int(max(range(10), key=probabilities.__getitem__) == labels[index])
    count = len(indices)
    return {"loss": loss / count, "correct": int(correct), "total": count}


def summarize_parameters(parameters):
    values = parameters[0] + parameters[2]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "stddev": math.sqrt(variance),
        "variance": variance,
    }
