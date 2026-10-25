"""Plaintext greedy Forward-Forward MNIST baseline for Stage A."""
import csv
import random
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "datasets"
OUT = ROOT / "results" / "ff_mnist_results.csv"
BASELINE_ARCHIVE = ROOT / "results" / "archive" / "ff_mnist_baseline_before_threshold_diagnostics.csv"
SEEDS = (42, 43, 44, 45, 46)
TIMING_REPETITIONS = 3
EPOCHS_PER_LAYER, BATCH_SIZE, LR, THRESHOLD = 5, 128, 0.01, 2.0
THRESHOLD_GRID = (0.5, 1.5, 2.0, 3.0)
TRAIN_N, VALID_N, TEST_N, OLD_TEST_N = 5000, 1000, 1000, 1000
torch.set_num_threads(4)


def get_data():
    train = datasets.MNIST(DATA, train=True, download=False, transform=transforms.ToTensor())
    test = datasets.MNIST(DATA, train=False, download=False, transform=transforms.ToTensor())
    rng = torch.Generator().manual_seed(42)
    train_perm = torch.randperm(len(train), generator=rng)
    old_test_perm = torch.randperm(len(test), generator=rng)
    tr = train_perm[:TRAIN_N].tolist()
    va = train_perm[TRAIN_N:TRAIN_N + VALID_N].tolist()
    te = old_test_perm[OLD_TEST_N:OLD_TEST_N + TEST_N].tolist()
    assert not set(tr) & set(va) and len(set(te)) == TEST_N
    return train, test, tr, va, te


def add_label(images, labels):
    x = images.reshape(images.shape[0], -1).clone()
    x[:, :10] = 0.0
    x[torch.arange(labels.numel()), labels] = 1.0
    return x


def normalized(x):
    return F.normalize(x, p=2, dim=1, eps=1e-6)


class FF(nn.Module):
    def __init__(self, width=128, hidden_layers=2):
        super().__init__()
        dims = [784] + [width] * hidden_layers
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                     for i in range(hidden_layers)])

    def candidate_scores(self, images):
        scores, second_scores, nonfirst_scores = [], [], []
        for label in range(10):
            y = torch.full((len(images),), label, dtype=torch.long)
            x, goodness = add_label(images, y), []
            for layer in self.layers:
                x = F.relu(layer(x))
                goodness.append(x.square().sum(dim=1))
                x = normalized(x)
            scores.append(torch.stack(goodness).sum(0))
            second_scores.append(goodness[1] if len(goodness) > 1 else goodness[0])
            nonfirst_scores.append(torch.stack(goodness[1:]).sum(0)
                                   if len(goodness) > 1 else goodness[0])
        all_scores = torch.stack(scores, dim=1)
        second_only = torch.stack(second_scores, dim=1)
        paper_scores = torch.stack(nonfirst_scores, dim=1)
        return all_scores, second_only, paper_scores

    def predict_both(self, images):
        scores_all, scores_second_layer, _ = self.candidate_scores(images)
        return scores_all.argmax(1), scores_second_layer.argmax(1)

    def predict_excluding_first(self, images):
        _, _, scores = self.candidate_scores(images)
        return scores.argmax(1)


def evaluate(model, loader):
    model.eval(); correct = [0, 0]; n = 0
    with torch.no_grad():
        for x, y in loader:
            p_all, p_second = model.predict_both(x)
            correct[0] += (p_all == y).sum().item()
            correct[1] += (p_second == y).sum().item()
            n += y.numel()
    return [100.0 * c / n for c in correct]


def evaluate_reference(model, loader):
    model.eval(); correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model.predict_excluding_first(x) == y).sum().item()
            n += y.numel()
    return 100.0 * correct / n


def run(train_ds, test_ds, tr, va, te, seed, repetition,
        threshold=THRESHOLD, score_test=True, hidden_layers=2, width=128,
        epochs_per_layer=EPOCHS_PER_LAYER, batch_size=BATCH_SIZE, reference=False):
    random.seed(seed); torch.manual_seed(seed)
    model = FF(width, hidden_layers)
    train_loader = DataLoader(Subset(train_ds, tr), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(Subset(train_ds, va), batch_size=batch_size, shuffle=False)
    test_loader = (DataLoader(Subset(test_ds, te), batch_size=batch_size, shuffle=False)
                   if score_test else None)
    neg_rng = torch.Generator().manual_seed(seed + 1000)
    history, best, best_state, best_epoch, selected_variant = [], -1.0, None, 0, ""
    for li, layer in enumerate(model.layers):
        optimizer = torch.optim.SGD(layer.parameters(), lr=LR)
        for epoch in range(1, epochs_per_layer + 1):
            start = time.perf_counter(); model.train()
            loss_sum = pos_sum = neg_sum = 0.0
            pos_above = neg_below = examples = 0
            for images, labels in train_loader:
                offsets = torch.randint(1, 10, labels.shape, generator=neg_rng)
                positive, negative = add_label(images, labels), add_label(images, (labels + offsets) % 10)
                with torch.no_grad():
                    for prior in model.layers[:li]:
                        positive = normalized(F.relu(prior(positive)))
                        negative = normalized(F.relu(prior(negative)))
                pg = F.relu(layer(positive)).square().sum(1)
                ng = F.relu(layer(negative)).square().sum(1)
                loss = F.softplus(threshold - pg).mean() + F.softplus(ng - threshold).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                n = labels.numel(); loss_sum += loss.item() * n
                pos_sum += pg.detach().sum().item(); neg_sum += ng.detach().sum().item(); examples += n
                pos_above += (pg.detach() > threshold).sum().item()
                neg_below += (ng.detach() < threshold).sum().item()
            elapsed = time.perf_counter() - start
            reference_val = (evaluate_reference(model, val_loader)
                             if reference and li == len(model.layers) - 1
                             and (epoch == 1 or epoch % 10 == 0) else None)
            metrics = (evaluate(model, val_loader) if li == 1 and not reference
                       else [None, None])
            choice = (max(range(2), key=lambda j: metrics[j])
                      if li == 1 and not reference else None)
            is_best = False
            if not reference and li == 1 and metrics[choice] > best:
                best, best_epoch, selected_variant = metrics[choice], epoch, ("all_layers" if choice == 0 else "second_layer_only")
                best_state = deepcopy(layer.state_dict()); is_best = True
            history.append(dict(record_type="epoch", seed=seed, repetition=repetition,
                layer=li + 1, epoch=epoch, threshold=threshold, local_loss=loss_sum/examples,
                mean_positive_goodness=pos_sum/examples, mean_negative_goodness=neg_sum/examples,
                goodness_gap=(pos_sum-neg_sum)/examples,
                positive_above_threshold_pct=100.0*pos_above/examples,
                negative_below_threshold_pct=100.0*neg_below/examples,
                validation_accuracy_all_layers_pct=metrics[0],
                validation_accuracy_second_layer_only_pct=metrics[1],
                selected_best_epoch=is_best, train_seconds=elapsed,
                reference_validation_accuracy_excluding_first_pct=reference_val))
        # Earlier layers are not optimized again; their parameters have no gradients here.
    if reference:
        final_validation = evaluate_reference(model, val_loader)
        final_metrics = ([evaluate_reference(model, test_loader), None]
                         if score_test else [None, None])
        test_accuracy = final_metrics[0]
        best_epoch, selected_variant, best = epochs_per_layer, "goodness_excluding_first", final_validation
    elif best_state is None:
        raise RuntimeError("Validation did not select a layer-2 checkpoint")
    else:
        model.layers[1].load_state_dict(best_state)
        final_metrics = evaluate(model, test_loader) if score_test else [None, None]
        test_accuracy = (final_metrics[0 if selected_variant == "all_layers" else 1]
                         if score_test else None)
    if reference:
        summary = dict(record_type="reference_ff_seed_summary", seed=seed, repetition=1,
            reference_id="hinton_ff_sanity_v1", final_validation_accuracy_pct=final_validation,
            final_test_accuracy_excluding_first_pct=test_accuracy,
            training_seconds=sum(r["train_seconds"] for r in history),
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            train_samples=len(tr), validation_samples=len(va), test_samples=len(te))
    else:
        summary = dict(record_type="seed_summary", seed=seed, repetition=repetition,
            threshold=threshold,
            selected_layer2_epoch=best_epoch, selected_inference_variant=selected_variant,
            best_validation_accuracy_pct=best, final_test_accuracy_pct=test_accuracy,
            final_test_all_layers_pct=final_metrics[0], final_test_second_layer_only_pct=final_metrics[1],
            training_seconds=sum(r["train_seconds"] for r in history),
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            train_samples=len(tr), validation_samples=len(va), test_samples=len(te))
    return history, summary


def reference_indices():
    """Full-data Hinton-style MNIST split, excluding legacy-exposed test cases."""
    rng = torch.Generator().manual_seed(42)
    train_perm = torch.randperm(60000, generator=rng)
    test_perm = torch.randperm(10000, generator=rng)
    return (train_perm[:50000].tolist(), train_perm[50000:].tolist(),
            test_perm[OLD_TEST_N:].tolist())


def run_reference_check():
    """Append a narrower source-aligned FF check without replacing baseline records."""
    if not OUT.is_file():
        raise FileNotFoundError(f"Canonical FF CSV is required: {OUT}")
    with OUT.open(newline="") as f:
        existing = list(csv.DictReader(f))
    if any(r.get("reference_id") == "hinton_ff_sanity_v1" for r in existing):
        raise RuntimeError("FF reference records already exist in the canonical CSV")
    train, test, _, _, _ = get_data()
    tr, va, te = reference_indices()
    seeds = (42, 43, 44)
    records, summaries = [], []
    for seed in seeds:
        history, summary = run(train, test, tr, va, te, seed, 1,
            threshold=2.0, score_test=True, hidden_layers=4, width=128,
            epochs_per_layer=60, batch_size=BATCH_SIZE, reference=True)
        for row in history:
            row.update(record_type="reference_ff_epoch", reference_id="hinton_ff_sanity_v1",
                architecture="784-128-128-128-128; ReLU; L2 normalization",
                negative_strategy="uniform random incorrect label (project method; paper uses hard negatives)",
                inference="candidate-label goodness sum excluding layer 1")
            records.append(row)
        summary.update(architecture="784-128-128-128-128; ReLU; L2 normalization",
            epochs_per_layer=60, batch_size=BATCH_SIZE, threshold=2.0,
            negative_strategy="uniform random incorrect label (project method; paper uses hard negatives)",
            inference="candidate-label goodness sum excluding layer 1",
            test_indices="official test permutation positions 1000:10000")
        summaries.append(summary)
        records.append(summary)
        print(f"FF reference seed={seed}: val={summary['final_validation_accuracy_pct']:.2f}% "
              f"test={summary['final_test_accuracy_excluding_first_pct']:.2f}% "
              f"train={summary['training_seconds']:.1f}s")
    vals = [r["final_test_accuracy_excluding_first_pct"] for r in summaries]
    times = [r["training_seconds"] for r in summaries]
    records.append(dict(record_type="reference_ff_aggregate", reference_id="hinton_ff_sanity_v1",
        final_test_accuracy_mean_pct=statistics.mean(vals),
        final_test_accuracy_sd_pct=statistics.stdev(vals),
        final_test_accuracy_min_pct=min(vals), final_test_accuracy_max_pct=max(vals),
        training_seconds_mean=statistics.mean(times), training_seconds_sd=statistics.stdev(times),
        training_seconds_min=min(times), training_seconds_max=max(times), seeds="42-44",
        train_samples=len(tr), validation_samples=len(va), test_samples=len(te),
        reference_note="source-aligned reduced-width check; not a Hinton reproduction"))
    all_rows = existing + records
    fields = sorted({k for row in all_rows for k in row})
    temp = OUT.with_name(".ff_mnist_results.reference.tmp")
    try:
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
        temp.replace(OUT)
    finally:
        if temp.exists():
            temp.unlink()
    print(f"Appended {len(records)} FF reference records to {OUT}; preserved {len(existing)} prior rows")


def main():
    if "--reference-check" in sys.argv:
        run_reference_check()
        return
    preserved_reference = []
    if OUT.is_file():
        with OUT.open(newline="") as f:
            preserved_reference = [r for r in csv.DictReader(f)
                                   if r.get("record_type", "").startswith("reference_ff_")]
    if not BASELINE_ARCHIVE.is_file():
        raise FileNotFoundError(f"Preserved baseline CSV is required: {BASELINE_ARCHIVE}")
    with BASELINE_ARCHIVE.open(newline="") as f:
        baseline_rows = list(csv.DictReader(f))
    train, test, tr, va, te = get_data()

    # Threshold selection is validation-only. Test loaders are not created for these runs.
    diagnostic_rows, diagnostic_summaries, aggregates = [], [], []
    for threshold in THRESHOLD_GRID:
        summaries_for_threshold = []
        for seed in SEEDS:
            history, summary = run(train, test, tr, va, te, seed, 1,
                                   threshold=threshold, score_test=False)
            for row in history:
                row["record_type"] = "threshold_diagnostic_epoch"
                diagnostic_rows.append(row)
            summary.update(record_type="threshold_diagnostic_seed", threshold=threshold)
            diagnostic_summaries.append(summary)
            summaries_for_threshold.append(summary)
        val_scores = [float(row["best_validation_accuracy_pct"]) for row in summaries_for_threshold]
        aggregates.append(dict(record_type="threshold_diagnostic_aggregate", threshold=threshold,
            mean_best_validation_accuracy_pct=statistics.mean(val_scores),
            validation_accuracy_std_pct=statistics.stdev(val_scores),
            validation_accuracy_min_pct=min(val_scores), validation_accuracy_max_pct=max(val_scores),
            selected_variant_counts=";".join(f"{variant}:{sum(r['selected_inference_variant']==variant for r in summaries_for_threshold)}"
                                               for variant in ("all_layers", "second_layer_only")),
            seeds="42-46", test_evaluated=False))
        print(f"validation-only threshold={threshold:g}: mean best validation="
              f"{statistics.mean(val_scores):.2f}%")

    # Prefer the established threshold on an exact validation tie.
    selected = max(aggregates, key=lambda row: (
        float(row["mean_best_validation_accuracy_pct"]),
        float(row["threshold"]) == THRESHOLD))
    selected_threshold = float(selected["threshold"])
    selection_row = dict(record_type="threshold_selection", selected_threshold=selected_threshold,
        threshold_grid=",".join(map(str, THRESHOLD_GRID)),
        criterion="highest mean best validation accuracy across seeds 42-46",
        test_evaluated_during_selection=False)
    baseline_records, selected_records = [], []
    if selected_threshold == THRESHOLD:
        # Existing test results were produced only after validation selected this same config.
        baseline_records.extend(baseline_rows)
    else:
        for row in baseline_rows:
            row = dict(row)
            if row.get("record_type") in ("epoch", "seed_summary", "aggregate"):
                row["record_type"] = "baseline_" + row["record_type"]
            baseline_records.append(row)
        summaries = []
        for repetition in range(1, TIMING_REPETITIONS + 1):
            for seed in SEEDS:
                history, summary = run(train, test, tr, va, te, seed, repetition,
                                       threshold=selected_threshold, score_test=True)
                selected_records.extend(history); summaries.append(summary)
                print(f"selected FF threshold={selected_threshold:g} rep={repetition} seed={seed}: "
                      f"val={summary['best_validation_accuracy_pct']:.2f}% "
                      f"test={summary['final_test_accuracy_pct']:.2f}%")
        values = [next(r["final_test_accuracy_pct"] for r in summaries if r["seed"] == seed)
                  for seed in SEEDS]
        times = [r["training_seconds"] for r in summaries]
        selected_records.extend(summaries)
        selected_records.append(dict(record_type="aggregate", seed="all", repetition="all",
            threshold=selected_threshold,
            final_test_accuracy_pct=statistics.mean(values),
            test_accuracy_std_pct=statistics.stdev(values), test_accuracy_min_pct=min(values),
            test_accuracy_max_pct=max(values), training_seconds_mean=statistics.mean(times),
            training_seconds_std=statistics.stdev(times), training_seconds_min=min(times),
            training_seconds_max=max(times), repetitions=TIMING_REPETITIONS, seeds="42-46"))
    diagnostic_rows.extend(diagnostic_summaries)
    # Save diagnostics and the selection decision ahead of any final-test records.
    rows = baseline_records + diagnostic_rows + aggregates + [selection_row] + selected_records
    common = dict(architecture="784-128-128; ReLU; L2 normalization", optimizer="per-layer SGD",
        learning_rate=LR, batch_size=BATCH_SIZE, epochs_per_layer=EPOCHS_PER_LAYER,
        train_indices_seed=42, legacy_test_excluded=OLD_TEST_N,
        final_test_indices="official test permutation positions 1000:2000")
    for row in rows:
        row.update(common)
        if row.get("record_type", "").startswith("threshold_diagnostic"):
            row["diagnostic_note"] = "validation-only; test was not evaluated"
    fields = sorted({k for row in rows + preserved_reference for k in row})
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        writer.writerows(preserved_reference)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__": main()
