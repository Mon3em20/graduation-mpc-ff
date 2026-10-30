#!/usr/bin/env python3
"""Standardized plaintext BP/FF benchmark.

This runner deliberately writes only below results/benchmarks/plaintext.
Historical baseline CSVs are never read as output targets or overwritten.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmarks" / "plaintext"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_REPETITIONS = 3
FF_BENCHMARK_INFERENCE_VARIANT = "all_layers"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BP = load_module("benchmark_bp", ROOT / "src" / "baselines" / "bp.py")
FF = load_module("benchmark_ff", ROOT / "src" / "baselines" / "ff.py")


def rss_mb():
    # This is process-wide high-water mark data, not an independent run peak.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def timed_call(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - start


def load_data():
    start = time.perf_counter()
    train, test, tr, va, te = BP.datasets_and_indices()
    return (train, test, tr, va, te), time.perf_counter() - start


def update_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_bp(data, seed, repetition, epochs, batch_size):
    train, test, tr, va, te = data
    random.seed(seed)
    torch.manual_seed(seed)
    init_start = time.perf_counter()
    model = BP.make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=BP.LR)
    init_time = time.perf_counter() - init_start
    loader_start = time.perf_counter()
    train_loader = DataLoader(Subset(train, tr), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(Subset(train, va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(test, te), batch_size=batch_size, shuffle=False)
    loader_time = time.perf_counter() - loader_start
    criterion = nn.CrossEntropyLoss()
    best_acc, best_epoch, best_state = -1.0, None, None
    epoch_rows = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, updates = BP.train_epoch(model, train_loader, optimizer, criterion)
        train_epoch_time = time.perf_counter() - epoch_start
        val_start = time.perf_counter()
        val_loss, val_acc = BP.evaluate(model, val_loader)
        val_time = time.perf_counter() - val_start
        best_acc, selected_epoch, selected_state, selected = BP.select_checkpoint(
            best_acc, val_acc, epoch, model)
        if selected:
            best_epoch, best_state = selected_epoch, selected_state
        epoch_rows.append({
            "algorithm": "BP", "profile": "project_baseline_standardized",
            "seed": seed, "repetition": repetition, "progress_unit": "epoch",
            "epoch": epoch, "layer": "", "training_loss": train_loss,
            "validation_loss": val_loss, "validation_accuracy_pct": val_acc,
            "training_time_seconds": train_epoch_time,
            "validation_time_seconds": val_time,
            "optimizer_updates": updates, "selected_checkpoint": selected,
            "epoch_wall_seconds": time.perf_counter() - epoch_start,
        })
    total_training = sum(row["training_time_seconds"] for row in epoch_rows)
    model.load_state_dict(best_state)
    inference_start = time.perf_counter()
    test_loss, test_acc = BP.evaluate(model, test_loader)
    inference_time = time.perf_counter() - inference_start
    return epoch_rows, {
        "algorithm": "BP", "profile": "project_baseline_standardized",
        "seed": seed, "repetition": repetition,
        "architecture": "784-128-128-10 ReLU",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_samples": len(tr), "validation_samples": len(va), "test_samples": len(te),
        "batch_size": batch_size, "epochs": epochs, "layers": 1,
        "training_time_seconds": total_training, "validation_time_seconds":
        sum(float(r["validation_time_seconds"]) for r in epoch_rows),
        "inference_time_seconds": inference_time, "model_initialization_seconds": init_time,
        "data_loader_seconds": loader_time, "total_experiment_seconds":
        loader_time + init_time + total_training + sum(float(r["validation_time_seconds"])
                                                       for r in epoch_rows) + inference_time,
        "peak_rss_mb": rss_mb(), "optimizer_updates": sum(
            row["optimizer_updates"] for row in epoch_rows),
        "training_presentations": len(tr) * epochs, "positive_presentations": "",
        "negative_presentations": "", "negative_sampling_seconds": "",
        "selected_checkpoint": best_epoch, "best_validation_accuracy_pct": best_acc,
        "test_accuracy_pct": test_acc, "test_loss": test_loss,
    }


def benchmark_ff(data, seed, repetition, epochs_per_layer, batch_size):
    train, test, tr, va, te = data
    random.seed(seed)
    torch.manual_seed(seed)
    init_start = time.perf_counter()
    model = FF.FF(width=128, hidden_layers=2)
    init_time = time.perf_counter() - init_start
    loader_start = time.perf_counter()
    train_loader = DataLoader(Subset(train, tr), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(Subset(train, va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(test, te), batch_size=batch_size, shuffle=False)
    loader_time = time.perf_counter() - loader_start
    neg_rng = torch.Generator().manual_seed(seed + 1000)
    best_acc, best_checkpoint, best_state = -1.0, None, None
    epoch_rows = []
    total_training = 0.0
    validation_total = 0.0
    updates_total = 0
    sampling_total = 0.0
    for layer_index, layer in enumerate(model.layers):
        optimizer = torch.optim.SGD(layer.parameters(), lr=FF.LR)
        for epoch in range(1, epochs_per_layer + 1):
            metrics = FF.train_layer_epoch(
                model, layer_index, train_loader, optimizer, FF.THRESHOLD, neg_rng)
            train_epoch_time = metrics["train_seconds"]
            total_training += train_epoch_time
            sampling_total += metrics["negative_sampling_seconds"]
            updates_total += metrics["optimizer_updates"]
            # Only layer 2 is a valid full-network checkpoint for this profile.
            val_time = 0.0
            val_acc = ""
            selected = False
            if layer_index == len(model.layers) - 1:
                val_start = time.perf_counter()
                val_acc = FF.evaluate(model, val_loader)[0]
                val_time = time.perf_counter() - val_start
                validation_total += val_time
                best_acc, best_checkpoint, selected_state, selected = (
                    FF.select_layer_checkpoint(
                        best_acc, val_acc, f"{layer_index + 1}:{epoch}", layer))
                if selected:
                    best_state = selected_state
            epoch_rows.append({
                "algorithm": "FF", "profile": "project_baseline_standardized",
                "seed": seed, "repetition": repetition,
                "progress_unit": "layer_epoch", "epoch": epoch,
                "layer": layer_index + 1, "training_loss": metrics["local_loss"],
                "validation_loss": "", "validation_accuracy_pct": val_acc,
                "training_time_seconds": train_epoch_time,
                "validation_time_seconds": val_time,
                "optimizer_updates": metrics["optimizer_updates"], "selected_checkpoint": selected,
                "mean_positive_goodness": metrics["mean_positive_goodness"],
                "mean_negative_goodness": metrics["mean_negative_goodness"],
                "negative_sampling_seconds": metrics["negative_sampling_seconds"],
            })
    model.layers[-1].load_state_dict(best_state)
    inference_start = time.perf_counter()
    final_metrics = FF.evaluate(model, test_loader)
    test_acc = final_metrics[0 if FF_BENCHMARK_INFERENCE_VARIANT == "all_layers" else 1]
    inference_time = time.perf_counter() - inference_start
    return epoch_rows, {
        "algorithm": "FF", "profile": "project_baseline_standardized",
        "seed": seed, "repetition": repetition,
        "architecture": "784-128-128; ReLU; L2 normalization",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_samples": len(tr), "validation_samples": len(va), "test_samples": len(te),
        "batch_size": batch_size, "epochs": epochs_per_layer, "layers": 2,
        "training_time_seconds": total_training, "validation_time_seconds": validation_total,
        "inference_time_seconds": inference_time, "model_initialization_seconds": init_time,
        "data_loader_seconds": loader_time, "total_experiment_seconds":
        loader_time + init_time + total_training + validation_total + inference_time,
        "peak_rss_mb": rss_mb(), "optimizer_updates": updates_total,
        "training_presentations": len(tr) * epochs_per_layer * len(model.layers) * 2,
        "positive_presentations": len(tr) * epochs_per_layer * len(model.layers),
        "negative_presentations": len(tr) * epochs_per_layer * len(model.layers),
        "negative_sampling_seconds": sampling_total,
        "selected_checkpoint": best_checkpoint, "best_validation_accuracy_pct": best_acc,
        "test_accuracy_pct": test_acc, "test_loss": "",
    }


def derive_targets(epoch_rows):
    values = {}
    for algorithm in ("BP", "FF"):
        vals = [float(row["validation_accuracy_pct"]) for row in epoch_rows
                if row["algorithm"] == algorithm and row["validation_accuracy_pct"] != ""]
        if vals:
            values[algorithm] = (min(vals), max(vals))
    if len(values) != 2:
        return []
    low = max(values["BP"][0], values["FF"][0])
    high = min(values["BP"][1], values["FF"][1])
    if low >= high:
        return []
    start = math.ceil(low / 5) * 5
    stop = math.floor(high / 5) * 5
    return list(range(start, stop + 1, 5))


def _row_order(row):
    if row["algorithm"] == "FF":
        return int(row["layer"]), int(row["epoch"])
    return int(row["epoch"]), 0


def matched_accuracy(rows, targets):
    output = []
    for target in targets:
        for algorithm in ("BP", "FF"):
            keys = sorted({(r["seed"], r["repetition"]) for r in rows
                           if r["algorithm"] == algorithm})
            for seed, repetition in keys:
                run_rows = [r for r in rows if r["algorithm"] == algorithm and
                            r["seed"] == seed and r["repetition"] == repetition]
                candidates = [r for r in run_rows
                              if r["validation_accuracy_pct"] != "" and
                              float(r["validation_accuracy_pct"]) >= target]
                if not candidates:
                    output.append({"target_accuracy_pct": target, "algorithm": algorithm,
                                   "seed": seed, "repetition": repetition,
                                   "status": "not_reached", "checkpoint": "",
                                   "training_only_seconds": "", "elapsed_seconds": ""})
                    continue
                first = min(candidates, key=_row_order)
                if algorithm == "BP":
                    prior = [r for r in run_rows if int(r["epoch"]) <= int(first["epoch"])]
                else:
                    # A layer-2 checkpoint includes every layer-1 epoch first.
                    prior = [r for r in run_rows if _row_order(r) <= _row_order(first)]
                training_time = sum(float(r["training_time_seconds"]) for r in prior)
                elapsed = training_time + sum(float(r["validation_time_seconds"])
                                             for r in prior)
                output.append({"target_accuracy_pct": target, "algorithm": algorithm,
                               "seed": seed, "repetition": repetition,
                               "status": "reached",
                               "checkpoint": f"{first['layer']}:{first['epoch']}" if algorithm == "FF"
                               else first["epoch"],
                               "training_only_seconds": training_time,
                               "elapsed_seconds": elapsed})
    return output


def rebuild_runs(runs, epoch_rows):
    """Repair run-level timing/count fields from immutable epoch measurements."""
    repaired = []
    for run in runs:
        key_rows = [row for row in epoch_rows
                    if row["algorithm"] == run["algorithm"]
                    and int(row["seed"]) == int(run["seed"])
                    and int(row["repetition"]) == int(run["repetition"])]
        training = sum(float(row["training_time_seconds"]) for row in key_rows)
        validation = sum(float(row["validation_time_seconds"]) for row in key_rows)
        updates = sum(int(row["optimizer_updates"]) for row in key_rows)
        repaired_row = dict(run)
        repaired_row["training_time_seconds"] = training
        repaired_row["validation_time_seconds"] = validation
        repaired_row["total_experiment_seconds"] = (
            float(run["data_loader_seconds"]) +
            float(run["model_initialization_seconds"]) +
            training + validation + float(run["inference_time_seconds"]))
        repaired_row["optimizer_updates"] = updates
        if run["algorithm"] == "BP":
            repaired_row["training_presentations"] = (
                int(run["train_samples"]) * int(run["epochs"]))
        repaired.append(repaired_row)
    return repaired


def summarize_runs(runs, repetitions):
    summary_rows = []
    for algorithm in sorted({row["algorithm"] for row in runs}):
        subset = [row for row in runs if row["algorithm"] == algorithm]
        timing = [float(row["training_time_seconds"]) for row in subset]
        utility = {}
        for row in subset:
            utility.setdefault(int(row["seed"]), row)
        accuracies = [float(row["test_accuracy_pct"]) for row in utility.values()]
        summary_rows.append({
            "algorithm": algorithm, "profile": "project_baseline_standardized",
            "mean_test_accuracy_pct": statistics.mean(accuracies),
            "test_accuracy_sd_pct": statistics.stdev(accuracies) if len(accuracies) > 1 else "",
            "test_accuracy_min_pct": min(accuracies), "test_accuracy_max_pct": max(accuracies),
            "mean_training_seconds": statistics.mean(timing),
            "training_seconds_sd": statistics.stdev(timing) if len(timing) > 1 else "",
            "training_seconds_median": statistics.median(timing),
            "training_seconds_min": min(timing), "training_seconds_max": max(timing),
            "mean_time_per_update_seconds": statistics.mean(
                float(row["training_time_seconds"]) / int(row["optimizer_updates"])
                for row in subset),
            "mean_time_per_presentation_seconds": statistics.mean(
                float(row["training_time_seconds"]) / float(row["training_presentations"])
                for row in subset),
            "mean_inference_seconds": statistics.mean(
                float(row["inference_time_seconds"]) for row in subset),
            "memory_status": "unavailable: process-wide RSS high-water mark",
            "independent_seeds": len(utility),
            "timing_runs": len(subset),
            "timing_repetitions": repetitions,
        })
    return summary_rows


def summarize_matched(rows):
    output = []
    for target in sorted({float(row["target_accuracy_pct"]) for row in rows}):
        for algorithm in ("BP", "FF"):
            subset = [row for row in rows if float(row["target_accuracy_pct"]) == target
                      and row["algorithm"] == algorithm and row["status"] == "reached"]
            total = [row for row in rows if float(row["target_accuracy_pct"]) == target
                     and row["algorithm"] == algorithm]
            times = [float(row["training_only_seconds"]) for row in subset]
            output.append({
                "target_accuracy_pct": target,
                "algorithm": algorithm,
                "reached_runs": len(subset),
                "total_runs": len(total),
                "reachability_pct": 100.0 * len(subset) / len(total) if total else "",
                "mean_training_only_seconds": statistics.mean(times) if times else "",
                "training_only_sd_seconds": statistics.stdev(times) if len(times) > 1 else "",
                "min_training_only_seconds": min(times) if times else "",
                "max_training_only_seconds": max(times) if times else "",
            })
    return output


def write_metadata(args, data, targets):
    train, test, tr, va, te = data
    metadata = {
        "benchmark_id": "plaintext_standardized_v1",
        "profile": "project_baseline_standardized",
        "command": " ".join(sys.argv),
        "python": sys.version,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "seeds": args.seeds,
        "timing_repetitions": args.repetitions,
        "train_samples": len(tr), "validation_samples": len(va), "test_samples": len(te),
        "test_exclusion": "official test permutation positions 0:1000 excluded; positions 1000:2000 evaluated",
        "validation_targets_pct": targets,
        "memory_method": "not used for method comparison: current-process RSS high-water mark",
        "ff_threshold": FF.THRESHOLD,
        "ff_threshold_decision": (
            "v1 intentionally uses the source default 2.0; historical validation-selected "
            "threshold-3.0 records remain separate and were not replaced"),
        "historical_results_untouched": True,
        "source_sha256": {
            "src/baselines/bp.py": hashlib.sha256(
                (ROOT / "src/baselines/bp.py").read_bytes()).hexdigest(),
            "src/baselines/ff.py": hashlib.sha256(
                (ROOT / "src/baselines/ff.py").read_bytes()).hexdigest(),
            "scripts/benchmark.py": hashlib.sha256(
                (ROOT / "scripts/benchmark.py").read_bytes()).hexdigest(),
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def generate_figures(epoch_rows, run_rows, matched_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"BP": "#1f77b4", "FF": "#d62728"}

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for axis, algorithm, title in (
        (axes[0], "BP", "BP: epoch"),
        (axes[1], "FF", "FF: layer-2 epoch (after layer 1)"),
    ):
        points = {}
        for row in epoch_rows:
            if row["algorithm"] != algorithm or row["validation_accuracy_pct"] == "":
                continue
            progress = int(row["epoch"])
            points.setdefault(progress, []).append(float(row["validation_accuracy_pct"]))
        if points:
            x = sorted(points)
            y = [statistics.mean(points[value]) for value in x]
            axis.plot(x, y, marker="o", color=colors[algorithm])
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation accuracy (%)")
    figure.suptitle("Validation accuracy with method-specific progress axes")
    figure.tight_layout()
    plt.savefig(figures / "accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    grouped = {algorithm: [float(r["training_time_seconds"]) for r in run_rows
                           if r["algorithm"] == algorithm] for algorithm in ("BP", "FF")}
    plt.bar(list(grouped), [statistics.mean(grouped[a]) for a in grouped],
            yerr=[statistics.stdev(grouped[a]) for a in grouped],
            color=[colors[a] for a in grouped], capsize=5)
    plt.ylabel("Training-only time (seconds)")
    plt.title("Training time across seeds and timing repetitions")
    plt.tight_layout()
    plt.savefig(figures / "training_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for algorithm in ("BP", "FF"):
        subset = [r for r in matched_rows if r["algorithm"] == algorithm and
                  r["status"] == "reached"]
        targets = sorted({float(r["target_accuracy_pct"]) for r in subset})
        means = [statistics.mean(float(r["training_only_seconds"]) for r in subset
                                 if float(r["target_accuracy_pct"]) == target)
                 if any(float(r["target_accuracy_pct"]) == target for r in subset) else 0
                 for target in targets]
        if targets:
            plt.plot(targets, means, marker="o", label=algorithm, color=colors[algorithm])
    plt.xlabel("Validation target accuracy (%)")
    plt.ylabel("Training-only time to target (seconds)")
    plt.title("Time to shared validation targets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "time_to_target.png", dpi=180)
    plt.close()

    for field, filename, ylabel, title in (
        ("inference_time_seconds", "inference_time.png", "Inference time (seconds)",
         "Selected-checkpoint inference time"),
    ):
        plt.figure(figsize=(6, 4.5))
        grouped = {algorithm: [float(r[field]) for r in run_rows
                               if r["algorithm"] == algorithm] for algorithm in ("BP", "FF")}
        plt.bar(list(grouped), [statistics.mean(grouped[a]) for a in grouped],
                yerr=[statistics.stdev(grouped[a]) for a in grouped],
                color=[colors[a] for a in grouped], capsize=5)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=180)
        plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("bp", "ff", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--quick", action="store_true", help="one seed, one repetition, one epoch/layer")
    parser.add_argument("--rebuild-existing", action="store_true",
                        help="repair outputs from existing runs.csv and epochs.csv without training")
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.repetitions = [args.seeds[0]], 1
    if args.rebuild_existing:
        with (OUT / "runs.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        with (OUT / "epochs.csv").open(newline="", encoding="utf-8") as stream:
            epochs = list(csv.DictReader(stream))
        rows = rebuild_runs(rows, epochs)
        targets = derive_targets(epochs)
        matched = matched_accuracy(epochs, targets)
        update_csv(OUT / "runs.csv", rows)
        update_csv(OUT / "epochs.csv", epochs)
        update_csv(OUT / "matched_accuracy.csv", matched)
        update_csv(OUT / "matched_summary.csv", summarize_matched(matched))
        update_csv(OUT / "summary.csv", summarize_runs(rows, args.repetitions))
        generate_figures(epochs, rows, matched)
        data, _ = load_data()
        write_metadata(args, data, targets)
        print(json.dumps({"output": str(OUT), "rebuild": True,
                          "validation_targets_pct": targets, "runs": len(rows)}, indent=2))
        return
    torch.set_num_threads(4)
    data, load_seconds = load_data()
    rows, epochs = [], []
    methods = ("bp", "ff") if args.method == "all" else (args.method,)
    for method in methods:
        for repetition in range(1, args.repetitions + 1):
            for seed in args.seeds:
                if method == "bp":
                    epoch_rows, summary = benchmark_bp(data, seed, repetition,
                                                       1 if args.quick else BP.EPOCHS, BP.BATCH_SIZE)
                else:
                    epoch_rows, summary = benchmark_ff(data, seed, repetition,
                                                       1 if args.quick else FF.EPOCHS_PER_LAYER,
                                                       FF.BATCH_SIZE)
                rows.append(summary)
                epochs.extend(epoch_rows)
    targets = derive_targets(epochs)
    matched = matched_accuracy(epochs, targets)
    OUT.mkdir(parents=True, exist_ok=True)
    update_csv(OUT / "runs.csv", rows)
    update_csv(OUT / "epochs.csv", epochs)
    update_csv(OUT / "matched_accuracy.csv", matched)
    update_csv(OUT / "matched_summary.csv", summarize_matched(matched))
    update_csv(OUT / "summary.csv", summarize_runs(rows, args.repetitions))
    write_metadata(args, data, targets)
    generate_figures(epochs, rows, matched)
    print(json.dumps({"output": str(OUT), "data_loading_seconds": load_seconds,
                      "validation_targets_pct": targets, "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
