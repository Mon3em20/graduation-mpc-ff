"""Independent comparison script for Secure BP Matrix Prototype numerical audit.

Compares MPC diagnostic reveals against the FP64 reference.
Computes:
- Tensor name and shape verification
- Maximum absolute error
- Mean absolute error
- Root mean square error (RMSE)
- Relative error under documented epsilon policies:
    1. Standard small epsilon (eps = 1e-8)
    2. Fixed-point quantum epsilon (eps = 2^-16 ~= 1.52588e-5, the physical resolution of sfix(16, 31))
- Relative L2 (Frobenius) norm error: ||actual - expected||_2 / ||expected||_2
- Exact prediction agreement
- Pre-update vs post-update categorization
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

FIXED_POINT_QUANTUM = 2.0 ** -16  # 1.52587890625e-05


def parse_audit_log(log_path):
    """Extract AUDIT_TENSOR lines from MP-SPDZ execution log."""
    tensors = {}
    lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    in_audit = False

    for line in lines:
        line = line.strip()
        if line == "AUDIT_START":
            in_audit = True
            continue
        elif line == "AUDIT_END":
            in_audit = False
            continue

        if not in_audit and not line.startswith("AUDIT_TENSOR"):
            continue

        if line.startswith("AUDIT_TENSOR"):
            match = re.match(r"^AUDIT_TENSOR\s+name=(\w+)\s+shape=([0-9,]*)\s+values=(.+)$", line)
            if not match:
                raise ValueError(f"Failed to parse AUDIT_TENSOR line: {line[:100]}...")
            name = match.group(1)
            shape_str = match.group(2)
            values_str = match.group(3).strip()

            if shape_str:
                shape = tuple(int(dim) for dim in shape_str.split(","))
            else:
                shape = ()

            try:
                values = json.loads(values_str)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON values for tensor {name}: {exc}") from exc

            tensor = torch.tensor(values, dtype=torch.float64)
            if shape and tuple(tensor.shape) != shape:
                tensor = tensor.reshape(shape)
            tensors[name] = tensor

    aggregate = {}
    for line in lines:
        agg_match = re.search(r"train_loss=([-+0-9.eE]+)\s+correct=(\d+)\s+total=(\d+)", line)
        if agg_match:
            aggregate["train_loss"] = float(agg_match.group(1))
            aggregate["correct"] = int(agg_match.group(2))
            aggregate["total"] = int(agg_match.group(3))
            break

    return tensors, aggregate


def compute_tensor_metrics(actual, expected):
    """Compute comprehensive numerical comparison metrics between actual and expected tensors.
    
    Documented epsilon policies:
    1. eps_small = 1e-8: standard numerical safeguard against division by exact zero.
       When |expected| < 1e-5 (below fixed-point resolution), element-wise relative error
       can appear artificially elevated due to near-zero denominators.
    2. eps_quantum = 2^-16 (~1.52588e-5): matches the minimum representable positive increment
       of sfix(16, 31). Prevents noise magnification for values smaller than fixed-point granularity.
    3. relative_l2_error: Frobenius/L2 norm ratio ||a - e||_2 / (||e||_2 + 1e-8), invariant to individual
       near-zero elements.
    """
    diff = (actual - expected).abs()
    max_abs_err = float(diff.max().item())
    mean_abs_err = float(diff.mean().item())
    rmse = float(torch.sqrt((diff ** 2).mean()).item())

    l2_expected = float(torch.linalg.vector_norm(expected).item())
    l2_diff = float(torch.linalg.vector_norm(diff).item())
    rel_l2_err = l2_diff / (l2_expected + 1e-8)

    # Policy 1: eps = 1e-8
    elem_rel_1e8 = diff / (expected.abs() + 1e-8)
    max_rel_1e8 = float(elem_rel_1e8.max().item())
    mean_rel_1e8 = float(elem_rel_1e8.mean().item())

    # Policy 2: eps = quantum (2^-16)
    elem_rel_q = diff / (expected.abs() + FIXED_POINT_QUANTUM)
    max_rel_q = float(elem_rel_q.max().item())
    mean_rel_q = float(elem_rel_q.mean().item())

    return {
        "shape_actual": list(actual.shape),
        "shape_expected": list(expected.shape),
        "shape_match": tuple(actual.shape) == tuple(expected.shape),
        "element_count": actual.numel(),
        "max_abs_error": max_abs_err,
        "mean_abs_error": mean_abs_err,
        "rmse": rmse,
        "relative_l2_error": rel_l2_err,
        "max_relative_error_eps_1e8": max_rel_1e8,
        "mean_relative_error_eps_1e8": mean_rel_1e8,
        "max_relative_error_eps_quantum": max_rel_q,
        "mean_relative_error_eps_quantum": mean_rel_q,
        "l2_norm_actual": float(torch.linalg.vector_norm(actual).item()),
        "l2_norm_expected": l2_expected,
        "fixed_point_quantum": FIXED_POINT_QUANTUM,
    }


def compare_audit(log_path, ref_pt_path, summary_json_path=None, output_json_path=None):
    mpc_tensors, aggregate = parse_audit_log(log_path)
    ref_tensors = torch.load(ref_pt_path, weights_only=False)

    summary_data = {}
    if summary_json_path and Path(summary_json_path).exists():
        summary_data = json.loads(Path(summary_json_path).read_text(encoding="utf-8"))

    results = {
        "metadata": {
            "log_path": str(log_path),
            "ref_pt_path": str(ref_pt_path),
            "summary_json_path": str(summary_json_path) if summary_json_path else None,
            "aggregate_reported": aggregate,
            "fixed_point_specification": {
                "type": "sfix(16, 31)",
                "fractional_bits": 16,
                "total_bits": 31,
                "resolution": FIXED_POINT_QUANTUM,
            },
        },
        "pre_update": {
            "forward": {},
            "backward": {},
        },
        "post_update": {},
        "overall": {
            "all_shapes_match": True,
            "exact_predictions_match": None,
            "max_abs_error_overall": 0.0,
            "max_rel_l2_error_overall": 0.0,
        },
    }

    forward_names = ["Z1", "H1", "Z2", "H2", "Z3", "P", "loss_each", "mean_loss"]
    backward_names = ["D3", "D2", "D1", "dW3", "db3", "dW2", "db2", "dW1", "db1"]
    post_update_names = ["W1_new", "b1_new", "W2_new", "b2_new", "W3_new", "b3_new"]

    # Special handling for predictions
    if "predictions" in mpc_tensors:
        mpc_preds = mpc_tensors["predictions"].long().tolist()
        ref_preds = summary_data.get("predictions")
        if ref_preds is None and "Z3" in ref_tensors:
            ref_preds = ref_tensors["Z3"].argmax(dim=1).tolist()
        pred_match = (mpc_preds == ref_preds)
        results["overall"]["exact_predictions_match"] = pred_match
        results["pre_update"]["forward"]["predictions"] = {
            "mpc_predictions": mpc_preds,
            "reference_predictions": ref_preds,
            "exact_match": pred_match,
        }

    # Compare mean_loss
    if "mean_loss" in mpc_tensors:
        ref_mean = None
        if "fp64_mean_loss" in summary_data:
            ref_mean = torch.tensor([summary_data["fp64_mean_loss"]], dtype=torch.float64)
        elif "loss_each" in ref_tensors:
            ref_mean = torch.tensor([ref_tensors["loss_each"].mean().item()], dtype=torch.float64)
        if ref_mean is not None:
            actual_mean = mpc_tensors["mean_loss"].view(1)
            metrics = compute_tensor_metrics(actual_mean, ref_mean)
            results["pre_update"]["forward"]["mean_loss"] = metrics
            results["overall"]["max_abs_error_overall"] = max(
                results["overall"]["max_abs_error_overall"], metrics["max_abs_error"]
            )
            results["overall"]["max_rel_l2_error_overall"] = max(
                results["overall"]["max_rel_l2_error_overall"], metrics["relative_l2_error"]
            )

    category_map = [
        (forward_names, results["pre_update"]["forward"]),
        (backward_names, results["pre_update"]["backward"]),
        (post_update_names, results["post_update"]),
    ]

    for names, target_dict in category_map:
        for name in names:
            if name == "mean_loss":
                continue
            if name not in mpc_tensors:
                target_dict[name] = {"error": "Missing from MPC audit output"}
                results["overall"]["all_shapes_match"] = False
                continue
            if name not in ref_tensors:
                target_dict[name] = {"error": "Missing from FP64 reference"}
                continue

            actual = mpc_tensors[name]
            expected = ref_tensors[name]

            metrics = compute_tensor_metrics(actual, expected)
            target_dict[name] = metrics

            if not metrics["shape_match"]:
                results["overall"]["all_shapes_match"] = False

            results["overall"]["max_abs_error_overall"] = max(
                results["overall"]["max_abs_error_overall"], metrics["max_abs_error"]
            )
            results["overall"]["max_rel_l2_error_overall"] = max(
                results["overall"]["max_rel_l2_error_overall"], metrics["relative_l2_error"]
            )

    if output_json_path:
        Path(output_json_path).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    return results


def print_summary_table(results):
    print("=" * 115)
    print(f"{'Category':<14} {'Tensor':<10} {'Shape':<15} {'Max Abs Err':<14} {'Mean Abs Err':<14} {'RMSE':<14} {'Rel L2 Err':<14} {'Rel (eps=q)':<14}")
    print("-" * 115)

    def print_section(section_name, d):
        for name, metrics in d.items():
            if name == "predictions":
                match_str = "MATCH" if metrics.get("exact_match") else "MISMATCH"
                print(f"{section_name:<14} {'preds':<10} {'(3,)':<15} {match_str:<14} {str(metrics.get('mpc_predictions')):<14}")
                continue
            if "error" in metrics:
                print(f"{section_name:<14} {name:<10} {'ERROR':<15} {metrics['error']}")
                continue
            shape_str = str(tuple(metrics["shape_expected"]))
            print(f"{section_name:<14} {name:<10} {shape_str:<15} {metrics['max_abs_error']:<14.6e} {metrics['mean_abs_error']:<14.6e} {metrics['rmse']:<14.6e} {metrics['relative_l2_error']:<14.6e} {metrics['max_relative_error_eps_quantum']:<14.6e}")

    print_section("Fwd (pre)", results["pre_update"]["forward"])
    print("-" * 115)
    print_section("Bwd (pre)", results["pre_update"]["backward"])
    print("-" * 115)
    print_section("SGD (post)", results["post_update"])
    print("=" * 115)
    print(f"Overall Max Absolute Error: {results['overall']['max_abs_error_overall']:.6e}")
    print(f"Overall Max Rel L2 Error  : {results['overall']['max_rel_l2_error_overall']:.6e}")
    print(f"Predictions Exact Match   : {results['overall']['exact_predictions_match']}")
    print(f"All Shapes Match          : {results['overall']['all_shapes_match']}")


def main():
    parser = argparse.ArgumentParser(description="Secure BP Matrix Prototype Numerical Audit Comparison")
    parser.add_argument("--log", type=Path, required=True, help="Path to MPC diagnostic log")
    parser.add_argument("--reference", type=Path, default=Path("/tmp/matrix_proto_fp64_reference_seed42.pt"), help="Path to FP64 reference .pt")
    parser.add_argument("--summary", type=Path, default=Path("/tmp/matrix_proto_fp64_summary_seed42.json"), help="Path to FP64 summary json")
    parser.add_argument("--output", type=Path, default=None, help="Output path for comparison json")
    args = parser.parse_args()

    results = compare_audit(args.log, args.reference, args.summary, args.output)
    print_summary_table(results)


if __name__ == "__main__":
    main()
