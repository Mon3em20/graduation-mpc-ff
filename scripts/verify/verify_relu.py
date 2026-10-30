"""Independent reference and verifier for the inactive-ReLU tiny BP test."""
import ast
import sys
from pathlib import Path


EXPECTED = {
    "z1": [-0.5, 2.5],
    "h1": [0.0, 2.5],
    "y_hat": -2.25,
    "loss": 2.53125,
    "d_w2": [0.0, -5.625],
    "d_b2": -2.25,
    "d_h1": [-2.25, 2.25],
    "d_z1": [0.0, 2.25],
    "d_w1": [[0.0, 0.0], [2.25, 4.5]],
    "d_b1": [0.0, 2.25],
    "new_w2": [1.0, -0.4375],
    "new_b2": 0.475,
    "new_w1": [[1.0, 0.0], [-0.225, 0.55]],
    "new_b1": [-1.5, 0.275],
}
TOLERANCE = 2e-4


def parse_value(text):
    return ast.literal_eval(text.strip())


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield float(value)


def assert_close(name, actual, expected):
    actual_values = list(flatten(actual))
    expected_values = list(flatten(expected))
    if len(actual_values) != len(expected_values):
        raise AssertionError(f"{name}: length {len(actual_values)} != {len(expected_values)}")
    maximum = max((abs(a - e) for a, e in zip(actual_values, expected_values)), default=0.0)
    if maximum > TOLERANCE:
        raise AssertionError(f"{name}: max error {maximum} > tolerance {TOLERANCE}")
    print(f"{name}: max_abs_error={maximum:.8f} PASS")


def parse_mpc_output(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("z1="):
            values["z1"] = parse_value(line.split("=", 1)[1])
        elif line.startswith("h1="):
            values["h1"] = parse_value(line.split("=", 1)[1])
        elif line.startswith("y_hat="):
            values["y_hat"] = float(line.split("=", 1)[1])
        elif line.startswith("loss="):
            values["loss"] = float(line.split("=", 1)[1])
        elif line.startswith("d_w2="):
            left, right = line.split(" d_b2=", 1)
            values["d_w2"] = parse_value(left.split("=", 1)[1])
            values["d_b2"] = float(right)
        elif line.startswith("d_h1="):
            left, right = line.split(" d_z1=", 1)
            values["d_h1"] = parse_value(left.split("=", 1)[1])
            values["d_z1"] = parse_value(right)
        elif line.startswith("d_w1="):
            left, right = line.split(" d_b1=", 1)
            values["d_w1"] = parse_value(left.split("=", 1)[1])
            values["d_b1"] = parse_value(right)
        elif line.startswith("new_w2="):
            left, right = line.split(" new_b2=", 1)
            values["new_w2"] = parse_value(left.split("=", 1)[1])
            values["new_b2"] = float(right)
        elif line.startswith("new_w1="):
            left, right = line.split(" new_b1=", 1)
            values["new_w1"] = parse_value(left.split("=", 1)[1])
            values["new_b1"] = parse_value(right)
    return values


def main():
    print(f"reference_tolerance={TOLERANCE}")
    for name, expected in EXPECTED.items():
        print(f"reference_{name}={expected}")
    if len(sys.argv) != 2:
        return
    actual = parse_mpc_output(sys.argv[1])
    missing = sorted(set(EXPECTED) - set(actual))
    if missing:
        raise AssertionError(f"missing MPC diagnostics: {missing}")
    for name, expected in EXPECTED.items():
        assert_close(name, actual[name], expected)
    if abs(actual["d_z1"][0]) > TOLERANCE:
        raise AssertionError("inactive ReLU unit has nonzero d_z1")
    if abs(actual["d_z1"][1] - 2.25) > TOLERANCE:
        raise AssertionError("active ReLU unit has unexpected d_z1")
    print("ReLU inactive-branch checks: PASS")
    print("MPC comparison: PASS")


if __name__ == "__main__":
    main()
