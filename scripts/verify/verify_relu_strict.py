"""Verify the tracked strict ReLU boundary diagnostic."""
import ast
import re
import sys
from pathlib import Path


LINE = re.compile(r"^(pre|relu|relu_derivative)=(.*)$")
EXPECTED = {
    "pre": [-1.0, 0.0, 1.0],
    "relu": [0.0, 0.0, 1.0],
    "relu_derivative": [0.0, 0.0, 1.0],
}


def parse(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = LINE.match(line.strip())
        if match:
            values[match.group(1)] = ast.literal_eval(match.group(2))
    return values


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_relu_strict.py MPC_LOG")
    values = parse(sys.argv[1])
    if values != EXPECTED:
        raise AssertionError(f"strict ReLU mismatch: {values}")
    print("strict ReLU boundary: PASS")


if __name__ == "__main__":
    main()
