"""Check aggregate-only diagnostics from bounded secure BP."""
import re
import sys
from pathlib import Path


LINE = re.compile(r"^(validation|test) loss=([-+0-9.eE]+) correct=(\d+) total=(\d+)$")


def parse(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = LINE.match(line.strip())
        if match:
            result[match.group(1)] = {
                "loss": float(match.group(2)),
                "correct": int(match.group(3)),
                "total": int(match.group(4)),
            }
    return result


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_bp_training.py MPC_LOG")
    values = parse(sys.argv[1])
    if set(values) != {"validation", "test"}:
        raise AssertionError("missing aggregate validation/test diagnostics")
    for split, record in values.items():
        if record["total"] != 30 or record["loss"] < 0:
            raise AssertionError(f"invalid {split} aggregate: {record}")
        if not 0 <= record["correct"] <= record["total"]:
            raise AssertionError(f"invalid {split} count: {record}")
    print("aggregate diagnostics: PASS")


if __name__ == "__main__":
    main()
