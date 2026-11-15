"""Shared input-stream primitives for secure BP experiment generators."""
import hashlib
from pathlib import Path


def write_values(streams, owner, values):
    streams[owner].write(" ".join(str(float(value)) for value in values) + "\n")


def input_hashes(output_dir):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(output_dir).glob("Input-P*-0"))
    }
