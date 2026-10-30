import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify" / "verify_bp.py"
SPEC = importlib.util.spec_from_file_location("tiny_bp_checker", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class TinyBPReferenceTests(unittest.TestCase):
    def test_synthetic_mpc_output_matches_reference(self):
        output = """\
z1=[1.5, 2.5]
h1=[1.5, 2.5]
y_hat=-0.75
loss=0.28125
d_w2=[-1.125, -1.875] d_b2=-0.75
d_w1=[[-0.75, -1.5], [0.75, 1.5]] d_b1=[-0.75, 0.75]
new_w2=[1.1125, -0.812485] new_b2=0.324997
new_w1=[[1.07501, 0.150009], [-0.0750122, 0.849991]] new_b1=[0.574997, 0.425003]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny_bp_mpc_output.txt"
            path.write_text(output, encoding="utf-8")
            actual = CHECKER.parse_mpc_output(path)
        for name, expected in CHECKER.EXPECTED.items():
            CHECKER.assert_close(name, actual[name], expected)


if __name__ == "__main__":
    unittest.main()
