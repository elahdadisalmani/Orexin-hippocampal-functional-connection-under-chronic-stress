"""Compares outputs/key_numbers.json with expected/key_numbers.json (reference values from the original run).
Usage: python scripts/check_reproduction.py [relative_tolerance]   -> exit code 0 if everything matches."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
tol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
new = json.load(open(ROOT / "outputs" / "key_numbers.json")); ref = json.load(open(ROOT / "expected" / "key_numbers.json"))
bad = []
def walk(a, b, path=""):
    if isinstance(b, dict):
        for k in b:
            if k not in a: bad.append((path + "/" + k, "missing", None)); continue
            walk(a[k], b[k], path + "/" + k)
    elif isinstance(b, list):
        if len(a) != len(b): bad.append((path, "length", (len(a), len(b)))); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]")
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        if a is None or abs(a - b) > tol * max(1.0, abs(b)): bad.append((path, a, b))
    elif a != b: bad.append((path, a, b))
walk(new, ref)
print("all key numbers reproduced" if not bad else f"{len(bad)} differences"); [print(" ", x) for x in bad[:25]]
sys.exit(1 if bad else 0)
