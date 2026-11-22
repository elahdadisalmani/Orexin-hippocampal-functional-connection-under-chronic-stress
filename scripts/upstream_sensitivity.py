"""
Sensitivity of the upstream orexin model (Supplementary Table S1) and the epsilon = 0 control.
Output: outputs/upstream_sensitivity.json
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from orexin_model import load_params, simulate, first_crossing, critical_stop_day
base = load_params()

def summarise(p):
    s = simulate(p, dt=0.25); t = s["t"]; g = lambda d, a: float(a[np.argmin(abs(t - d))])
    return dict(day_R_below_Rc=first_crossing(t, s["R"], p["Rc"], p["t_on"]), I_O_45=round(g(45, s["I_O"]), 3),
                I_O_105=round(g(105, s["I_O"]), 3), I_O_120=round(g(120, s["I_O"]), 3), critical_stop_day=critical_stop_day(p))

out = {"reference": summarise(base), "variations": {}}
for k, f in [("eps", 0.6), ("eps", 1.4), ("kappa", 0.8), ("kappa", 1.2), ("beta", 0.75), ("beta", 1.25), ("alpha", 0.8), ("alpha", 1.2)]:
    p = dict(base); p[k] = base[k] * f
    out["variations"][f"{k}x{f}"] = summarise(p); print(k, f, out["variations"][f"{k}x{f}"], flush=True)
out["eps0"] = summarise(dict(base, eps=0.0))      # critical_stop_day is None when no delayed failure occurs up to day 100
json.dump(out, open(ROOT / "outputs" / "upstream_sensitivity.json", "w"), indent=1)
print("reference", out["reference"]); print("eps0", out["eps0"])
