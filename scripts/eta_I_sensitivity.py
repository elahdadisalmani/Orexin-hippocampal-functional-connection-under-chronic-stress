"""
Supplementary Fig. S2: sensitivity of the CA1 rhythm to the interneuron orexin susceptibility eta_I
(eta_E = 0.5 fixed; A = 1; 3 seeds).  Output: outputs/ca1_etaI_sensitivity.json (figure: scripts/make_figureS2.py)
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT / "src"))
from ca1_network import load_net_params, run_network, analyse
net = load_net_params(); grid = [1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.56, 0.52, 0.48, 0.42, 0.3]
res = {}
for etaI in (0.0, 0.25, 0.5):
    p = dict(net, eta_I=etaI); m, s = [], []
    for io in grid:
        v = [analyse(run_network(p, io, seed=k, net_seed=k, T_ms=5000))["theta_abs"] for k in (1, 2, 3)]
        m.append(float(np.mean(v))); s.append(float(np.std(v)))
    res[str(etaI)] = dict(IO=grid, mean=m, sd=s); print("eta_I", etaI, np.round(m, 2), flush=True)
base = np.array(res["0.0"]["mean"])
res["max_abs_difference_mV2_per_Hz"] = {k: round(float(np.max(np.abs(np.array(res[k]["mean"]) - base))), 3) for k in ("0.25", "0.5")}
res["max_relative_difference_vs_control_power"] = {k: round(float(np.max(np.abs(np.array(res[k]["mean"]) - base)) / base[0]), 3) for k in ("0.25", "0.5")}
json.dump(res, open(OUT / "ca1_etaI_sensitivity.json", "w"), indent=1)
print(res['max_abs_difference_mV2_per_Hz'], res['max_relative_difference_vs_control_power'])
