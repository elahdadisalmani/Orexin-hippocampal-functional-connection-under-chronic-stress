"""
Ablation of the two pathways driven by chronic stress (answer to R1-M1) and network-level recovery scenarios.
Variants, evaluated on the day grid of the upstream model:
   full          : (I_O(day), A(day))        orexin pathway + dendritic atrophy
   orexin_only   : (I_O(day), 1)
   atrophy_only  : (1, A(day))               'classical' local pathway without orexin loss
Recovery: stress stopped at day 40 / 50 / 60 (A frozen at its value on the stop day) or continued.
Output: outputs/ablation.json   (theta power, 3 seeds for ablation, 1 seed for recovery)
Run after make_figure1.py:  python scripts/run_ablation.py     (~8 min)
"""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
from orexin_model import load_params
from ca1_network import load_net_params, run_network, analyse
up, net = load_params(), load_net_params()
ts = np.genfromtxt(OUT / "figure1_timeseries.csv", delimiter=",", names=True)
I_of = lambda d: float(np.interp(d, ts["day"], ts["I_O"]))
A_of = lambda d, off=None: float(np.exp(-up["sigma_A"] * max(0.0, min(d, off if off else d) - up["t_on"])))
th = lambda IO, A, s, T=6000, p=net: float(analyse(run_network(p, IO, A=A, seed=s, net_seed=s, T_ms=T))["theta_abs"])
net_all = dict(net, A_target="all")        # negative control: A scales every synapse
import os
days = list(range(10, 121, 10)); OUTF = OUT / "ablation.json"
out = json.load(open(OUTF)) if OUTF.exists() else {"days": days, "ablation": {}, "recovery": {}}
ONLY = os.environ.get("ONLY"); want = set(ONLY.split(",")) if ONLY else None      # e.g. ONLY=full,recovery (to resume a long run)
VARIANTS = {"full": lambda d: (I_of(d), A_of(d)), "orexin_only": lambda d: (I_of(d), 1.0), "atrophy_only": lambda d: (1.0, A_of(d)),
            "atrophy_only_all_synapses": lambda d: (1.0, A_of(d)), "full_all_synapses": lambda d: (I_of(d), A_of(d))}
for name, f in VARIANTS.items():
    if want and name not in want: continue
    pp = net_all if name.endswith('all_synapses') else net
    vals = np.array([[th(*f(d), s, p=pp) for s in (1, 2, 3)] for d in days])
    out["ablation"][name] = dict(mean=vals.mean(1).tolist(), sd=vals.std(1).tolist()); print(name, np.round(vals.mean(1), 2), flush=True)
    json.dump(out, open(OUTF, "w"), indent=1)
if not want or "recovery" in want:
    rec = {}
    for r in csv.DictReader(open(OUT / "figure1_recovery.csv")):
        rec.setdefault(r["scenario_stop_day"], []).append((float(r["day"]), float(r["I_O"])))
    rdays = list(range(10, 251, 10))
    for sc in ["40", "50", "60", "continuous"]:
        d_, i_ = map(np.array, zip(*rec[sc])); off = None if sc == "continuous" else float(sc)
        out["recovery"][sc] = dict(days=rdays, I_O=[float(np.interp(d, d_, i_)) for d in rdays],
                                   theta=[th(float(np.interp(d, d_, i_)), A_of(d, off), 1, 4000) for d in rdays]); print("recovery", sc, flush=True)
    json.dump(out, open(OUTF, "w"), indent=1)
