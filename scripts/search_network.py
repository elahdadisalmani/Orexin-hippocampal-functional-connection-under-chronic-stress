"""
Parameter search for the CA1 network.  Criteria were fixed BEFORE looking at any transition result:
 Stage A (control, I_O = 1, A = 1): LFP peak 4-8 Hz; pyramidal rate 0.5-8 Hz; interneuron rate 5-40 Hz;
          burst participation (99th pct of pyramidal fraction per 10 ms) <= 0.35; harmonic ratio PSD(2f)/PSD(f) <= 0.25;
          peak-to-background ratio >= 30.
 Stage B (orexin above baseline, eta_E = 0.5, eta_I = 0): pyramidal rate non-decreasing from I_O = 1 to 1.25 to 1.5 (>= 0.95 x previous);
          theta power at 1.5 >= 0.9 x theta power at 1.
 Selection: among candidates passing A and B, the smallest burst participation.
All evaluated candidates are written to outputs/network_search.json.
Usage: [RESUME=1] python scripts/search_network.py [N_candidates]   (default 344, about 9 minutes on one core; random seed 2026).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from ca1_network import load_net_params, run_network, analyse, sync_metrics
base = load_net_params(); rng = np.random.default_rng(2026)
N_CAND = int(sys.argv[1]) if len(sys.argv) > 1 else 344      # number of candidates; the seed makes the sequence reproducible
OUTF = ROOT / "outputs" / "network_search.json"
log = json.load(open(OUTF)) if os.environ.get("RESUME") and OUTF.exists() else []     # RESUME=1 continues a stopped search
for _ in log: rng.uniform(size=7)                                                   # advance the random stream past finished candidates
while len(log) < N_CAND:
    q = dict(w_EE=rng.uniform(0, 6), w_EI=rng.uniform(0, 5), w_IE=rng.uniform(0, 8), w_II=rng.uniform(0, 3),
             I_base_E=rng.uniform(3.5, 5.5), I_base_I=rng.uniform(1, 4), tau_gaba=rng.uniform(8, 40))
    p = dict(base, **q)
    r = run_network(p, 1.0, T_ms=4000, warm_ms=500)
    if not np.isfinite(r["lfp"]).all():               # numerical blow-up (runaway excitation): invalid candidate
        log.append(dict(params=q, passA=False, passB=None, invalid=True)); continue
    a = analyse(r, nperseg_s=1.0); part, harm = sync_metrics(r, a)
    rec = dict(params=q, peak=a["peak_freq"], rE=a["rate_E"], rI=a["rate_I"], part=part, harm=harm, ratio=a["peak_ratio"], theta=a["theta_abs"])
    rec["passA"] = bool(4 <= a["peak_freq"] <= 8 and 0.5 <= a["rate_E"] <= 8 and 5 <= a["rate_I"] <= 40 and part <= 0.35 and harm <= 0.25 and a["peak_ratio"] >= 30)
    rec["passB"] = None
    if rec["passA"]:
        rr = [run_network(p, io, T_ms=4000, warm_ms=500) for io in (1.25, 1.5)]
        if not all(np.isfinite(x["lfp"]).all() for x in rr):
            rec["passB"] = False; log.append(rec); continue
        out = [analyse(x, nperseg_s=1.0) for x in rr]
        r1, r2 = out[0]["rate_E"], out[1]["rate_E"]
        rec["up"] = dict(rE_125=r1, rE_150=r2, theta_150=out[1]["theta_abs"])
        rec["passB"] = bool(r1 >= 0.95 * a["rate_E"] and r2 >= 0.95 * r1 and out[1]["theta_abs"] >= 0.9 * a["theta_abs"])
    log.append(rec)
    json.dump(log, open(ROOT / "outputs" / "network_search.json", "w"))
nA = sum(r["passA"] for r in log); passed = [r for r in log if r["passB"]]
print(len(log), "evaluated;", nA, "pass A;", len(passed), "pass A+B")
for r in sorted(passed, key=lambda r: r["part"])[:5]:
    print(round(r["part"], 2), round(r["harm"], 3), "pk", r["peak"], "rE %.2f rI %.1f ratio %.0f" % (r["rE"], r["rI"], r["ratio"]), {k: round(v, 2) for k, v in r["params"].items()})
