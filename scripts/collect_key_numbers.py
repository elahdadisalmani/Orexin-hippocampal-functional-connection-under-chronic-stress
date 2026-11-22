"""
Collects every number quoted in the Results/Discussion from the raw outputs into outputs/key_numbers.json
(one place to check the manuscript against).  Run after the figure scripts.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"
J = lambda n: json.load(open(OUT / n))
r4 = lambda x: None if x is None else round(float(x), 4)
K = {}

# 3.1 upstream
f1 = J("figure1_summary.json")
K["3.1_upstream"] = dict(values_by_day=f1["values"], day_R_below_Rc=f1["day_R_below_Rc"], I_O_range_days_20_45=f1["I_O_range_days_20_45"],
                          f_O_max=f1["f_O_max_Hz"], day_I_O_below_0p5=f1["I_O_below_0.5_day"], day_I_O_below_0p25=f1["I_O_below_0.25_day"],
                          critical_stop_day=f1["critical_stop_day"], critical_exposure_days=r4(f1["critical_stop_day"] - f1["parameters"]["t_on"]),
                          withdrawal=f1["recovery"])
try:
    K["3.1_upstream"]["sensitivity"] = J("upstream_sensitivity.json")
except FileNotFoundError:
    pass

# 3.2 network epochs
ep = J("ca1_epochs.json"); c = ep["Control (day 10)"]; d45 = ep["Compensation (day 45)"]
K["3.2_epochs"] = {k: {a: ([r4(x) for x in b] if isinstance(b, list) else r4(b)) for a, b in v.items()} for k, v in ep.items()}
K["3.2_epochs"]["derived"] = dict(theta_change_day45_pct=r4(100 * (d45["theta_abs"][0] / c["theta_abs"][0] - 1)),
                                  rateE_change_day45_pct=r4(100 * (d45["rate_E"][0] / c["rate_E"][0] - 1)))

# 3.3 BCM
K["3.3_bcm"] = J("bcm_summary.json")
raw = np.load(OUT / "ca1_epochs_raw.npz"); NE = 800
rate = lambda k: np.bincount(raw[k + "_id"][raw[k + "_id"] < NE], minlength=NE) / 10.0
f2c, f245 = np.mean(rate("control") ** 2), np.mean(rate("compensation") ** 2)
K["3.3_bcm"]["derived"] = dict(f2_ratio_day45=r4(f245 / f2c), f_target_sq_ratio_day45=r4(ep["Compensation (day 45)"]["I_O"] ** 2))

# 3.4 ablation
ab = J("ablation.json"); days = ab["days"]
K["3.4_ablation"] = dict(summary=J("ablation_summary.json"),
                         theta_by_day={k: dict(zip(map(str, days), [r4(x) for x in v["mean"]])) for k, v in ab["ablation"].items()})

# 3.5 transition
sw = J("ca1_sweep.json")
K["3.5_transition"] = dict(summary=J("ca1_transition_summary.json"),
                           sweep={k: dict(theta=r4(v["theta_abs"][0]), peak_freq=r4(v["peak_freq"][0]), rate_E=r4(v["rate_E"][0]),
                                          peak_ratio=r4(v["peak_ratio"][0])) for k, v in sw.items()})
m = np.load(OUT / "ca1_map.npz"); IO, A, M = np.array(m["IO"]), np.array(m["A"]), m["theta"]; ctrl = ep["Control (day 10)"]["theta_abs"][0]
cont = {}
for a, row in zip(A, M / ctrl):
    x = None
    for j in range(len(IO) - 1):
        if row[j] >= 0.5 > row[j + 1]:
            x = IO[j] + (0.5 - row[j]) * (IO[j + 1] - IO[j]) / (row[j + 1] - row[j]); break
    cont[f"A={a:.2f}"] = r4(x)
K["3.5_transition"]["contour_50pct_I_O_by_A"] = cont
em = J("ca1_eta_map.json"); T = np.array(em["theta"]); N = T / T[:, :1]; IOe = np.array(em["IO"]); e50 = {}
for e, row in zip(em["etas"], N):
    x = None
    for j in range(len(IOe) - 1):
        if row[j] >= 0.5 > row[j + 1]:
            x = IOe[j] + (0.5 - row[j]) * (IOe[j + 1] - IOe[j]) / (row[j + 1] - row[j]); break
    e50[f"eta_E={e}"] = r4(x)
K["3.5_transition"]["I_O_at_50pct_by_eta_E"] = e50
K["3.5_transition"]["eta_I"] = {k: v for k, v in J("ca1_etaI_sensitivity.json").items() if k.startswith("max_")}
json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1); print("key_numbers.json written")
