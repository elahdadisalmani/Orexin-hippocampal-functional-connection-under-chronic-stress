"""
Figure 3: which pathway drives the failure of the CA1 rhythm, and network-level consequences of stress withdrawal.
  A  theta power vs day for: full model, orexin pathway only, dendritic atrophy only (3 seeds, mean +/- sd)
  B  day on which theta power falls below 50 % of control, per variant
  C  theta power when stress is stopped at day 40 / 50 / 60 (upstream I_O from Fig. 1C)
Input: outputs/ablation.json, outputs/ca1_epochs.json     Output: outputs/Figure_3.tif/.png, outputs/ablation_summary.json
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
OUT = Path(__file__).resolve().parents[1] / "outputs"
d = json.load(open(OUT / "ablation.json")); days = np.array(d["days"]); ctrl = json.load(open(OUT / "ca1_epochs.json"))["Control (day 10)"]["theta_abs"][0]
V = {"full": ("Orexin loss + atrophy (full model)", "#c0392b", "-"), "orexin_only": ("Orexin loss only", "#1f4fd8", "--"),
     "atrophy_only": ("Atrophy only (classical local pathway)", "#2e8b57", "-.")}

def day50(x, y):
    """first day on which y < 0.5*control (linear interpolation); None if never"""
    n = np.asarray(y) / ctrl
    if not (n < 0.5).any(): return None
    j = int(np.argmax(n < 0.5)); return float(x[j-1] + (0.5 - n[j-1]) * (x[j] - x[j-1]) / (n[j] - n[j-1]))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(11, 8.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.3)
A, B, C = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
summ = {}
for k, (lab, col, ls) in V.items():
    m, s = np.array(d["ablation"][k]["mean"]), np.array(d["ablation"][k]["sd"])
    A.plot(days, m, color=col, ls=ls, lw=2.2, marker="o", ms=4, label=lab); A.fill_between(days, m - s, m + s, color=col, alpha=0.2, lw=0)
    summ[k] = {"day_theta_below_50pct": None if day50(days, m) is None else round(day50(days, m), 1),
               "theta_power_day_100": round(float(m[days == 100][0]), 3), "theta_power_day_110": round(float(m[days == 110][0]), 3),
               "theta_power_day_120": round(float(m[days == 120][0]), 3)}
A.axhline(ctrl, color="gray", lw=0.8, ls=":"); A.axvline(20, color="gray", ls=":", lw=1.1)
A.set_xlabel("Time (days)"); A.set_ylabel("Theta (4-8 Hz) power (mV²/Hz)"); A.set_xlim(10, 120); A.set_ylim(0, 1.7)
A.legend(fontsize=8.5, frameon=False, loc="lower left"); A.set_title("A  Ablation of the two pathways", fontsize=10, fontweight="bold", loc="left")
names = list(V); vals = [summ[k]["day_theta_below_50pct"] for k in names]
B.bar(range(3), [v if v else 0 for v in vals], color=[V[k][1] for k in names], width=0.6)
for i, v in enumerate(vals): B.text(i, (v if v else 0) + 2, f"day {v:.0f}" if v else "never\n(within 120 d)", ha="center", fontsize=9)
B.set_xticks(range(3)); B.set_xticklabels(["Full", "Orexin\nonly", "Atrophy\nonly"]); B.set_ylim(0, 135); B.set_ylabel("Day of 50 % theta loss")
B.set_title("B  Time to rhythm failure", fontsize=10, fontweight="bold", loc="left")
cols = {"continuous": ("k", "--", "Stress continues"), "60": ("#7fbf3f", "-", "Stress stops at day 60"), "50": ("#238a8d", "-", "Stress stops at day 50"), "40": ("#440154", "-", "Stress stops at day 40")}
for sc, (c_, ls, lab) in cols.items():
    r = d["recovery"][sc]; C.plot(r["days"], r["theta"], color=c_, ls=ls, lw=2.2, label=lab, marker="o", ms=3)
    summ[f"recovery_{sc}"] = {"theta_day_250": round(r["theta"][-1], 3), "day_theta_below_50pct": None if day50(np.array(r["days"], float), r["theta"]) is None else round(day50(np.array(r["days"], float), r["theta"]), 1)}
C.axhline(ctrl, color="gray", lw=0.8, ls=":"); C.set_xlim(10, 250); C.set_ylim(0, 1.7)
C.set_xlabel("Time (days)"); C.set_ylabel("Theta power (mV²/Hz)"); C.legend(fontsize=8.5, frameon=False, loc="lower left")
C.set_title("C  Network consequence of stress withdrawal (single seed)", fontsize=10, fontweight="bold", loc="left")
png = OUT / "Figure_3.png"; fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
Image.open(png).convert("RGB").save(OUT / "Figure_3.tif", dpi=(300, 300), compression="tiff_lzw")
json.dump(summ, open(OUT / "ablation_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
