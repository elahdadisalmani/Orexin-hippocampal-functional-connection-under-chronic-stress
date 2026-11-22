"""
Figure 1 - upstream orexin trajectory under chronic stress.
Everything plotted and every number reported comes from src/orexin_model.py.

Panels:  A  active population P and resource R
         B  firing rate f_O (Hz) and normalised output I_O
         C  recovery scenarios: stress stopped at different days
Outputs (outputs/): Figure_1.tif (RGB, 300 dpi), Figure_1.png, figure1_timeseries.csv,
                    figure1_recovery.csv, figure1_summary.json
Run:  python scripts/make_figure1.py
"""
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orexin_model import load_params, simulate, first_crossing, critical_stop_day  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
p = load_params()
STOP_DAYS = [30, 40, 50, 60, 80]          # stress-withdrawal scenarios (panel C)
T_REC = 250.0

# ---------------------------------------------------------------- simulations
base = simulate(p)
t, P, f, R, I = (base[k] for k in ("t", "P", "f", "R", "I_O"))
day_R = first_crossing(t, R, p["Rc"], p["t_on"])
rec = {d: simulate(p, t_off=float(d), T=T_REC, dt=0.5) for d in STOP_DAYS}
rec_cont = simulate(p, T=T_REC, dt=0.5)
d_crit = critical_stop_day(p)

at = lambda d, a: float(a[np.argmin(abs(t - d))])
m = (t >= p["t_on"]) & (t <= 45)
summary = {
    "parameters": p,
    "day_R_below_Rc": day_R,
    "I_O_range_days_20_45": [round(float(I[m].min()), 3), round(float(I[m].max()), 3)],
    "f_O_max_Hz": {"value": round(float(f.max()), 2), "day": float(t[f.argmax()])},
    "I_O_below_0.5_day": first_crossing(t, I, 0.5, p["t_on"]),
    "I_O_below_0.25_day": first_crossing(t, I, 0.25, p["t_on"]),
    "critical_stop_day": d_crit,
    "values": {str(d): {"P": round(at(d, P), 3), "f_Hz": round(at(d, f), 2),
                        "R": round(at(d, R), 3), "I_O": round(at(d, I), 3)}
               for d in (10, 20, 30, 45, 55, 70, 105, 120)},
    "recovery": {},
}
for d, s in rec.items():
    i = int(np.argmin(abs(s["t"] - d)))
    late = (s["t"] > d) & (s["I_O"] < 0.5)
    summary["recovery"][str(d)] = {
        "I_O_at_stop": round(float(s["I_O"][i]), 3), "R_at_stop": round(float(s["R"][i]), 3),
        "R_min_after_stop": round(float(s["R"][s["t"] >= d].min()), 3),
        "I_O_at_end": round(float(s["I_O"][-1]), 3),
        "days_to_I_O_below_0.5": round(float(s["t"][np.argmax(late)] - d), 1) if late.any() else None}
json.dump(summary, open(OUT / "figure1_summary.json", "w"), indent=1)

with open(OUT / "figure1_timeseries.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["day", "P_active", "f_O_Hz", "R", "I_O"])
    w.writerows(zip(t, P, f, R, I))
with open(OUT / "figure1_recovery.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["scenario_stop_day", "day", "P_active", "f_O_Hz", "R", "I_O"])
    for d, s in list(rec.items()) + [("continuous", rec_cont)]:
        w.writerows([(d, a, b, c, e, g) for a, b, c, e, g in zip(s["t"], s["P"], s["f"], s["R"], s["I_O"])])

# ------------------------------------------------------------------- plotting
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C_P, C_R, C_F, C_I = "#c0392b", "#2e8b57", "#7b4fa3", "#1f4fd8"
fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.32)
axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

def phases(ax, ymax):
    ax.axvspan(0, p["t_on"], color="#dfe9f7", alpha=0.6, lw=0)
    ax.axvspan(p["t_on"], day_R, color="#e3f3e8", alpha=0.6, lw=0)
    ax.axvspan(day_R, p["T"], color="#fbe4e1", alpha=0.6, lw=0)
    ax.axvline(p["t_on"], color="gray", ls=":", lw=1.2)
    ax.axvline(day_R, color="gray", ls="-.", lw=1.2)
    for x, lab in [(p["t_on"] / 2, "Control"), ((p["t_on"] + day_R) / 2, "Compensation"),
                   ((day_R + p["T"]) / 2, "Failure")]:
        ax.text(x, ymax * 0.985, lab, ha="center", va="top", fontsize=9, fontweight="bold", color="#444")

# A
phases(axA, 118)
axA.plot(t, 100 * P, color=C_P, lw=2.2, label=r"Active neurons $P_{active}$")
axA.plot(t, 100 * R, color=C_R, lw=2.2, ls="--", label=r"Resources $R$")
axA.axhline(100 * p["Rc"], color=C_R, lw=0.9, ls=":")
axA.text(p["T"] - 1, 100 * p["Rc"] + 2, r"$R_c$", color=C_R, ha="right", fontsize=9)
axA.set_xlim(0, p["T"]); axA.set_ylim(0, 118)
axA.set_xlabel("Time (days)"); axA.set_ylabel("Percentage of baseline (%)")
axA.legend(loc="lower left", fontsize=8.5, frameon=False)

# B
phases(axB, 6.1)
axB.plot(t, f, color=C_F, lw=2.2, label=r"Firing rate $f_O$ (left)")
axB.axhline(p["f0"], color=C_F, lw=0.9, ls=":")
axB.set_xlim(0, p["T"]); axB.set_ylim(0, 6.1)
axB.set_xlabel("Time (days)"); axB.set_ylabel(r"$f_O$ (Hz)", color=C_F)
axB.tick_params(axis="y", labelcolor=C_F)
axB2 = axB.twinx(); axB2.spines["right"].set_visible(True)
axB2.plot(t, I, color=C_I, lw=2.4, label=r"Normalised output $I_O$ (right)")
axB2.set_ylim(0, 1.22); axB2.set_ylabel(r"$I_O$ (baseline = 1)", color=C_I)
axB2.tick_params(axis="y", labelcolor=C_I)
h1, l1 = axB.get_legend_handles_labels(); h2, l2 = axB2.get_legend_handles_labels()
axB.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.5, frameon=False, bbox_to_anchor=(0.0, 0.08))

# C
cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(STOP_DAYS)))
axC.axvspan(p["t_on"], 0, color="none")
axC.plot(rec_cont["t"], rec_cont["I_O"], color="k", lw=1.6, ls="--", label="Stress continues")
for c, d in zip(cmap, STOP_DAYS):
    s = rec[d]
    axC.plot(s["t"], s["I_O"], color=c, lw=2, label=f"Stress stops at day {d}")
    axC.plot([d], [s["I_O"][int(np.argmin(abs(s["t"] - d)))]], "o", color=c, ms=5)
axC.axhline(1, color="gray", lw=0.8, ls=":")
axC.axvline(p["t_on"], color="gray", ls=":", lw=1.2)
axC.set_xlim(0, T_REC); axC.set_ylim(-0.02, 1.12)
axC.set_xlabel("Time (days)"); axC.set_ylabel(r"$I_O$ (baseline = 1)")
axC.axvline(d_crit, color="crimson", lw=1.1, ls=(0, (4, 3)))
axC.text(d_crit + 3, 1.035, f"critical stop day ≈ {d_crit:.0f}", color="crimson", fontsize=9, va="bottom")
axC.text(0.02, 0.36, "Stopped before the critical day:\npartial recovery, stable output.\n"
                    "Stopped after it:\ndelayed, self-sustained failure.",
         transform=axC.transAxes, ha="left", va="top", fontsize=9,
         bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
axC.legend(loc="center right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 0.45))

for ax, lab in zip((axA, axB, axC), "ABC"):
    ax.text(-0.09 if ax is not axC else -0.045, 1.05, lab, transform=ax.transAxes,
            fontsize=14, fontweight="bold")

png = OUT / "Figure_1.png"
fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
Image.open(png).convert("RGB").save(OUT / "Figure_1.tif", dpi=(300, 300), compression="tiff_lzw")
print(json.dumps({k: summary[k] for k in ("day_R_below_Rc", "I_O_range_days_20_45", "f_O_max_Hz",
                                          "I_O_below_0.5_day", "I_O_below_0.25_day", "critical_stop_day")}, indent=1))
