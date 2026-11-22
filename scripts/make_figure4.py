"""
Figure 4: transition of the CA1 rhythm as a function of orexin output.
  A  theta power and pyramidal rate vs I_O (A = 1, 5 seeds)
  B  hysteresis test: slow down- and up-ramp of I_O
  C  (I_O, A) map of theta power with the trajectory of the upstream model
Input : outputs/ca1_sweep.json, ca1_hysteresis.csv, ca1_map.npz, ca1_epochs.json, figure1_timeseries.csv
Output: outputs/Figure_4.tif / .png, outputs/ca1_transition_summary.json
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
from orexin_model import load_params   # noqa: E402
up = load_params()
sw = json.load(open(OUT / "ca1_sweep.json")); ep = json.load(open(OUT / "ca1_epochs.json"))
io = np.array(sorted((float(k) for k in sw), reverse=True))
th = np.array([sw[f"{x:.2f}"]["theta_abs"][0] for x in io]); ths = np.array([sw[f"{x:.2f}"]["theta_abs"][1] for x in io])
re_ = np.array([sw[f"{x:.2f}"]["rate_E"][0] for x in io]); pk = np.array([sw[f"{x:.2f}"]["peak_freq"][0] for x in io])
pr = np.array([sw[f"{x:.2f}"]["peak_ratio"][0] for x in io])

# transition characterisation (theta power normalised to control)
n = th / th[0]
def cross(level):          # I_O at which the normalised theta power first drops below `level`
    j = int(np.argmax(n < level)); x0, x1, y0, y1 = io[j-1], io[j], n[j-1], n[j]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
i60, i50, i20 = cross(0.6), cross(0.5), cross(0.2)
hy = np.genfromtxt(OUT / "ca1_hysteresis.csv", delimiter=",", names=True, dtype=None, encoding=None)
dn = hy["direction"] == "down"
lev = np.linspace(0.40, 0.75, 15)
d_i = np.interp(lev, hy["I_O"][dn][::-1], hy["theta_power"][dn][::-1]); u_i = np.interp(lev, hy["I_O"][~dn], hy["theta_power"][~dn])
hyst_gap = float(np.max(np.abs(d_i - u_i)) / th[0])

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(11, 8.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.32)
A, B, C = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
CP, CE = "#1f4fd8", "#7b4fa3"
A.errorbar(io, th, ths, color=CP, marker="o", ms=4, lw=2, capsize=2, label="Theta (4-8 Hz) power")
A.set_xlabel(r"Orexin output $I_O$ (baseline = 1)"); A.set_ylabel("Theta power (mV²/Hz)", color=CP); A.invert_xaxis()
A.tick_params(axis="y", labelcolor=CP); A.set_ylim(0, 1.45)
A2 = A.twinx(); A2.spines["right"].set_visible(True)
A2.plot(io, re_, color=CE, ls="--", lw=1.8, marker="s", ms=3, label="Pyramidal rate (right)")
A2.set_ylabel("Pyramidal firing rate (Hz)", color=CE); A2.tick_params(axis="y", labelcolor=CE); A2.set_ylim(0, 5.5)
for name, c_ in [("Compensation (day 45)", "#2e8b57"), ("Late (day 105)", "#e08a00"), ("Failure (day 120)", "#c0392b")]:
    A.axvline(ep[name]["I_O"], color=c_, ls=":", lw=1.4); A.text(ep[name]["I_O"], 1.42, name.split("(")[1][:-1], color=c_, rotation=90, ha="right", va="top", fontsize=8)
h1, l1 = A.get_legend_handles_labels(); h2, l2 = A2.get_legend_handles_labels(); A.legend(h1 + h2, l1 + l2, fontsize=8.5, frameon=False, loc="lower left")
A.set_title("A  Sweep of $I_O$ (A = 1)", fontsize=10, fontweight="bold", loc="left")

B.plot(hy["I_O"][dn], hy["theta_power"][dn], color=CP, lw=2, label="$I_O$ decreasing")
B.plot(hy["I_O"][~dn], hy["theta_power"][~dn], color="#e08a00", lw=2, ls="--", label="$I_O$ increasing")
B.set_xlim(0.80, 0.30); B.set_xlabel(r"Orexin output $I_O$"); B.set_ylabel("Theta power (mV²/Hz)")
B.legend(fontsize=8.5, frameon=False); B.set_title("B  Slow ramp (0.8 → 0.3 → 0.8, 60 s each way)", fontsize=10, fontweight="bold", loc="left")

m = np.load(OUT / "ca1_map.npz"); IOg, Ag, M = m["IO"], m["A"], m["theta"]
cf = C.contourf(IOg, Ag, M / th[0], levels=np.linspace(0, 1.4, 15), cmap="viridis", extend="max")
C.contour(IOg, Ag, M / th[0], levels=[0.5], colors="white", linewidths=1.6)
cb = fig.colorbar(cf, ax=C, pad=0.01); cb.set_label("Theta power / control")
ts = np.genfromtxt(OUT / "figure1_timeseries.csv", delimiter=",", names=True)
d = ts["day"]; sel = d >= up["t_on"]
C.plot(ts["I_O"][sel], np.exp(-up["sigma_A"] * (d[sel] - up["t_on"])), color="white", lw=2.4, ls="--", label="Trajectory day 20 → 120")
for name, c_ in [("Control (day 10)", "white"), ("Compensation (day 45)", "#2e8b57"), ("Late (day 105)", "#e08a00"), ("Failure (day 120)", "#c0392b")]:
    C.plot(ep[name]["I_O"], ep[name]["A"], "o", ms=8, mfc=c_, mec="k")
C.set_xlim(1.0, 0.05); C.set_ylim(0.3, 1.0)
C.set_xlabel(r"Orexin output $I_O$"); C.set_ylabel("Dendritic integrity A")
C.set_title("C  Theta power over the ($I_O$, A) plane; white contour = 50 % of control", fontsize=10, fontweight="bold", loc="left")
C.legend(fontsize=8.5, frameon=True, loc="lower left")
png = OUT / "Figure_4.png"; fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
Image.open(png).convert("RGB").save(OUT / "Figure_4.tif", dpi=(300, 300), compression="tiff_lzw")

summ = {"I_O_at_60pct_theta": round(i60, 3), "I_O_at_50pct_theta": round(i50, 3), "I_O_at_20pct_theta": round(i20, 3),
        "transition_width_I_O_60_to_20pct": round(i60 - i20, 3),
        "max_hysteresis_gap_fraction_of_control": round(hyst_gap, 3),
        "theta_power_control": round(float(th[0]), 3),
        "A_only_theta_power_at_A_0.3_over_control": round(float(M[-1, 0] / th[0]), 3)}
json.dump(summ, open(OUT / "ca1_transition_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
