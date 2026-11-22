"""
Figure 2: CA1 network dynamics at the four simulated epochs (A1-A3 rasters, B LFP spectra) and the BCM
modification function (C).  theta_M = theta0 <(f_i/f_target)^2>, f_target = f_t0 * I_O (see config).
Input : outputs/ca1_epochs_raw.npz, outputs/ca1_epochs.json   (from scripts/run_ca1_analyses.py)
Output: outputs/Figure_2.tif / .png
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
from ca1_network import load_net_params   # noqa: E402
raw = np.load(OUT / "ca1_epochs_raw.npz"); ep = json.load(open(OUT / "ca1_epochs.json"))
NE = 800
KEYS = {"Control (day 10)": "control", "Compensation (day 45)": "compensation", "Late (day 105)": "late", "Failure (day 120)": "failure"}
T_S = 10.0
rates = {n: np.bincount(raw[f"{k}_id"][raw[f"{k}_id"] < NE], minlength=NE) / T_S for n, k in KEYS.items()}
f2_ctrl = np.mean(rates["Control (day 10)"] ** 2)
THETA0 = float(np.sqrt(f2_ctrl))        # theta0 = f_t0 = rms control rate (BCM fixed point)
bcm = {}
for n in KEYS:
    r = rates[n]; IO = ep[n]["I_O"]
    thM = THETA0 * np.mean(r ** 2) / (f2_ctrl * IO ** 2)
    thM_alt = THETA0 * np.mean(r ** 2) * IO ** 2 / f2_ctrl      # alternative sign: f_target = f_t0 / I_O (orexin raises the threshold)
    bcm[n] = dict(I_O=round(IO, 3), mean_rate_E_Hz=round(float(r.mean()), 2), theta_M_Hz=round(float(thM), 2), theta_M_alt_sign_Hz=round(float(thM_alt), 2),
                  fraction_cells_above_theta_M=round(float(np.mean(r > thM)), 3),
                  mean_phi_Hz2=round(float(np.mean(r * (r - thM))), 2))
bcm["_theta0_Hz"] = round(THETA0, 3)
json.dump(bcm, open(OUT / "bcm_summary.json", "w"), indent=1)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = {"control": "#1f4fd8", "compensation": "#2e8b57", "late": "#e08a00", "failure": "#c0392b"}
LAB = {k: f"{k.split('(')[0].strip()}" for k in ep}

def raster(ax, key, title, t0=4000.0, win=1000.0):
    t, i = raw[f"{key}_t"], raw[f"{key}_id"]
    m = (t >= t0) & (t < t0 + win)
    e = m & (i < NE); n = m & (i >= NE)
    ax.plot(t[e] - t0, i[e], "|", color="#1f4fd8", ms=2.0, mew=0.5, rasterized=True)
    ax.plot(t[n] - t0, i[n], "|", color="#e08a00", ms=2.0, mew=0.5, rasterized=True)
    ax.axhline(NE, color="gray", lw=0.6, ls=":")
    ax.set_xlim(0, win); ax.set_ylim(0, 1000)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Neuron index (0-799 E, 800-999 I)")
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")

fig = plt.figure(figsize=(13, 8.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.38, wspace=0.32)
a1, a2, a3 = (fig.add_subplot(gs[0, k]) for k in range(3))
b = fig.add_subplot(gs[1, :2]); cc = fig.add_subplot(gs[1, 2])
for ax_, key, name, lab in [(a1, "control", "Control (day 10)", "A1"), (a2, "compensation", "Compensation (day 45)", "A2"),
                            (a3, "late", "Late (day 105)", "A3")]:
    s_ = ep[name]
    raster(ax_, key, f"{lab}  {name.split('(')[0].strip()}, day {s_['day']}\n$I_O$ = {s_['I_O']:.2f}, A = {s_['A']:.2f}")
    if lab != "A1": ax_.set_ylabel("")

for key, name in [("control", "Control (day 10)"), ("compensation", "Compensation (day 45)"),
                  ("late", "Late (day 105)"), ("failure", "Failure (day 120)")]:
    fr, ps = raw[f"{key}_f"], raw[f"{key}_psd"]
    s = ep[name]
    b.semilogy(fr, np.maximum(ps, 1e-5), color=COL[key], lw=2,
               label=f"{name}: $I_O$={s['I_O']:.2f}, peak {s['peak_freq'][0]:.1f} Hz, "
                     f"4-8 Hz power {s['theta_abs'][0]:.2f} ± {s['theta_abs'][1]:.2f}")
b.axvspan(4, 8, color="khaki", alpha=0.35, lw=0, label="Theta band (4-8 Hz)")
b.set_xlim(0, 30); b.set_ylim(1e-5, 1e4)   # silent network: PSD floored at 1e-5
b.set_xlabel("Frequency (Hz)"); b.set_ylabel("LFP power spectral density (mV²/Hz)")
b.set_title("B  LFP power spectrum (mean pyramidal membrane potential)", fontsize=10, fontweight="bold", loc="left")
b.legend(fontsize=8.5, frameon=False, loc="upper right")
fr_ = np.linspace(0, 6, 300)
for name in KEYS:
    k = name.split("(")[0].strip().lower(); th = bcm[name]["theta_M_Hz"]
    if k in ("failure", "late") and th == 0.0:      # silent network: theta_M is a numerical artefact (0/0), curve shown only for completeness
        cc.plot(fr_, fr_ * (fr_ - th), color=COL[k], lw=1.5, ls=":", label=f"{name.split('(')[0].strip()}: network silent, $\\theta_M$ undefined")
    else:
        cc.plot(fr_, fr_ * (fr_ - th), color=COL[k], lw=2, label=f"{name.split('(')[0].strip()}: $\\theta_M$ = {th:.1f} Hz")
cc.axhline(0, color="k", lw=0.8, ls="--"); cc.set_ylim(-4, 8); cc.set_xlim(0, 6)
cc.set_xlabel("Postsynaptic firing rate $f$ (Hz)"); cc.set_ylabel(r"$\phi(f,\theta_M) = f\,(f-\theta_M)$ (Hz²)")
cc.set_title("C  BCM modification function", fontsize=10, fontweight="bold", loc="left"); cc.legend(fontsize=7.5, frameon=False, loc="upper left")
png = OUT / "Figure_2.png"
fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
Image.open(png).convert("RGB").save(OUT / "Figure_2.tif", dpi=(300, 300), compression="tiff_lzw")
print("Figure 2 written")
