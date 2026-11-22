"""Supplementary Fig. S2 from outputs/ca1_etaI_sensitivity.json"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
OUT = Path(__file__).resolve().parents[1] / "outputs"; res = json.load(open(OUT / "ca1_etaI_sensitivity.json"))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.5, 4.4))
for k, c, ls in (("0.0", "#1f4fd8", "-"), ("0.25", "#e08a00", "--"), ("0.5", "#2e8b57", ":")):
    ax.errorbar(res[k]["IO"], res[k]["mean"], res[k]["sd"], color=c, ls=ls, lw=2, marker="o", ms=4, capsize=2, label=rf"$\eta_I$ = {k}")
ax.invert_xaxis(); ax.set_xlabel(r"Orexin output $I_O$ (baseline = 1)"); ax.set_ylabel("Theta (4-8 Hz) power (mV²/Hz)")
ax.legend(frameon=False); ax.set_title(r"Sensitivity to the interneuron susceptibility $\eta_I$ ($\eta_E$ = 0.5)", fontsize=10, fontweight="bold", loc="left")
fig.savefig(OUT / "Figure_S2.png", dpi=300, bbox_inches="tight", facecolor="white")
Image.open(OUT / "Figure_S2.png").convert("RGB").save(OUT / "Figure_S2.tif", dpi=(300, 300), compression="tiff_lzw")
