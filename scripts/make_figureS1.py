"""Supplementary Fig. S1 from outputs/ca1_eta_map.json"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from PIL import Image
OUT = Path(__file__).resolve().parents[1] / "outputs"; d = json.load(open(OUT / "ca1_eta_map.json"))
etas, IO, M = np.array(d["etas"]), np.array(d["IO"]), np.array(d["theta"]); N = M / M[:, :1]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8, 5.2))
cf = ax.contourf(IO, etas, N, levels=np.linspace(0, 1.2, 13), cmap="viridis", extend="max")
ax.contour(IO, etas, N, levels=[0.5], colors="white", linewidths=1.8)
for name, c in [("Compensation (day 45)", "#2e8b57"), ("Late (day 105)", "#e08a00"), ("Failure (day 120)", "#c0392b")]:
    ax.axvline(d["epochs"][name], color=c, ls="--", lw=1.6); ax.text(d["epochs"][name], etas.max() * 1.01, name.split("(")[1][:-1], color=c, rotation=90, va="top", ha="right", fontsize=8)
ax.set_xlim(1.0, 0.05); ax.set_xlabel(r"Orexin output $I_O$"); ax.set_ylabel(r"OX1R gain $\eta_E$  ($\eta_I$ = 0)")
fig.colorbar(cf, ax=ax).set_label("Theta power / control"); ax.set_title("Loss of the theta rhythm depends on the OX1R gain", fontsize=10, fontweight="bold", loc="left")
fig.savefig(OUT / "Figure_S1.png", dpi=300, bbox_inches="tight", facecolor="white"); Image.open(OUT / "Figure_S1.png").convert("RGB").save(OUT / "Figure_S1.tif", dpi=(300, 300), compression="tiff_lzw")
