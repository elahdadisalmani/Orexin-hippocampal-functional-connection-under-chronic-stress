"""
Runs all CA1-network simulations used by Figures 2 and 4 and stores the raw results in outputs/.
  1. epochs     : day 10 / 45 / 105 / 120 (I_O from the upstream model, A = exp(-sigma_A (day - t_on)))   x 5 seeds
  2. sweep      : I_O from 1 to 0.05 at A = 1                                                                x 5 seeds
  3. hysteresis : I_O ramped 0.45 -> 0.05 -> 0.45 (60 s each way)
  4. map        : theta power on an (I_O, A) grid (1 seed)
Run:  python scripts/make_figure1.py && python scripts/run_ca1_analyses.py      (about 6-8 min on one CPU core)
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.signal import spectrogram
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orexin_model import load_params          # noqa: E402
from ca1_network import load_net_params, run_network, analyse   # noqa: E402

OUT = ROOT / "outputs"
up, net = load_params(), load_net_params()
ts = np.genfromtxt(OUT / "figure1_timeseries.csv", delimiter=",", names=True)
I_of_day = lambda d: float(np.interp(d, ts["day"], ts["I_O"]))
A_of_day = lambda d: float(np.exp(-up["sigma_A"] * max(0.0, d - up["t_on"])))
import os
PARTS = os.environ.get('PARTS', 'epochs,sweep,hyst,map').split(',')
SEEDS = [1, 2, 3, 4, 5]
EPOCHS = {"Control (day 10)": 10, "Compensation (day 45)": 45, "Late (day 105)": 105, "Failure (day 120)": 120}

def stats(IO, A, seeds=SEEDS, T=6000):
    rows = [analyse(run_network(net, IO, A=A, seed=s, net_seed=s, T_ms=T)) for s in seeds]
    g = lambda k: [float(r[k]) for r in rows]
    return {k: [float(np.mean(g(k))), float(np.std(g(k)))] for k in
            ("theta_abs", "theta_rel", "peak_freq", "peak_ratio", "rate_E", "rate_I")}

# 1. epochs (raster / LFP / PSD of seed 1 are kept for the figure)
epochs, keep = {}, {}
for name, day in (EPOCHS.items() if 'epochs' in PARTS else []):
    IO, A = I_of_day(day), A_of_day(day)
    epochs[name] = dict(day=day, I_O=IO, A=A, **stats(IO, A, T=10000))
    r = run_network(net, IO, A=A, seed=1, net_seed=1, T_ms=10000)
    a = analyse(r)
    k = name.split("(")[0].strip().lower()
    keep[f"{k}_t"] = r["t_spk"].astype(np.float32); keep[f"{k}_id"] = r["id_spk"].astype(np.int16)
    keep[f"{k}_lfp"] = r["lfp"].astype(np.float32); keep[f"{k}_f"] = a["f"]; keep[f"{k}_psd"] = a["psd"]
    print(name, {k_: round(v[0], 3) for k_, v in epochs[name].items() if isinstance(v, list)}, flush=True)
if "epochs" in PARTS:
    np.savez_compressed(OUT / "ca1_epochs_raw.npz", **keep)
    json.dump(epochs, open(OUT / "ca1_epochs.json", "w"), indent=1)

# 2. I_O sweep (A = 1)
grid = [1.0, 0.9, 0.8, 0.7, 0.65, 0.62, 0.60, 0.58, 0.56, 0.54, 0.52, 0.50, 0.47, 0.45, 0.42, 0.40, 0.35, 0.30, 0.20, 0.05]
if "sweep" in PARTS:
    sweep = {f"{io:.2f}": stats(io, 1.0) for io in grid}
    json.dump(sweep, open(OUT / "ca1_sweep.json", "w"), indent=1); print("sweep done", flush=True)

# 3. hysteresis ramp
T, warm = 60000, 1000
down = np.linspace(0.80, 0.30, T // 2)
IO = np.r_[np.full(warm, 0.80), down, down[::-1]]
if 'hyst' in PARTS:
    r = run_network(net, IO, T_ms=T, warm_ms=warm, seed=1, net_seed=1)
    x = r["lfp"] - r["lfp"].mean()
    f, tt, S = spectrogram(x, fs=1000, nperseg=2000, noverlap=1000, detrend="linear")
    band = (f >= 4) & (f <= 8)
    io_t = np.interp(tt * 1000, np.arange(T), IO[warm:])
    with open(OUT / "ca1_hysteresis.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["time_s", "I_O", "theta_power", "direction"])
        for t_, io_, th_ in zip(tt, io_t, S[band].mean(0)):
            w.writerow([round(t_, 2), round(io_, 4), round(th_, 5), "down" if t_ < T / 2000 else "up"])
    print("hysteresis done", flush=True)

# 4. (I_O, A) map
IOg = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3, 0.2, 0.05]
Ag = [1.0, 0.85, 0.7, 0.55, 0.4, 0.3]
if 'map' in PARTS:
    M = np.array([[analyse(run_network(net, io, A=a, seed=1, net_seed=1, T_ms=5000))["theta_abs"] for io in IOg] for a in Ag])
    np.savez(OUT / "ca1_map.npz", IO=IOg, A=Ag, theta=M); print("map done", flush=True)
