"""
Supplementary Fig. S1: where does the loss of the theta rhythm occur as a function of the OX1R gain eta_E and of I_O?
eta_I = 0.  Theta power is normalised to the control (I_O = 1) value of the same eta (identical, since I_O = 1 -> I_ext = I_base).
Output: outputs/ca1_eta_map.json, outputs/Figure_S1.png/.tif
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs"; sys.path.insert(0, str(ROOT / "src"))
from ca1_network import load_net_params, run_network, analyse
net = load_net_params(); ep = json.load(open(OUT / "ca1_epochs.json"))
etas = [0.25, 0.375, 0.5, 0.75, 1.0, 1.5]; IOs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
import os
OUTF = OUT / "ca1_eta_map.json"; prev = json.load(open(OUTF)) if OUTF.exists() else {}
M = np.array(prev["theta"]) if prev and len(prev["theta"]) == len(etas) else np.zeros((len(etas), len(IOs)))
only = set(float(x) for x in os.environ["ETAS"].split(",")) if os.environ.get("ETAS") else None   # e.g. ETAS=0.25,0.375 to resume
for i, e in enumerate(etas):
    if only and e not in only: continue
    p = dict(net, eta_E=e, eta_I=0.0)
    for j, io in enumerate(IOs):
        M[i, j] = np.mean([analyse(run_network(p, io, seed=s, net_seed=s, T_ms=4000))["theta_abs"] for s in (1, 2)])
    print("eta", e, np.round(M[i] / M[i, 0], 2), flush=True)
    json.dump(dict(etas=etas, IO=IOs, theta=M.tolist(), epochs={k: v["I_O"] for k, v in ep.items()}), open(OUTF, "w"))
json.dump(dict(etas=etas, IO=IOs, theta=M.tolist(), epochs={k: v["I_O"] for k, v in ep.items()}), open(OUT / "ca1_eta_map.json", "w"))
