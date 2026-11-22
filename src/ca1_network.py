"""
CA1 spiking microcircuit: N_E = 800 pyramidal (RS) + N_I = 200 fast-spiking (FS) Izhikevich neurons.

    dv/dt = 0.04 v^2 + 5 v + 140 - u + I_syn + I_ext          (RK4, dt = 0.1 ms)
    du/dt = a (b v - u);   if v >= 30: v <- c, u <- u + d
    I_syn,i = A * [ -gE_i (v_i - V_AMPA) - gI_i (v_i - V_GABA) ]          (Eq. 5)
    gE_i, gI_i : synaptic conductances; every presynaptic spike j adds W_ij and the sum decays
                 exponentially (tau_ampa, tau_gaba).  Weights are scaled by 1/sqrt(N).
    I_ext,i = I_base,i * [1 + eta_i * (I_O - 1)] + noise                   (Eq. 8; I_O = 1 at baseline)
    noise   : Gaussian, std = sigma_noise * I_base,i, redrawn every `noise_ms` ms.
Plasticity is OFF here (static W); BCM is treated in a separate module.
"""
import json
from pathlib import Path
import numpy as np
from numba import njit
from scipy.signal import welch

CONFIG = Path(__file__).resolve().parents[1] / "config" / "network_parameters.json"


def load_net_params(path=CONFIG):
    p = json.load(open(path))
    return {k: v for k, v in p.items() if not k.startswith("_")}


def build_network(p, seed):
    """Random directed graph (connection probability p_conn); returns pre-major weight matrix Wt[pre, post]."""
    rng = np.random.default_rng(seed)
    NE, NI = p["N_E"], p["N_I"]
    N = NE + NI
    conn = rng.random((N, N)) < p["p_conn"]          # conn[post, pre]
    np.fill_diagonal(conn, False)
    W = np.zeros((N, N))
    s = 1.0 / np.sqrt(N)
    W[:NE, :NE] = p["w_EE"] * s
    W[NE:, :NE] = p["w_EI"] * s                      # E -> I
    W[:NE, NE:] = p["w_IE"] * s                      # I -> E
    W[NE:, NE:] = p["w_II"] * s
    W *= conn
    return np.ascontiguousarray(W.T.astype(np.float64))   # Wt[pre, post]


@njit(cache=True)
def _run(Wt, NE, a, b, c, d, I_base, eta, IO_ms, AE, AI, sigma_noise, noise_steps,
         tau_ampa, tau_gaba, dt, n_steps, rec_every, seed):
    np.random.seed(seed)
    N = a.shape[0]
    v = -65.0 + 5.0 * np.random.randn(N)
    u = b * v
    gE = np.zeros(N); gI = np.zeros(N)
    dE = np.exp(-dt / tau_ampa); dI = np.exp(-dt / tau_gaba)
    I_ext = I_base * (1.0 + eta * (IO_ms[0] - 1.0))
    noise = np.zeros(N)
    n_rec = n_steps // rec_every
    lfp = np.zeros(n_rec)
    max_spk = 4_000_000
    sp_t = np.empty(max_spk, np.int32); sp_i = np.empty(max_spk, np.int32); ns = 0
    acc = 0.0
    for step in range(n_steps):
        if step % noise_steps == 0:
            k = min(step // noise_steps, IO_ms.shape[0] - 1)
            for i in range(N):
                I_ext[i] = I_base[i] * (1.0 + eta[i] * (IO_ms[k] - 1.0))
                noise[i] = sigma_noise * I_base[i] * np.random.randn()
        # RK4 with conductances frozen during the step
        k1v = np.empty(N); k1u = np.empty(N)
        for i in range(N):
            isyn = -AE * gE[i] * v[i] - AI * gI[i] * (v[i] + 70.0)
            k1v[i] = 0.04 * v[i] ** 2 + 5.0 * v[i] + 140.0 - u[i] + isyn + I_ext[i] + noise[i]
            k1u[i] = a[i] * (b[i] * v[i] - u[i])
        k2v = np.empty(N); k2u = np.empty(N)
        for i in range(N):
            vv = v[i] + 0.5 * dt * k1v[i]; uu = u[i] + 0.5 * dt * k1u[i]
            isyn = -AE * gE[i] * vv - AI * gI[i] * (vv + 70.0)
            k2v[i] = 0.04 * vv ** 2 + 5.0 * vv + 140.0 - uu + isyn + I_ext[i] + noise[i]
            k2u[i] = a[i] * (b[i] * vv - uu)
        k3v = np.empty(N); k3u = np.empty(N)
        for i in range(N):
            vv = v[i] + 0.5 * dt * k2v[i]; uu = u[i] + 0.5 * dt * k2u[i]
            isyn = -AE * gE[i] * vv - AI * gI[i] * (vv + 70.0)
            k3v[i] = 0.04 * vv ** 2 + 5.0 * vv + 140.0 - uu + isyn + I_ext[i] + noise[i]
            k3u[i] = a[i] * (b[i] * vv - uu)
        for i in range(N):
            vv = v[i] + dt * k3v[i]; uu = u[i] + dt * k3u[i]
            isyn = -AE * gE[i] * vv - AI * gI[i] * (vv + 70.0)
            k4v = 0.04 * vv ** 2 + 5.0 * vv + 140.0 - uu + isyn + I_ext[i] + noise[i]
            k4u = a[i] * (b[i] * vv - uu)
            v[i] += dt / 6.0 * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v)
            u[i] += dt / 6.0 * (k1u[i] + 2 * k2u[i] + 2 * k3u[i] + k4u)
            if v[i] < -100.0: v[i] = -100.0       # numerical safety
        # synaptic decay, then spikes
        for i in range(N):
            gE[i] *= dE; gI[i] *= dI
        for j in range(N):
            if v[j] >= 30.0:
                if ns < max_spk:
                    sp_t[ns] = step; sp_i[ns] = j; ns += 1
                v[j] = c[j]; u[j] += d[j]
                row = Wt[j]
                if j < NE:
                    for i in range(N): gE[i] += row[i]
                else:
                    for i in range(N): gI[i] += row[i]
        s = 0.0
        for i in range(NE): s += v[i]
        acc += s / NE
        if (step + 1) % rec_every == 0:
            lfp[(step + 1) // rec_every - 1] = acc / rec_every; acc = 0.0
    return lfp, sp_t[:ns], sp_i[:ns]


def run_network(p, I_O, A=1.0, A_inh=None, seed=1, T_ms=6000.0, warm_ms=1000.0, Wt=None, net_seed=1):
    """Simulate; returns dict(lfp (1 kHz), spikes (t_ms, id)).  Warm-up is discarded.
    I_O may be a number (constant) or an array with one value per ms of (warm-up + T_ms)."""
    NE, NI = p["N_E"], p["N_I"]; N = NE + NI
    dt = p["dt"]
    a = np.r_[np.full(NE, 0.02), np.full(NI, 0.1)]
    b = np.full(N, 0.2); c = np.full(N, -65.0)
    d = np.r_[np.full(NE, 8.0), np.full(NI, 2.0)]
    rng = np.random.default_rng(net_seed + 1000)
    het = 1.0 + p["heterogeneity"] * rng.uniform(-1, 1, N)
    I_base = np.r_[np.full(NE, p["I_base_E"]), np.full(NI, p["I_base_I"])] * het
    eta = np.r_[np.full(NE, p["eta_E"]), np.full(NI, p["eta_I"])]
    AE = AI = float(A)
    if p.get("A_target", "all") == "EE":          # atrophy removes excitatory input onto pyramidal dendrites only
        Wt = build_network(dict(p, w_EE=p["w_EE"] * float(A)), net_seed) if Wt is None else Wt
        AE = AI = 1.0
    if A_inh is not None:
        AI = float(A_inh)
    if Wt is None:
        Wt = build_network(p, net_seed)
    n_steps = int(round((T_ms + warm_ms) / dt)); rec_every = int(round(1.0 / dt))   # 1 kHz LFP
    n_ms = int(round(T_ms + warm_ms))
    IO_ms = np.full(n_ms, float(I_O)) if np.isscalar(I_O) else np.asarray(I_O, float)
    lfp, st, si = _run(Wt, NE, a, b, c, d, I_base, eta, IO_ms, AE, AI, p["sigma_noise"],
                       int(round(p["noise_ms"] / dt)), p["tau_ampa"], p["tau_gaba"], dt, n_steps, rec_every, seed)
    w0 = int(round(warm_ms))
    keep = st * dt >= warm_ms
    return dict(lfp=lfp[w0:], t_spk=st[keep] * dt - warm_ms, id_spk=si[keep], T_ms=T_ms, NE=NE, NI=NI)


def analyse(res, f_lo=4.0, f_hi=8.0, nperseg_s=2.0):
    """LFP spectrum (Welch) and spike statistics."""
    x = res["lfp"] - res["lfp"].mean()
    fs = 1000.0
    f, Pxx = welch(x, fs=fs, nperseg=int(nperseg_s * fs), noverlap=int(nperseg_s * fs / 2), detrend="linear")
    band = (f >= f_lo) & (f <= f_hi); tot = (f >= 1) & (f <= 45)
    sel = (f >= 2) & (f <= 15)
    fpk = float(f[sel][np.argmax(Pxx[sel])])
    ratio = float(Pxx[sel].max() / np.median(Pxx[tot]))     # peak-to-background ratio (band independent)
    NE, NI, T = res["NE"], res["NI"], res["T_ms"] / 1000.0
    is_e = res["id_spk"] < NE
    # Golomb synchrony chi on the pyramidal population (binned rate, 1 ms)
    nb = int(res["T_ms"])
    rate = np.bincount(res["t_spk"][is_e].astype(int), minlength=nb)[:nb] / NE
    return dict(f=f, psd=Pxx, peak_freq=fpk, peak_ratio=ratio, theta_rel=float(Pxx[band].sum() / Pxx[tot].sum()),
                theta_abs=float(Pxx[band].mean()), rate_E=float(is_e.sum() / NE / T),
                rate_I=float((~is_e).sum() / NI / T),
                cv_pop=float(rate.std() / (rate.mean() + 1e-12)))


def sync_metrics(res, a, bin_ms=10):
    """99th percentile of the fraction of pyramidal cells spiking in a 10-ms bin, and PSD(2f)/PSD(f) at the spectral peak."""
    NE = res["NE"]
    t = res["t_spk"][res["id_spk"] < NE]; nb = int(res["T_ms"] // bin_ms)
    c = np.bincount((t // bin_ms).astype(int), minlength=nb)[:nb] / NE
    f, P = a["f"], a["psd"]; k = int(np.argmax(np.where((f >= 2) & (f <= 15), P, 0))); h = int(np.argmin(abs(f - 2 * f[k])))
    return float(np.percentile(c, 99)), float(P[h] / max(P[k], 1e-12))
