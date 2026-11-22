"""
Upstream (lateral hypothalamus) orexin model.

State variables
    P : fraction of active orexin neurons (1 at baseline)
    f : mean firing rate of surviving neurons (Hz)
    R : metabolic / vesicular resource (1 = full)

Equations (time in days)
    dP/dt = -[ alpha*S(t) + eps*max(Rc - R, 0)/Rc ] * P          (Eq. 1)
    df/dt =  beta*R*max(1 - I_O, 0) - gamma*(f - f0)             (Eq. 2)
    dR/dt = (1 - R)/tau - kappa*max(f - f0, 0)*R                 (Eq. 3)
    I_O   =  P*f/f0       (normalised output, = 1 at baseline)   (Eq. 7)

S(t) = 1 between t_on and t_off (t_off = None: until the end), else 0.
S(t) enters ONLY Eq. 1.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

CONFIG = Path(__file__).resolve().parents[1] / "config" / "parameters.json"


def load_params(path=CONFIG):
    p = json.load(open(path))
    return {k: v for k, v in p.items() if not k.startswith("_")}


def _rhs(t, y, S, p):
    P, f, R = y
    I_O = P * f / p["f0"]
    loss = p["alpha"] * S + p["eps"] * max(p["Rc"] - R, 0.0) / p["Rc"]
    return [
        -loss * P,
        p["beta"] * R * max(1.0 - I_O, 0.0) - p["gamma"] * (f - p["f0"]),
        (1.0 - R) / p["tau"] - p["kappa"] * max(f - p["f0"], 0.0) * R,
    ]


def simulate(p, t_off=None, T=None, dt=0.25):
    """Integrate piecewise so that the discontinuities of S(t) are handled exactly.
    Returns dict of arrays: t, P, f, R, I_O."""
    T = p["T"] if T is None else T
    edges = [0.0, p["t_on"]] + ([t_off] if t_off is not None else []) + [T]
    y = [1.0, p["f0"], 1.0]
    ts, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        S = 1.0 if (a >= p["t_on"] and (t_off is None or a < t_off)) else 0.0
        sol = solve_ivp(_rhs, [a, b], y, args=(S, p), rtol=1e-9, atol=1e-12,
                        dense_output=True, max_step=0.2)
        tt = np.arange(a, b, dt)
        ts.append(tt)
        ys.append(sol.sol(tt))
        y = sol.y[:, -1]
    ts.append(np.array([T]))
    ys.append(np.array(y).reshape(3, 1))
    t = np.concatenate(ts)
    P, f, R = np.hstack(ys)
    return dict(t=t, P=P, f=f, R=R, I_O=P * f / p["f0"])


def first_crossing(t, x, thr, t_start, below=True):
    m = (t > t_start) & ((x < thr) if below else (x > thr))
    return float(t[np.argmax(m)]) if m.any() else None


def critical_stop_day(p, lo=None, hi=100.0, T=500.0, tol=0.01):
    """Earliest stress-stop day for which the system still collapses (P_final < 0.1)."""
    lo = p["t_on"] + 1.0 if lo is None else lo

    def collapses(off):
        return simulate(p, t_off=off, T=T, dt=1.0)["P"][-1] < 0.1

    if not collapses(hi):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if collapses(mid):
            hi = mid
        else:
            lo = mid
    return round(0.5 * (lo + hi), 1)
