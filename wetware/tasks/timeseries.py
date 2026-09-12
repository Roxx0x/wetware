"""
Task: predict a nonlinear time-series.

Drive the brain with a random signal; ask the readout to output a value that
depends nonlinearly on several *past* inputs (a NARMA-style target). Getting it
right needs both memory and nonlinearity — exactly what a recurrent network of
neurons provides and a memoryless model can't fake. This is the classic reservoir
benchmark, run on a real connectome.
"""

from __future__ import annotations

import numpy as np

from ..readout import Readout
from ..reservoir import Reservoir


def make(T: int = 1500, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, (T, 1))
    y = np.zeros(T)
    for t in range(3, T):
        y[t] = 0.4 * y[t - 1] + 0.4 * u[t - 1, 0] * u[t - 2, 0] + 0.6 * u[t - 3, 0] ** 2 + 0.1
    return u, y.reshape(-1, 1)


def _nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (true.std() + 1e-9))


def demo(W: np.ndarray, *, T: int = 1500, washout: int = 100, train: int = 1000,
         spectral_radius: float = 0.95, leak: float = 0.3, ridge: float = 1e-2,
         seed: int = 0) -> dict:
    u, y = make(T)
    res = Reservoir(W, spectral_radius=spectral_radius, leak=leak, seed=seed)
    X = res.run(u, washout=washout)
    Y = y[washout:]
    ro = Readout(ridge=ridge).fit(X[:train], Y[:train])
    pred, true = ro.predict(X[train:]), Y[train:]
    baseline = np.full_like(true, Y[:train].mean())   # predict-the-mean
    return {
        "task": "timeseries",
        "nrmse": _nrmse(pred, true),
        "baseline_nrmse": _nrmse(baseline, true),
        "neurons": int(W.shape[0]),
    }
