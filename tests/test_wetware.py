"""Tests run on the offline synthetic sample — no download, no network."""

import numpy as np
import pytest

from wetware import Readout, Reservoir, load_sample
from wetware.tasks import timeseries


# --- connectome sample ----------------------------------------------------

def test_sample_shape_and_sparsity():
    W, ids = load_sample(n=400)
    assert W.shape == (400, 400)
    assert len(ids) == 400
    density = np.count_nonzero(W) / W.size
    assert 0.0 < density < 0.15   # sparse, like a real connectome


# --- reservoir ------------------------------------------------------------

def test_reservoir_is_stable_echo_state():
    W, _ = load_sample(n=300)
    res = Reservoir(W, spectral_radius=0.9, seed=0)
    states = res.run(np.random.default_rng(0).uniform(0, 0.5, (200, 1)))
    assert states.shape == (200, 300)
    assert np.all(np.abs(states) <= 1.0 + 1e-6)   # tanh-bounded, not blowing up
    assert np.isfinite(states).all()


def test_reservoir_washout_trims():
    W, _ = load_sample(n=200)
    res = Reservoir(W, seed=0)
    states = res.run(np.zeros((100, 1)), washout=20)
    assert states.shape[0] == 80


def test_spectral_radius_is_scaled():
    # after scaling, the reservoir's largest eigenvalue magnitude ~ target
    W, _ = load_sample(n=250)
    res = Reservoir(W, spectral_radius=0.8, seed=0)
    Wd = res.W.toarray() if hasattr(res.W, "toarray") else res.W
    radius = np.max(np.abs(np.linalg.eigvals(Wd)))
    assert abs(radius - 0.8) < 0.05


# --- readout --------------------------------------------------------------

def test_readout_fits_a_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    true_w = rng.normal(size=(10, 1))
    Y = X @ true_w
    ro = Readout(ridge=1e-6).fit(X, Y)
    assert np.corrcoef(ro.predict(X).ravel(), Y.ravel())[0, 1] > 0.99


def test_readout_untrained_raises():
    with pytest.raises(RuntimeError):
        Readout().predict(np.zeros((5, 3)))


# --- it actually learns (the whole point) ---------------------------------

def test_timeseries_beats_the_baseline_on_the_sample():
    W, _ = load_sample(n=500)
    r = timeseries.demo(W)
    # the brain-reservoir must predict the nonlinear series better than predict-the-mean
    assert r["nrmse"] < r["baseline_nrmse"]
