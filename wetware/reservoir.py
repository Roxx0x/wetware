"""
The reservoir: the fly's brain, run as a fixed dynamical system.

The connectome gives a wiring matrix W — who connects to whom, weighted by real
synapse counts. We do not train it. It is the animal's brain; you don't get to
edit it. Instead we drive it with input and read the pattern of activity it
produces, and train only a thin readout on top (see readout.py). That split —
fixed biological recurrence, trained linear readout — is reservoir computing, and
using a connectome as the reservoir is the "Biological Processing Unit" idea from
the recent literature (see docs/the-science.md).

One knob matters for it to work at all: the spectral radius. Scale W so its
largest eigenvalue magnitude sits just below ~1 and the network has the echo-state
property — activity fades rather than blows up or dies. Everything else is the
brain as nature wired it.

The connectome is sparse (~1% of pairs connect), so if SciPy is installed the
recurrence runs as a sparse matrix-vector product — dozens of times faster than
dense. NumPy-only still works, just slower.
"""

from __future__ import annotations

import numpy as np

try:
    import scipy.sparse as _sp
    from scipy.sparse.linalg import eigs as _eigs
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


class Reservoir:
    def __init__(
        self,
        W: np.ndarray,
        *,
        spectral_radius: float = 0.95,
        leak: float = 0.3,
        input_scale: float = 1.0,
        inhibitory_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.n = W.shape[0]
        self.leak = leak
        rng = np.random.default_rng(seed)

        W = np.asarray(W, dtype=np.float64).copy()

        # Dale-ish signs: a fraction of neurons are inhibitory (negative out-weights).
        # The synapse-count matrix is unsigned; real circuits are E/I, and giving the
        # reservoir a sign structure makes its dynamics rich enough to compute with.
        signs = np.ones(self.n)
        signs[rng.permutation(self.n)[: int(inhibitory_fraction * self.n)]] = -1.0
        W *= signs[:, None]

        # scale to the target spectral radius (the echo-state condition)
        radius = self._spectral_radius(W)
        if radius > 0:
            W *= spectral_radius / radius

        # keep W sparse for the hot loop when SciPy is around
        self.W = _sp.csr_matrix(W) if _HAVE_SCIPY else W

        self._rng = rng
        self._input_scale = input_scale
        self._Win: np.ndarray | None = None

    @staticmethod
    def _spectral_radius(W: np.ndarray) -> float:
        if _HAVE_SCIPY and W.shape[0] > 400:
            try:
                return float(np.abs(_eigs(_sp.csr_matrix(W), k=1, which="LM",
                                          return_eigenvectors=False, maxiter=5000)[0]))
            except Exception:
                pass
        return float(np.max(np.abs(np.linalg.eigvals(W))))

    def _input_weights(self, in_dim: int) -> np.ndarray:
        if self._Win is None or self._Win.shape[1] != in_dim:
            self._Win = self._rng.uniform(-1, 1, (self.n, in_dim)) * self._input_scale
        return self._Win

    def run(self, inputs: np.ndarray, *, washout: int = 0) -> np.ndarray:
        """Drive the brain with a sequence of inputs; return the state at each step.

        inputs: (T, in_dim). Returns (T - washout, n) — the activity of every
        neuron over time, which is what the readout learns to decode.
        """
        inputs = np.atleast_2d(inputs)
        T, in_dim = inputs.shape
        Win = self._input_weights(in_dim)
        W = self.W
        x = np.zeros(self.n)
        states = np.empty((T, self.n))
        for t in range(T):
            pre = W @ x + Win @ inputs[t]
            x = (1 - self.leak) * x + self.leak * np.tanh(pre)
            states[t] = x
        return states[washout:]
