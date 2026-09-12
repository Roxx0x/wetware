"""
The readout: the only thing that learns.

Given the brain's activity (the reservoir states) and the answers you want, fit a
single linear layer by ridge regression — a closed-form least-squares solve, no
backprop, no epochs. This is the whole "training": the brain stays fixed, and a
matrix of weights on top learns to read its activity as a decision.

    Wout = (XᵀX + λI)⁻¹ XᵀY

That's it. If the reservoir is any good, a linear readout of it can do the task.
"""

from __future__ import annotations

import numpy as np


class Readout:
    def __init__(self, ridge: float = 1e-2) -> None:
        self.ridge = ridge
        self.W: np.ndarray | None = None

    def _design(self, states: np.ndarray) -> np.ndarray:
        # states (T, n) -> (T, n+1) with a bias column
        return np.hstack([states, np.ones((states.shape[0], 1))])

    def fit(self, states: np.ndarray, targets: np.ndarray) -> "Readout":
        X = self._design(states)
        Y = np.atleast_2d(targets)
        if Y.shape[0] != X.shape[0]:
            Y = Y.T
        n_feat = X.shape[1]
        self.W = np.linalg.solve(X.T @ X + self.ridge * np.eye(n_feat), X.T @ Y)
        return self

    def predict(self, states: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("readout is not trained")
        return self._design(states) @ self.W

    def classify(self, states: np.ndarray) -> np.ndarray:
        """For one-hot targets: the argmax class per row."""
        return np.argmax(self.predict(states), axis=1)
