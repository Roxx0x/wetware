"""
Task: read handwritten digits.

The flagship. Present each 8x8 handwritten digit to the brain one column at a
time, let the activity settle, and read the final state. Train a linear readout
to name the digit. The wiring never changes — a real animal's brain, used as the
feature extractor, and a one-layer readout that learns to see.

Needs scikit-learn for the digits dataset:  pip install scikit-learn
"""

from __future__ import annotations

import numpy as np

from ..readout import Readout
from ..reservoir import Reservoir


def _load_digits():
    try:
        from sklearn.datasets import load_digits
    except ImportError as e:
        raise SystemExit("this demo needs scikit-learn:  pip install scikit-learn") from e
    d = load_digits()
    X = d.images / 16.0          # (N, 8, 8) in [0,1]
    return X, d.target


def _features(res: Reservoir, images: np.ndarray) -> np.ndarray:
    """Present each image column-by-column; use the final reservoir state as its
    feature vector. State is reset between images."""
    feats = np.empty((len(images), res.n))
    Win = res._input_weights(images.shape[2])
    W = res.W
    for i, img in enumerate(images):
        x = np.zeros(res.n)
        for col in img.T:                       # 8 timesteps of 8 pixels
            x = (1 - res.leak) * x + res.leak * np.tanh(W @ x + Win @ col)
        feats[i] = x
    return feats


def demo(W: np.ndarray, *, train: int = 1200, spectral_radius: float = 1.1,
         leak: float = 0.5, ridge: float = 1.0, seed: int = 0) -> dict:
    images, labels = _load_digits()
    res = Reservoir(W, spectral_radius=spectral_radius, leak=leak, seed=seed)
    feats = _features(res, images)

    onehot = np.eye(10)[labels]
    ro = Readout(ridge=ridge).fit(feats[:train], onehot[:train])
    pred = ro.classify(feats[train:])
    true = labels[train:]
    acc = float(np.mean(pred == true))
    # baseline: majority class
    majority = np.bincount(labels[:train]).argmax()
    baseline = float(np.mean(true == majority))
    return {
        "task": "digits",
        "accuracy": acc,
        "baseline_accuracy": baseline,
        "test_n": int(len(true)),
        "neurons": int(W.shape[0]),
    }
