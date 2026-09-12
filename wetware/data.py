"""
Loading the brain.

`load()` gives you the wiring matrix W of the *complete larval Drosophila
connectome* — the first and only synapse-resolution wiring diagram of an entire
animal brain (Winding et al., Science 2023). 2952 neurons, ~110k weighted
directed connections. The first call downloads it (~1 MB) and caches a compact
`.npz`; after that it's instant.

`load_sample()` builds a small synthetic small-world network with the same shape
of statistics, clearly labelled as *not* the real brain. It exists so the tests
and a first look run offline with nothing downloaded. Anything you publish should
run on `load()` — the real thing is the whole point.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

_URL = "https://codeload.github.com/brain-networks/larval-drosophila-connectome/zip/refs/heads/main"
_INNER = "Supplementary-Data-S1/all-all_connectivity_matrix.csv"


def _cache_dir() -> Path:
    d = Path(os.environ.get("WETWARE_CACHE", Path.home() / ".cache" / "wetware"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(force: bool = False) -> Path:
    """Fetch the real connectome, parse it, cache a compact npz. Returns the path."""
    out = _cache_dir() / "larval_connectome.npz"
    if out.exists() and not force:
        return out
    print("downloading the larval Drosophila connectome (~1 MB)...")
    raw = urllib.request.urlopen(_URL, timeout=120).read()
    outer = zipfile.ZipFile(io.BytesIO(raw))
    inner_name = next(n for n in outer.namelist() if n.endswith("Supplementary-Data-S1.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    csv_name = next(n for n in inner.namelist() if n.endswith(_INNER))
    print("parsing the connectivity matrix...")
    with inner.open(csv_name) as f:
        text = io.TextIOWrapper(f, encoding="utf-8")
        header = text.readline().rstrip("\n").split(",")
        ids = np.array(header[1:], dtype=np.int64)
        n = len(ids)
        W = np.zeros((n, n), dtype=np.float32)
        for i, line in enumerate(text):
            parts = line.rstrip("\n").split(",")
            W[i] = np.array(parts[1:], dtype=np.float32)
    np.savez_compressed(out, W=W, ids=ids)
    print(f"cached {n} neurons -> {out}")
    return out


def load() -> tuple[np.ndarray, np.ndarray]:
    """Return (W, neuron_ids) for the real connectome. Downloads on first use."""
    path = download()
    d = np.load(path)
    return d["W"].astype(np.float64), d["ids"]


def load_sample(n: int = 600, density: float = 0.013, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic small-world stand-in — NOT the real brain. Offline/tests only.

    Matches the connectome's rough sparsity and heavy-tailed weights so the engine
    behaves similarly, but it is generated, not measured. Never present output from
    this as 'the fruit-fly brain'.
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n), dtype=np.float64)
    k = max(1, int(density * n))
    for i in range(n):
        # local ring + a few long-range shortcuts (Watts-Strogatz flavour)
        targets = [(i + j) % n for j in range(1, k + 1)]
        targets += list(rng.integers(0, n, size=k))
        for j in targets:
            if j != i:
                W[i, j] = rng.gamma(1.5, 2.0)   # heavy-tailed, like synapse counts
    ids = np.arange(n, dtype=np.int64)
    return W, ids
