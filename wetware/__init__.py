"""
wetware — compute on a real brain.

The complete larval Drosophila connectome (the first synapse-resolution wiring
diagram of an entire animal brain) run as a fixed reservoir. The wiring never
changes — it's the animal's brain — and a one-layer readout learns to read its
activity into a decision. Reservoir computing on real biological hardware.

    from wetware import load, Reservoir, Readout
    from wetware.tasks import digits

    W, ids = load()                 # the real fly brain, 2952 neurons
    print(digits.demo(W))           # train a readout to read handwriting
"""

from .data import download, load, load_sample
from .readout import Readout
from .reservoir import Reservoir

__version__ = "0.1.0"
__all__ = ["load", "load_sample", "download", "Reservoir", "Readout", "__version__"]
