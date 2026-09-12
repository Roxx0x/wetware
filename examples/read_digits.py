"""
The flagship: a real fruit-fly brain reading handwritten digits.

Downloads the connectome on first run (~1 MB), presents each 8x8 digit to the
fixed brain, trains a one-layer readout on its activity, and reports accuracy.
The wiring never changes — only the readout learns.

    pip install "wetware[all]"     # numpy + scipy + scikit-learn
    python examples/read_digits.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wetware import load
from wetware.tasks import digits

print("loading the fruit-fly connectome...")
W, ids = load()
print(f"brain: {W.shape[0]} neurons, {int((W != 0).sum())} connections\n")

print("presenting handwritten digits, training a readout on the brain's activity...")
r = digits.demo(W)

print(f"\n  a real fly brain read handwriting at {r['accuracy']:.0%}")
print(f"  (majority-class baseline: {r['baseline_accuracy']:.0%}, over {r['test_n']} test digits)")
print("\nThe connectome was never trained. Only a one-layer readout learned.")
