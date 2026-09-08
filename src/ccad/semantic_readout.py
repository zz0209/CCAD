"""NumPy application of saved ordered attribute projectors.

The input is a physical decoded donor-minus-recipient difference. This does
not encode text, fit a semantic alignment, or run the downstream model.
"""
from __future__ import annotations
import numpy as np


def apply_numpy(difference, bases, control, reverse=False):
    difference = np.asarray(difference)
    bases = np.asarray(bases)
    control = np.asarray(control)
    if difference.ndim != 2 or bases.ndim != 3 or difference.shape[1] != bases.shape[2]:
        raise ValueError('Input/basis dimensions differ')
    if control.shape != (len(bases),) or not np.isfinite(control).all() or np.any((control < 0) | (control > 1)):
        raise ValueError('Expected one finite [0,1] strength per attribute')
    if not np.isfinite(difference).all() or not np.isfinite(bases).all():
        raise ValueError('Nonfinite input')
    delta = np.zeros_like(difference, dtype=np.result_type(difference, bases, control))
    for a in (reversed(range(len(bases))) if reverse else range(len(bases))):
        if control[a] != 0:
            weight = bases[a]
            delta = delta + control[a] * ((difference-delta) @ weight.T) @ weight
    return delta
