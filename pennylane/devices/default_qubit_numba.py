from . import DefaultQubit
import numpy as np
from numba import jit


class DefaultQubitNumba(DefaultQubit):
    short_name = "numba.device"
    pennylane_requires = ">=0.13"
    version = "0.1.0"
    author = "Your Name"
    _capabilities = {
        "model": "qubit",
        "supports_reversible_diff": False,
    }

    def __init__(self, wires, shots=1000, **kwargs):
        super().__init__(wires=wires, shots=shots, **kwargs)

    @jit(nopython=True)
    def apply(self, operations, rotations=None, **kwargs):
        # Implement the method to apply quantum operations to the device
        super().apply(operations, rotations=None, **kwargs)

    @jit(nopython=True)
    def expval(self, observable):
        # Implement the method to compute expectation values of observables
        return super().expval(observable, shot_range=None, bin_size=None)

    @jit(nopython=True)
    def var(self, observable):
        # Implement the method to compute the variance of observables
        return super().var(observable, shot_range=None, bin_size=None)

    @jit(nopython=True)
    def sample(self, observable):
        # Implement the method to draw samples from the device
        return super().sample(observable, shot_range=None, bin_size=None, counts=False)
