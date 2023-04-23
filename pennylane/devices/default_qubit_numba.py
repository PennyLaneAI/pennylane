# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
The qubit.numba device is PennyLane's standard qubit-based device.

It implements the necessary :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qubit operations <pennylane.ops.qubit>`, it provides numba interface and provides a very
simple pure state
simulation of a numba-based quantum circuit architecture.
"""
from numba import jit
from . import DefaultQubit

class DefaultQubitNumba(DefaultQubit):
    """Default numba device for PennyLane.

        Args:
            wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
                or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
                or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
            shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
                the expectation values. Defaults to ``None`` if not specified, which means that the device
                returns analytical results.
    """

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
    def expval(self, observable, shot_range=None, bin_size=None):
        # Implement the method to compute expectation values of observables
        return super().expval(observable, shot_range=None, bin_size=None)

    @jit(nopython=True)
    def var(self, observable, shot_range=None, bin_size=None):
        # Implement the method to compute the variance of observables
        return super().var(observable, shot_range=None, bin_size=None)

    @jit(nopython=True)
    def sample(self, observable, shot_range=None, bin_size=None, counts=False):
        # Implement the method to draw samples from the device
        return super().sample(observable, shot_range=None, bin_size=None, counts=False)
