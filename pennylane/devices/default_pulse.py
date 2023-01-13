# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
The default.pulse device builds on PennyLane's standard qubit-based device, adding support for
pulse-based circuits.
"""
import numpy as np

from .._version import __version__
from .default_qubit import DefaultQubit


class DefaultPulse(DefaultQubit):
    """Default pulses device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that the device
            returns analytical results.
        dt (float): the time step used by the differential equation solver to evolve the
             time-dependent Hamiltonian.
        dim (int): dimensions of the system (2 for qubits, 3 for qutrits, etc)
        drift(Operator): optional Hamiltonian term representing a constant drift term for the system Hamiltonian
    """

    name = "Default PennyLane plugin for pulses"
    short_name = "default.pulse"
    pennylane_requires = __version__
    version = __version__
    author = "Xanadu Inc."

    def __init__(self, wires, *, shots=None, analytic=None, dt=1e-5, dim=2, drift=None):
        r_dtype = np.float64
        c_dtype = np.complex128

        self.dt = dt
        self.dim = dim
        self.drift = drift

        super().__init__(wires, shots=shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
