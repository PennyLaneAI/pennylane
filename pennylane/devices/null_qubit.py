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
The null.qubit device is a no-op device for benchmarking PennyLane's auxiliary functionality outside direct circuit evaluations.
"""
import itertools
import functools
from string import ascii_letters as ABC

import numpy as np
from scipy.sparse import coo_matrix

import pennylane as qml
from pennylane import QubitDevice, DeviceError, QubitStateVector, BasisState, Snapshot
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane.wires import WireError
from .._version import __version__

# pylint: disable=unused-argument
class NullQubit(DefaultQubit):
    """Null qubit device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
    """

    name = "Null qubit PennyLane plugin"
    short_name = "null.qubit"
    pennylane_requires = __version__
    version = __version__
    author = "Xanadu Inc."

    def __init__(self, wires, *args, **kwargs):
        pass

    # pylint: disable=arguments-differ
    def apply(self, operations, rotations=None, **kwargs):
        pass

    def _apply_operation(self, state, operation):
        pass

    def _apply_x(self, state, axes, **kwargs):
        pass

    def _apply_y(self, state, axes, **kwargs):
        pass

    def _apply_z(self, state, axes, **kwargs):
        pass

    def _apply_hadamard(self, state, axes, **kwargs):
        pass

    def _apply_s(self, state, axes, inverse=False):
        return self._apply_phase(state, axes, 1j, inverse)

    def _apply_t(self, state, axes, inverse=False):
        return self._apply_phase(state, axes, TPHASE, inverse)

    def _apply_sx(self, state, axes, inverse=False):
        pass

    def _apply_cnot(self, state, axes, **kwargs):
        pass

    def _apply_toffoli(self, state, axes, **kwargs):
        pass

    def _apply_swap(self, state, axes, **kwargs):
        pass

    def _apply_cz(self, state, axes, **kwargs):
        pass

    def _apply_phase(self, state, axes, parameters, inverse=False):
        pass

    def expval(self, observable, shot_range=None, bin_size=None):
        pass

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_reversible_diff=True,
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            returns_state=True,
        )
        return capabilities

    def _create_basis_state(self, index):
        pass

    @property
    def state(self):
        pass

    def density_matrix(self, wires):
        pass

    def _apply_state_vector(self, state, device_wires):
        pass

    def _apply_basis_state(self, state, wires):
        pass

    def _apply_unitary(self, state, mat, wires):
        pass

    def _apply_unitary_einsum(self, state, mat, wires):
        pass

    def _apply_diagonal_unitary(self, state, phases, wires):
        pass

    def reset(self):
        pass

    def analytic_probability(self, wires=None):
        pass