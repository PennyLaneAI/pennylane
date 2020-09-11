# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
The default.mixed device is PennyLane's standard qubit simulator for mixed-state computations.

It implements the necessary :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qubit operations <pennylane.ops.qubit>`, providing a simple mixed-state simulation of
qubit-based quantum circuits.
"""

import numpy as np

from pennylane import QubitDevice
from pennylane.operation import DiagonalOperation, Channel

# tolerance for numerical errors
tolerance = 1e-10


class DefaultMixed(QubitDevice):
    """Default qubit device for performing mixed-state computations in PennyLane.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not
             specified.
        shots (int): Number of times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to 1000 if not specified.
            If ``analytic == True``, the number of shots is ignored
            in the calculation of expectation values and variances, and only controls the number
            of samples returned by ``sample``.
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically.
    """

    name = "Default mixed-state qubit PennyLane plugin"
    short_name = "default.mixed"
    pennylane_requires = "0.12"
    version = "0.12.0"
    author = "Xanadu Inc."
    _capabilities = {"inverse_operations": True, "reversible_diff": True}  # check this

    operations = {
        "BasisState",
        "QubitStateVector",
        "QubitUnitary",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "Hadamard",
        "S",
        "T",
        "CNOT",
        "SWAP",
        "CSWAP",
        "Toffoli",
        "CZ",
        "PhaseShift",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "AmplitudeDamping",
        "GeneralizedAmplitudeDamping",
        "PhaseDamping",
        "DepolarizingChannel",
        "QubitChannel",
    }

    observables = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian", "Identity"}

    def __init__(self, wires, *, shots=1000, analytic=True):
        # call QubitDevice init
        super().__init__(wires, shots, analytic)

        # Create the initial state.
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def _create_basis_state(self, index):
        """Return the density matrix representing a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state.

        Returns:
            array[complex]: complex array of shape ``[2] * (2 * num_wires)``
            representing the density matrix of the basis state.
        """
        state = np.zeros(2 ** self.num_wires, dtype=np.complex128)
        state[index] = 1
        state = self._asarray(state, dtype=self.C_DTYPE)
        # density matrix is |state><state|
        rho = self._outer(state, state)
        return self._reshape(rho, [2] * (2 * self.num_wires))

    @property
    def state(self):
        """Allows users to retrieve state as a property"""
        # User obtains state as a matrix
        return self._reshape(self._pre_rotated_state, (2 ** self.num_wires, 2 ** self.num_wires))

    def reset(self):
        """Resets the device"""
        super().reset()

        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):
        """Computes the probability of observing each computational basis state"""

        if self._state is None:
            return None

        # convert rho from tensor to matrix
        rho = self._reshape(self._state, (2 ** self.num_wires, 2 ** self.num_wires))
        # probs are diagonal elements
        probs = self.marginal_prob(self._diag(rho), wires)
        return probs

    def _get_kraus(self, operation):  # pylint: disable=no-self-use
        """Return the Kraus operators representing the operation.

        Args:
            operation (~.Operation): a PennyLane operation

        Returns:
            array[complex]: Returns a list of 2D matrices representing the Kraus operators. If
            the operation is unitary, returns a single Kraus operator. In the case of a diagonal
             unitary, returns a 1D array representing the matrix diagonal.
        """
        if isinstance(operation, DiagonalOperation):
            return operation.eigvals

        if isinstance(operation, Channel):
            return operation.kraus_matrices

        return operation.matrix
