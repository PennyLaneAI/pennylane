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

It implements the necessary :class:`~pennylane.Device` methods as well as some built-in
qubit :doc:`operations </introduction/operations>`, providing a simple mixed-state simulation of
qubit-based quantum circuits.
"""

import functools
from string import ascii_letters as ABC

import numpy as np
from pennylane import QubitDevice
from pennylane.operation import DiagonalOperation, Channel

ABC_ARRAY = np.array(list(ABC))


class DefaultMixed(QubitDevice):
    """Default qubit device for performing mixed-state computations in PennyLane.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
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

    operations = {
        #     "BasisState",
        #     "QubitStateVector",
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

    def __init__(self, wires, *, shots=1000, analytic=True):
        if wires > 23:
            raise ValueError(
                "This device does not currently support computations on more than" "23 wires"
            )
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
        rho = np.zeros((2 ** self.num_wires, 2 ** self.num_wires), dtype=np.complex128)
        rho[index, index] = 1
        rho = self._asarray(rho, dtype=self.C_DTYPE)
        return self._reshape(rho, [2] * (2 * self.num_wires))

    @property
    def state(self):
        """Returns the state density matrix of the circuit prior to measurement"""
        dim = 2 ** self.num_wires
        # User obtains state as a matrix
        return self._reshape(self._pre_rotated_state, (dim, dim))

    def reset(self):
        """Resets the device"""
        super().reset()

        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):

        if self._state is None:
            return None

        # convert rho from tensor to matrix
        rho = self._reshape(self._state, (2 ** self.num_wires, 2 ** self.num_wires))
        # probs are diagonal elements
        probs = self.marginal_prob(self._diag(rho), wires)

        # take the real part so probabilities are not shown as complex numbers
        return self._real(probs)

    def _get_kraus(self, operation):  # pylint: disable=no-self-use
        """Return the Kraus operators representing the operation.

        Args:
            operation (.Operation): a PennyLane operation

        Returns:
            list[array[complex]]: Returns a list of 2D matrices representing the Kraus operators. If
            the operation is unitary, returns a single Kraus operator. In the case of a diagonal
            unitary, returns a 1D array representing the matrix diagonal.
        """
        if isinstance(operation, DiagonalOperation):
            return operation.eigvals

        if isinstance(operation, Channel):
            return operation.kraus_matrices

        return [operation.matrix]

    def _apply_channel(self, kraus, wires):
        r"""Apply a quantum channel specified by a list of Kraus operators to subsystems of the
        quantum state. For a unitary gate, there is a single Kraus operator.

        Args:
            kraus (list[array]): Kraus operators
            wires (Wires): target wires
        """

        channel_wires = self.map_wires(wires)
        rho_dim = 2 * self.num_wires
        num_ch_wires = len(channel_wires)

        # Computes K^\dagger, needed for the transformation K \rho K^\dagger
        kraus_dagger = [self._conj(self._transpose(k)) for k in kraus]

        # Changes tensor shape
        kraus_shape = [len(kraus)] + [2] * num_ch_wires * 2
        kraus = self._cast(self._reshape(kraus, kraus_shape), dtype=self.C_DTYPE)
        kraus_dagger = self._cast(self._reshape(kraus_dagger, kraus_shape), dtype=self.C_DTYPE)

        # Tensor indices of the state. For each qubit, need an index for rows *and* columns
        state_indices = ABC[:rho_dim]

        # row indices of the quantum state affected by this operation
        row_wires_list = channel_wires.tolist()
        row_indices = "".join(ABC_ARRAY[row_wires_list].tolist())

        # column indices are shifted by the number of wires
        col_wires_list = [w + self.num_wires for w in row_wires_list]
        col_indices = "".join(ABC_ARRAY[col_wires_list].tolist())

        # indices in einsum must be replaced with new ones
        new_row_indices = ABC[rho_dim : rho_dim + num_ch_wires]
        new_col_indices = ABC[rho_dim + num_ch_wires : rho_dim + 2 * num_ch_wires]

        # index for summation over Kraus operators
        kraus_index = ABC[rho_dim + 2 * num_ch_wires : rho_dim + 2 * num_ch_wires + 1]

        # new state indices replace row and column indices with new ones
        new_state_indices = functools.reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
            zip(col_indices + row_indices, new_col_indices + new_row_indices),
            state_indices,
        )

        # index mapping for einsum, e.g., 'iga,abcdef,idh->gbchef'
        einsum_indices = (
            "{kraus_index}{new_row_indices}{row_indices}, {state_indices},"
            "{kraus_index}{col_indices}{new_col_indices}->{new_state_indices}".format(
                kraus_index=kraus_index,
                new_col_indices=new_col_indices,
                col_indices=col_indices,
                state_indices=state_indices,
                row_indices=row_indices,
                new_row_indices=new_row_indices,
                new_state_indices=new_state_indices,
            )
        )

        self._state = self._einsum(einsum_indices, kraus, self._state, kraus_dagger)

    def _apply_diagonal_unitary(self, eigvals, wires):
        r"""Apply a diagonal unitary gate specified by a list of eigenvalues. This method uses
        the fact that the unitary is diagonal for a more efficient implementation.

        Args:
            eigvals (array): eigenvalues (phases) of the diagonal unitary
            wires (Wires): target wires
        """

        channel_wires = self.map_wires(wires)

        # reshape vectors
        eigvals = self._cast(self._reshape(eigvals, [2] * len(channel_wires)), dtype=self.C_DTYPE)

        # Tensor indices of the state. For each qubit, need an index for rows *and* columns
        state_indices = ABC[: 2 * self.num_wires]

        # row indices of the quantum state affected by this operation
        row_wires_list = channel_wires.tolist()
        row_indices = "".join(ABC_ARRAY[row_wires_list].tolist())

        # column indices are shifted by the number of wires
        col_wires_list = [w + self.num_wires for w in row_wires_list]
        col_indices = "".join(ABC_ARRAY[col_wires_list].tolist())

        einsum_indices = "{row_indices},{state_indices},{col_indices}->{state_indices}".format(
            col_indices=col_indices, state_indices=state_indices, row_indices=row_indices
        )

        self._state = self._einsum(einsum_indices, eigvals, self._state, self._conj(eigvals))

    def _apply_operation(self, operation):
        """Applies operations to the internal device state.

        Args:
            operation (.Operation): operation to apply on the device
        """
        wires = operation.wires
        matrices = self._get_kraus(operation)

        if isinstance(operation, DiagonalOperation):
            self._apply_diagonal_unitary(matrices, wires)
        else:
            self._apply_channel(matrices, wires)

    # pylint: disable=arguments-differ
    def apply(self, operations, rotations=None, **kwargs):

        rotations = rotations or []

        # apply the circuit operations
        for operation in operations:

            self._apply_operation(operation)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            self._apply_operation(operation)
