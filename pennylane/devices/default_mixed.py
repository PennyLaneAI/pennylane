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
The default.mixed device is PennyLane's standard qubit simulator for mixed-state computations.

It implements the necessary :class:`~pennylane.Device` methods as well as some built-in
qubit :doc:`operations </introduction/operations>`, providing a simple mixed-state simulation of
qubit-based quantum circuits.
"""

import functools
import itertools
from string import ascii_letters as ABC

import pennylane.numpy as np
import pennylane.math as qnp
from pennylane import QubitDevice, QubitStateVector, BasisState, DeviceError, QubitDensityMatrix
from pennylane.operation import Channel
from pennylane.wires import Wires
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from .._version import __version__

ABC_ARRAY = np.array(list(ABC))
tolerance = 1e-10


class DefaultMixed(QubitDevice):
    """Default qubit device for performing mixed-state computations in PennyLane.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        shots (None, int): Number of times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that
            outputs are computed exactly.
        cache (int): Number of device executions to store in a cache to speed up subsequent
            executions. A value of ``0`` indicates that no caching will take place. Once filled,
            older elements of the cache are removed and replaced with the most recent device
            executions to keep the cache up to date.
    """

    name = "Default mixed-state qubit PennyLane plugin"
    short_name = "default.mixed"
    pennylane_requires = __version__
    version = __version__
    author = "Xanadu Inc."

    operations = {
        "Identity",
        "BasisState",
        "QubitStateVector",
        "QubitDensityMatrix",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "Hadamard",
        "S",
        "T",
        "SX",
        "CNOT",
        "SWAP",
        "CSWAP",
        "Toffoli",
        "CZ",
        "PhaseShift",
        "ControlledPhaseShift",
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
        "BitFlip",
        "PhaseFlip",
        "PauliError",
        "ResetError",
        "QubitChannel",
        "SingleExcitation",
        "SingleExcitationPlus",
        "SingleExcitationMinus",
        "DoubleExcitation",
        "DoubleExcitationPlus",
        "DoubleExcitationMinus",
        "QubitCarry",
        "QubitSum",
        "OrbitalRotation",
        "QFT",
        "ThermalRelaxationError",
    }

    def __init__(self, wires, *, shots=None, cache=0, analytic=None):
        if isinstance(wires, int) and wires > 23:
            raise ValueError(
                "This device does not currently support computations on more than 23 wires"
            )

        # call QubitDevice init
        super().__init__(wires, shots, cache=cache, analytic=analytic)

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
        rho = np.zeros((2**self.num_wires, 2**self.num_wires), dtype=np.complex128)
        rho[index, index] = 1
        rho = self._asarray(rho, dtype=self.C_DTYPE)
        return self._reshape(rho, [2] * (2 * self.num_wires))

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            returns_state=True,
        )
        return capabilities

    @property
    def state(self):
        """Returns the state density matrix of the circuit prior to measurement"""
        dim = 2**self.num_wires
        # User obtains state as a matrix
        return self._reshape(self._pre_rotated_state, (dim, dim))

    def density_matrix(self, wires):
        """Returns the reduced density matrix over the given wires.

        Args:
            wires (Wires): wires of the reduced system

        Returns:
            array[complex]: complex array of shape ``(2 ** len(wires), 2 ** len(wires))``
            representing the reduced density matrix of the state prior to measurement.
        """
        # Return the full density matrix if all the wires are given
        if wires == self.wires:
            return self.state

        traced_wires = [x for x in self.wires if x not in wires]

        # Trace first subsystem by applying kraus operators of the partial trace
        tr_op = self._cast(np.eye(2), dtype=self.C_DTYPE)
        tr_op = self._reshape(tr_op, (2, 1, 2))

        self._apply_channel(tr_op, Wires(traced_wires[0]))

        # Trace next subsystem by applying kraus operators of the partial trace
        for traced_wire in traced_wires[1:]:
            self._apply_channel(tr_op, Wires(traced_wire))

        return self._reshape(self._state, (2 ** len(wires), 2 ** len(wires)))

    def reset(self):
        """Resets the device"""
        super().reset()

        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):

        if self._state is None:
            return None

        # convert rho from tensor to matrix
        rho = self._reshape(self._state, (2**self.num_wires, 2**self.num_wires))
        # probs are diagonal elements
        probs = self.marginal_prob(self._diag(rho), wires)

        # take the real part so probabilities are not shown as complex numbers
        return self._abs(self._real(probs))

    def _get_kraus(self, operation):  # pylint: disable=no-self-use
        """Return the Kraus operators representing the operation.

        Args:
            operation (.Operation): a PennyLane operation

        Returns:
            list[array[complex]]: Returns a list of 2D matrices representing the Kraus operators. If
            the operation is unitary, returns a single Kraus operator. In the case of a diagonal
            unitary, returns a 1D array representing the matrix diagonal.
        """
        if operation in diagonal_in_z_basis:
            return operation.get_eigvals()

        if isinstance(operation, Channel):
            return operation.kraus_matrices()

        return [operation.get_matrix()]

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
        if kraus[0].shape[0] == kraus[0].shape[1]:
            kraus_shape = [len(kraus)] + [2] * num_ch_wires * 2
            kraus = self._cast(self._reshape(kraus, kraus_shape), dtype=self.C_DTYPE)
            kraus_dagger = self._cast(self._reshape(kraus_dagger, kraus_shape), dtype=self.C_DTYPE)

        # Add the possibility to give a (1,2) shape Kraus operator
        elif (kraus[0].shape == (1, 2)) and (num_ch_wires == 1):
            kraus_shape = [len(kraus)] + list(kraus[0].shape)
            kraus = self._cast(self._reshape(kraus, kraus_shape), dtype=self.C_DTYPE)
            kraus_dagger_shape = [len(kraus)] + list(kraus[0].shape)[::-1]
            kraus_dagger = self._cast(
                self._reshape(kraus_dagger, kraus_dagger_shape), dtype=self.C_DTYPE
            )

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
            f"{kraus_index}{new_row_indices}{row_indices}, {state_indices},"
            f"{kraus_index}{col_indices}{new_col_indices}->{new_state_indices}"
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

        einsum_indices = f"{row_indices},{state_indices},{col_indices}->{state_indices}"

        self._state = self._einsum(einsum_indices, eigvals, self._state, self._conj(eigvals))

    def _apply_basis_state(self, state, wires):
        """Initialize the device in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be initialized on
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(device_wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self.num_wires - 1 - device_wires.toarray())
        num = int(np.dot(state, basis_states))

        self._state = self._create_basis_state(num)

    def _apply_state_vector(self, state, device_wires):
        """Initialize the internal state in a specified pure state.

        Args:
            state (array[complex]): normalized input state of length
                ``2**len(wires)``
            device_wires (Wires): wires that get initialized in the state
        """

        # translate to wire labels used by device
        device_wires = self.map_wires(device_wires)

        state = self._asarray(state, dtype=self.C_DTYPE)
        n_state_vector = state.shape[0]

        if state.ndim != 1 or n_state_vector != 2 ** len(device_wires):
            raise ValueError("State vector must be of length 2**wires.")

        if not np.allclose(np.linalg.norm(state, ord=2), 1.0, atol=tolerance):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        if len(device_wires) == self.num_wires and sorted(device_wires.labels) == list(
            device_wires.labels
        ):
            # Initialize the entire wires with the state
            rho = self._outer(state, self._conj(state))
            self._state = self._reshape(rho, [2] * 2 * self.num_wires)

        else:
            # generate basis states on subset of qubits via the cartesian product
            basis_states = np.array(list(itertools.product([0, 1], repeat=len(device_wires))))

            # get basis states to alter on full set of qubits
            unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
            unravelled_indices[:, device_wires] = basis_states

            # get indices for which the state is changed to input state vector elements
            ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

            state = self._scatter(ravelled_indices, state, [2**self.num_wires])
            rho = self._outer(state, self._conj(state))
            rho = self._reshape(rho, [2] * 2 * self.num_wires)
            self._state = self._asarray(rho, dtype=self.C_DTYPE)

    def _apply_density_matrix(self, state, device_wires):
        r"""Initialize the internal state in a specified mixed state.
        If not all the wires are specified in the full state :math:`\rho`, remaining subsystem is filled by
        `\mathrm{tr}_in(\rho)`, which results in the full system state :math:`\mathrm{tr}_{in}(\rho) \otimes \rho_{in}`,
        where :math:`\rho_{in}` is the argument `state` of this function and :math:`\mathrm{tr}_{in}` is a partial
        trace over the subsystem to be replaced by this operation.

           Args:
               state (array[complex]): density matrix of length
                   ``(2**len(wires), 2**len(wires))``
               device_wires (Wires): wires that get initialized in the state
        """

        # translate to wire labels used by device
        device_wires = self.map_wires(device_wires)

        state = self._asarray(state, dtype=self.C_DTYPE)
        state = qnp.reshape(state, (-1,))

        state_dim = 2 ** len(device_wires)
        dm_dim = state_dim**2
        if dm_dim != state.shape[0]:
            raise ValueError("Density matrix must be of length (2**wires, 2**wires)")

        if not qnp.allclose(
            qnp.trace(qnp.reshape(state, (state_dim, state_dim))), 1.0, atol=tolerance
        ):
            raise ValueError("Trace of density matrix is not equal one.")

        if len(device_wires) == self.num_wires and sorted(device_wires.labels) == list(
            device_wires.labels
        ):
            # Initialize the entire wires with the state
            self._state = self._reshape(state, [2] * 2 * self.num_wires)

        else:
            # Initialize tr_in(ρ) ⊗ ρ_in with transposed wires where ρ is the density matrix before this operation.

            complement_wires = list(sorted(list(set(range(self.num_wires)) - set(device_wires))))
            sigma = self.density_matrix(Wires(complement_wires))
            rho = qnp.kron(sigma, state.reshape(state_dim, state_dim))
            rho = rho.reshape([2] * 2 * self.num_wires)

            # Construct transposition axis to revert back to the original wire order
            left_axes = []
            right_axes = []
            complement_wires_count = len(complement_wires)
            for i in range(self.num_wires):
                if i in device_wires:
                    index = device_wires.index(i)
                    left_axes.append(complement_wires_count + index)
                    right_axes.append(complement_wires_count + index + self.num_wires)
                elif i in complement_wires:
                    index = complement_wires.index(i)
                    left_axes.append(index)
                    right_axes.append(index + self.num_wires)
            transpose_axes = left_axes + right_axes
            rho = qnp.transpose(rho, axes=transpose_axes)
            assert qnp.allclose(
                qnp.trace(qnp.reshape(rho, (2**self.num_wires, 2**self.num_wires))),
                1.0,
                atol=tolerance,
            )
            self._state = self._asarray(rho, dtype=self.C_DTYPE)

    def _apply_operation(self, operation):
        """Applies operations to the internal device state.

        Args:
            operation (.Operation): operation to apply on the device
        """
        wires = operation.wires

        if isinstance(operation, QubitStateVector):
            self._apply_state_vector(operation.parameters[0], wires)
            return

        if isinstance(operation, BasisState):
            self._apply_basis_state(operation.parameters[0], wires)
            return

        if isinstance(operation, QubitDensityMatrix):
            self._apply_density_matrix(operation.parameters[0], wires)
            return

        matrices = self._get_kraus(operation)

        if operation in diagonal_in_z_basis:
            self._apply_diagonal_unitary(matrices, wires)
        else:
            self._apply_channel(matrices, wires)

    # pylint: disable=arguments-differ
    def apply(self, operations, rotations=None, **kwargs):

        rotations = rotations or []

        # apply the circuit operations
        for i, operation in enumerate(operations):

            if i > 0 and isinstance(operation, (QubitStateVector, BasisState)):
                raise DeviceError(
                    f"Operation {operation.name} cannot be used after other Operations have already been applied "
                    f"on a {self.short_name} device."
                )

        for operation in operations:
            self._apply_operation(operation)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            self._apply_operation(operation)
