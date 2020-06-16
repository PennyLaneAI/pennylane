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
The default plugin is meant to be used as a template for writing PennyLane device
plugins for new qubit-based backends.

It implements the necessary :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qubit operations <pennylane.ops.qubit>`, and provides a very simple pure state
simulation of a qubit-based quantum circuit architecture.
"""
import itertools
import functools
from string import ascii_letters as ABC

import numpy as np

from pennylane import QubitDevice, DeviceError, QubitStateVector, BasisState
from pennylane.operation import DiagonalOperation

ABC_ARRAY = np.array(list(ABC))

# tolerance for numerical errors
tolerance = 1e-10


class DefaultQubit(QubitDevice):
    """Default qubit device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to 1000 if not specified.
            If ``analytic == True``, then the number of shots is ignored
            in the calculation of expectation values and variances, and only controls the number
            of samples returned by ``sample``.
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically
    """

    name = "Default qubit PennyLane plugin"
    short_name = "default.qubit"
    pennylane_requires = "0.10"
    version = "0.10.0"
    author = "Xanadu Inc."
    _capabilities = {"inverse_operations": True, "reversible_diff": True}

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
    }

    observables = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian", "Identity"}

    def __init__(self, wires, *, shots=1000, analytic=True):
        # call QubitDevice init
        super().__init__(wires, shots, analytic)

        # Create the initial state. Internally, we store the
        # state as an array of dimension [2]*wires.
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []

        # apply the circuit operations
        for i, operation in enumerate(operations):

            if i > 0 and isinstance(operation, (QubitStateVector, BasisState)):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(operation.name, self.short_name)
                )

            self._apply_operation(operation)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            self._apply_operation(operation)

    def _apply_operation(self, operation):
        """Applies operations to the internal device state.

        Args:
            operation (~.Operation): operation to apply on the device
        """

        if isinstance(operation, QubitStateVector):
            self._apply_state_vector(operation.parameters[0], operation.wires)
            return

        if isinstance(operation, BasisState):
            self._apply_basis_state(operation.parameters[0], operation.wires)
            return

        matrix = self._get_unitary_matrix(operation)

        if isinstance(operation, DiagonalOperation):
            self._apply_diagonal_unitary(matrix, operation.wires)
        elif len(operation.wires) <= 2:
            # Einsum is faster for small gates
            self._apply_unitary_einsum(matrix, operation.wires)
        else:
            self._apply_unitary(matrix, operation.wires)

    def _get_unitary_matrix(self, unitary):  # pylint: disable=no-self-use
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            array[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis, or, in the case of a diagonal unitary,
            a 1D array representing the matrix diagonal.
        """
        if isinstance(unitary, DiagonalOperation):
            return unitary.eigvals

        return unitary.matrix

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state

        Returns:
            array[complex]: complex array of shape ``[2]*self.num_wires``
            representing the statevector of the basis state
        """
        state = np.zeros(2 ** self.num_wires, dtype=np.complex128)
        state[index] = 1
        state = self._asarray(state, dtype=self.C_DTYPE)
        return self._reshape(state, [2] * self.num_wires)

    @property
    def state(self):
        return self._flatten(self._pre_rotated_state)

    def _apply_state_vector(self, state, wires):
        """Initialize the internal state vector in a specified state.

        Args:
            state (array[complex]): normalized input state of length
                ``2**len(wires)``
            wires (list[int]): list of wires where the provided state should
                be initialized
        """
        state = self._asarray(state, dtype=self.C_DTYPE)
        n_state_vector = state.shape[0]

        if state.ndim != 1 or n_state_vector != 2 ** len(wires):
            raise ValueError("State vector must be of length 2**wires.")

        if not np.allclose(np.linalg.norm(state, ord=2), 1.0, atol=tolerance):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        if len(wires) == self.num_wires and sorted(wires) == wires:
            # Initialize the entire register with the state
            self._state = self._reshape(state, [2] * self.num_wires)
            return

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(itertools.product([0, 1], repeat=len(wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(wires), self.num_wires), dtype=int)
        unravelled_indices[:, wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

        state = self._scatter(ravelled_indices, state, [2 ** self.num_wires])
        state = self._reshape(state, [2] * self.num_wires)
        self._state = self._asarray(state, dtype=self.C_DTYPE)

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (list[int]): list of wires where the provided computational state should
                be initialized
        """
        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self.num_wires - 1 - np.array(wires))
        num = int(np.dot(state, basis_states))

        self._state = self._create_basis_state(num)

    def _apply_unitary(self, mat, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            mat (array): matrix to multiply
            wires (Sequence[int]): target subsystems
        """
        mat = self._cast(self._reshape(mat, [2] * len(wires) * 2), dtype=self.C_DTYPE)
        axes = (np.arange(len(wires), 2 * len(wires)), wires)
        tdot = self._tensordot(mat, self._state, axes=axes)

        # tensordot causes the axes given in `wires` to end up in the first positions
        # of the resulting tensor. This corresponds to a (partial) transpose of
        # the correct output state
        # We'll need to invert this permutation to put the indices in the correct place
        unused_idxs = [idx for idx in range(self.num_wires) if idx not in wires]
        perm = list(wires) + unused_idxs
        inv_perm = np.argsort(perm)  # argsort gives inverse permutation
        self._state = self._transpose(tdot, inv_perm)

    def _apply_unitary_einsum(self, mat, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        This function uses einsum instead of tensordot. This approach is only
        faster for single- and two-qubit gates.

        Args:
            mat (array): matrix to multiply
            wires (Sequence[int]): target subsystems
        """
        mat = self._cast(self._reshape(mat, [2] * len(wires) * 2), dtype=self.C_DTYPE)

        # Tensor indices of the quantum state
        state_indices = ABC[: self.num_wires]

        # Indices of the quantum state affected by this operation
        affected_indices = "".join(ABC_ARRAY[wires].tolist())

        # All affected indices will be summed over, so we need the same number of new indices
        new_indices = ABC[self.num_wires : self.num_wires + len(wires)]

        # The new indices of the state are given by the old ones with the affected indices
        # replaced by the new_indices
        new_state_indices = functools.reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
            zip(affected_indices, new_indices),
            state_indices,
        )

        # We now put together the indices in the notation numpy's einsum requires
        einsum_indices = "{new_indices}{affected_indices},{state_indices}->{new_state_indices}".format(
            affected_indices=affected_indices,
            state_indices=state_indices,
            new_indices=new_indices,
            new_state_indices=new_state_indices,
        )

        self._state = self._einsum(einsum_indices, mat, self._state)

    def _apply_diagonal_unitary(self, phases, wires):
        r"""Apply multiplication of a phase vector to subsystems of the quantum state.

        This represents the multiplication with diagonal gates in a more efficient manner.

        Args:
            phases (array): vector to multiply
            wires (Sequence[int]): target subsystems
        """
        # reshape vectors
        phases = self._cast(self._reshape(phases, [2] * len(wires)), dtype=self.C_DTYPE)

        state_indices = ABC[: self.num_wires]
        affected_indices = "".join(ABC_ARRAY[wires].tolist())

        einsum_indices = "{affected_indices},{state_indices}->{state_indices}".format(
            affected_indices=affected_indices, state_indices=state_indices
        )

        self._state = self._einsum(einsum_indices, phases, self._state)

    def reset(self):
        """Reset the device"""
        super().reset()

        # init the state vector to |00..0>
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):
        """Return the (marginal) analytic probability of each computational basis state."""
        if self._state is None:
            return None

        wires = wires or range(self.num_wires)
        prob = self.marginal_prob(self._abs(self._flatten(self._state)) ** 2, wires)
        return prob
