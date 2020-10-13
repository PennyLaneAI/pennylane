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
The default.qubit device is PennyLane's standard qubit-based device.

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
SQRT2INV = 1 / np.sqrt(2)
TPHASE = np.exp(1j * np.pi / 4)


def _get_slice(index, axis, num_axes):
    """Allows slicing along an arbitrary axis of an array or tensor.

    Args:
        index (int): the index to access
        axis (int): the axis to slice into
        num_axes (int): total number of axes

    Returns:
        tuple[slice or int]: a tuple that can be used to slice into an array or tensor

    **Example:**

    Accessing the 2 index along axis 1 of a 3-axis array:

    >>> sl = _get_slice(2, 1, 3)
    >>> sl
    (slice(None, None, None), 2, slice(None, None, None))
    >>> a = np.arange(27).reshape((3, 3, 3))
    >>> a[sl]
    array([[ 6,  7,  8],
           [15, 16, 17],
           [24, 25, 26]])
    """
    idx = [slice(None)] * num_axes
    idx[axis] = index
    return tuple(idx)


# pylint: disable=unused-argument
class DefaultQubit(QubitDevice):
    """Default qubit device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
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
    pennylane_requires = "0.12"
    version = "0.12.0"
    author = "Xanadu Inc."

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
        "CY",
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

        self._apply_ops = {
            "PauliX": self._apply_x,
            "PauliY": self._apply_y,
            "PauliZ": self._apply_z,
            "Hadamard": self._apply_hadamard,
            "S": self._apply_s,
            "T": self._apply_t,
            "CNOT": self._apply_cnot,
            "SWAP": self._apply_swap,
            "CZ": self._apply_cz,
        }

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
        wires = operation.wires

        if isinstance(operation, QubitStateVector):
            self._apply_state_vector(operation.parameters[0], wires)
            return

        if isinstance(operation, BasisState):
            self._apply_basis_state(operation.parameters[0], wires)
            return

        if operation.name in self._apply_ops:
            axes = self.wires.indices(wires)
            self._state = self._apply_ops[operation.name](
                self._state, axes, inverse=operation.inverse
            )
            return

        matrix = self._get_unitary_matrix(operation)

        if isinstance(operation, DiagonalOperation):
            self._apply_diagonal_unitary(matrix, wires)
        elif len(wires) <= 2:
            # Einsum is faster for small gates
            self._apply_unitary_einsum(matrix, wires)
        else:
            self._apply_unitary(matrix, wires)

    def _apply_x(self, state, axes, **kwargs):
        """Applies a PauliX gate by rolling 1 unit along the axis specified in ``axes``.

        Rolling by 1 unit along the axis means that the :math:`|0 \rangle` state with index ``0`` is
        shifted to the :math:`|1 \rangle` state with index ``1``. Likewise, since rolling beyond
        the last index loops back to the first, :math:`|1 \rangle` is transformed to
        :math:`|0\rangle`.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        return self._roll(state, 1, axes[0])

    def _apply_y(self, state, axes, **kwargs):
        """Applies a PauliY gate by adding a negative sign to the 1 index along the axis specified
        in ``axes``, rolling one unit along the same axis, and multiplying the result by 1j.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        return 1j * self._apply_x(self._apply_z(state, axes), axes)

    def _apply_z(self, state, axes, **kwargs):
        """Applies a PauliZ gate by adding a negative sign to the 1 index along the axis specified
        in ``axes``.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        return self._apply_phase(state, axes, -1)

    def _apply_hadamard(self, state, axes, **kwargs):
        """Apply the Hadamard gate by combining the results of applying the PauliX and PauliZ gates.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        state_x = self._apply_x(state, axes)
        state_z = self._apply_z(state, axes)
        return SQRT2INV * (state_x + state_z)

    def _apply_s(self, state, axes, inverse=False):
        return self._apply_phase(state, axes, 1j, inverse)

    def _apply_t(self, state, axes, inverse=False):
        return self._apply_phase(state, axes, TPHASE, inverse)

    def _apply_cnot(self, state, axes, **kwargs):
        """Applies a CNOT gate by slicing along the first axis specified in ``axes`` and then
        applying an X transformation along the second axis.

        By slicing along the first axis, we are able to select all of the amplitudes with a
        corresponding :math:`|1\rangle` for the control qubit. This means we then just need to apply
        a :class:`~.PauliX` (NOT) gate to the result.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        sl_0 = _get_slice(0, axes[0], self.num_wires)
        sl_1 = _get_slice(1, axes[0], self.num_wires)

        # We will be slicing into the state according to state[sl_1], giving us all of the
        # amplitudes with a |1> for the control qubit. The resulting array has lost an axis
        # relative to state and we need to be careful about the axis we apply the PauliX rotation
        # to. If axes[1] is larger than axes[0], then we need to shift the target axis down by
        # one, otherwise we can leave as-is. For example: a state has [0, 1, 2, 3], control=1,
        # target=3. Then, state[sl_1] has 3 axes and target=3 now corresponds to the second axis.
        if axes[1] > axes[0]:
            target_axes = [axes[1] - 1]
        else:
            target_axes = [axes[1]]

        state_x = self._apply_x(state[sl_1], axes=target_axes)
        return self._stack([state[sl_0], state_x], axis=axes[0])

    def _apply_swap(self, state, axes, **kwargs):
        """Applies a SWAP gate by performing a partial transposition along the specified axes.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        all_axes = list(range(len(state.shape)))
        all_axes[axes[0]] = axes[1]
        all_axes[axes[1]] = axes[0]
        return self._transpose(state, all_axes)

    def _apply_cz(self, state, axes, **kwargs):
        """Applies a CZ gate by slicing along the first axis specified in ``axes`` and then
        applying a Z transformation along the second axis.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        sl_0 = _get_slice(0, axes[0], self.num_wires)
        sl_1 = _get_slice(1, axes[0], self.num_wires)

        if axes[1] > axes[0]:
            target_axes = [axes[1] - 1]
        else:
            target_axes = [axes[1]]

        state_z = self._apply_z(state[sl_1], axes=target_axes)
        return self._stack([state[sl_0], state_z], axis=axes[0])

    def _apply_phase(self, state, axes, parameters, inverse=False):
        """Applies a phase onto the 1 index along the axis specified in ``axes``.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            parameters (float): phase to apply
            inverse (bool): whether to apply the inverse phase

        Returns:
            array[complex]: output state
        """
        num_wires = len(state.shape)
        sl_0 = _get_slice(0, axes[0], num_wires)
        sl_1 = _get_slice(1, axes[0], num_wires)

        phase = self._conj(parameters) if inverse else parameters
        return self._stack([state[sl_0], phase * state[sl_1]], axis=axes[0])

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

    def _apply_state_vector(self, state, device_wires):
        """Initialize the internal state vector in a specified state.

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

        if (
            len(device_wires) == self.num_wires
            and sorted(device_wires.labels) == device_wires.labels
        ):
            # Initialize the entire wires with the state
            self._state = self._reshape(state, [2] * self.num_wires)
            return

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(itertools.product([0, 1], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

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
            wires (Wires): wires that the provided computational state should be initialized on
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state.tolist()).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(device_wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self.num_wires - 1 - device_wires.toarray())
        num = int(np.dot(state, basis_states))

        self._state = self._create_basis_state(num)

    def _apply_unitary(self, mat, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            mat (array): matrix to multiply
            wires (Wires): target wires
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        mat = self._cast(self._reshape(mat, [2] * len(device_wires) * 2), dtype=self.C_DTYPE)
        axes = (np.arange(len(device_wires), 2 * len(device_wires)), device_wires.labels)
        tdot = self._tensordot(mat, self._state, axes=axes)

        # tensordot causes the axes given in `wires` to end up in the first positions
        # of the resulting tensor. This corresponds to a (partial) transpose of
        # the correct output state
        # We'll need to invert this permutation to put the indices in the correct place
        unused_idxs = [idx for idx in range(self.num_wires) if idx not in device_wires.labels]
        perm = list(device_wires.labels) + unused_idxs
        inv_perm = np.argsort(perm)  # argsort gives inverse permutation
        self._state = self._transpose(tdot, inv_perm)

    def _apply_unitary_einsum(self, mat, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        This function uses einsum instead of tensordot. This approach is only
        faster for single- and two-qubit gates.

        Args:
            mat (array): matrix to multiply
            wires (Wires): target wires
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        mat = self._cast(self._reshape(mat, [2] * len(device_wires) * 2), dtype=self.C_DTYPE)

        # Tensor indices of the quantum state
        state_indices = ABC[: self.num_wires]

        # Indices of the quantum state affected by this operation
        affected_indices = "".join(ABC_ARRAY[device_wires.tolist()].tolist())

        # All affected indices will be summed over, so we need the same number of new indices
        new_indices = ABC[self.num_wires : self.num_wires + len(device_wires)]

        # The new indices of the state are given by the old ones with the affected indices
        # replaced by the new_indices
        new_state_indices = functools.reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
            zip(affected_indices, new_indices),
            state_indices,
        )

        # We now put together the indices in the notation numpy's einsum requires
        einsum_indices = (
            "{new_indices}{affected_indices},{state_indices}->{new_state_indices}".format(
                affected_indices=affected_indices,
                state_indices=state_indices,
                new_indices=new_indices,
                new_state_indices=new_state_indices,
            )
        )

        self._state = self._einsum(einsum_indices, mat, self._state)

    def _apply_diagonal_unitary(self, phases, wires):
        r"""Apply multiplication of a phase vector to subsystems of the quantum state.

        This represents the multiplication with diagonal gates in a more efficient manner.

        Args:
            phases (array): vector to multiply
            wires (Wires): target wires
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        # reshape vectors
        phases = self._cast(self._reshape(phases, [2] * len(device_wires)), dtype=self.C_DTYPE)

        state_indices = ABC[: self.num_wires]
        affected_indices = "".join(ABC_ARRAY[device_wires.tolist()].tolist())

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

        if self._state is None:
            return None

        prob = self.marginal_prob(self._abs(self._flatten(self._state)) ** 2, wires)
        return prob
