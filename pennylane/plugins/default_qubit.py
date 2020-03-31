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

import numpy as np

from pennylane import QubitDevice, DeviceError, QubitStateVector, BasisState


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
    pennylane_requires = "0.9"
    version = "0.9.0"
    author = "Xanadu Inc."
    _capabilities = {"inverse_operations": True}

    operations = {
        "BasisState",
        "QubitStateVector",
        "QubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
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
        self.eng = None
        self.analytic = analytic

        self._state = np.zeros(2 ** wires, dtype=complex)
        self._state[0] = 1
        self._pre_rotated_state = self._state

        super().__init__(wires, shots, analytic)

    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []

        # apply the circuit operations
        for i, operation in enumerate(operations):
            # number of wires on device
            wires = operation.wires
            par = operation.parameters

            if i > 0 and isinstance(operation, (QubitStateVector, BasisState)):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(operation.name, self.short_name)
                )

            if isinstance(operation, QubitStateVector):
                input_state = np.asarray(par[0], dtype=np.complex128)
                self.apply_state_vector(input_state, wires)

            elif isinstance(operation, BasisState):
                basis_state = par[0]
                self.apply_basis_state(basis_state, wires)

            else:
                self._state = self.mat_vec_product(operation.matrix, self._state, wires)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            wires = operation.wires
            par = operation.parameters
            self._state = self.mat_vec_product(operation.matrix, self._state, wires)

    @property
    def state(self):
        return self._pre_rotated_state

    def apply_state_vector(self, input_state, wires):
        """Initialize the internal state vector in a specified state.

        Args:
            input_state (array[complex]): normalized input state of length
                ``2**len(wires)``
            wires (list[int]): list of wires where the provided state should
                be initialized
        """
        if not np.isclose(np.linalg.norm(input_state, 2), 1.0, atol=tolerance):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        n_state_vector = input_state.shape[0]

        if input_state.ndim == 1 and n_state_vector == 2 ** len(wires):
            # generate basis states on subset of qubits via the cartesian product
            basis_states = np.array(list(itertools.product([0, 1], repeat=len(wires))))

            # get basis states to alter on full set of qubits
            unravelled_indices = np.zeros((2 ** len(wires), self.num_wires), dtype=int)
            unravelled_indices[:, wires] = basis_states

            # get indices for which the state is changed to input state vector elements
            ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)
            self._state = np.zeros_like(self._state)
            self._state[ravelled_indices] = input_state
        else:
            raise ValueError("State vector must be of length 2**wires.")

    def apply_basis_state(self, state, wires):
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

        self._state = np.zeros_like(self._state)
        self._state[num] = 1.0

    def mat_vec_product(self, mat, vec, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            mat (array): matrix to multiply
            vec (array): state vector to multiply
            wires (Sequence[int]): target subsystems

        Returns:
            array: output vector after applying ``mat`` to input ``vec`` on specified subsystems
        """

        # TODO: use multi-index vectors/matrices to represent states/gates internally
        mat = np.reshape(mat, [2] * len(wires) * 2)
        vec = np.reshape(vec, [2] * self.num_wires)
        axes = (np.arange(len(wires), 2 * len(wires)), wires)
        tdot = np.tensordot(mat, vec, axes=axes)

        # tensordot causes the axes given in `wires` to end up in the first positions
        # of the resulting tensor. This corresponds to a (partial) transpose of
        # the correct output state
        # We'll need to invert this permutation to put the indices in the correct place
        unused_idxs = [idx for idx in range(self.num_wires) if idx not in wires]
        perm = wires + unused_idxs
        inv_perm = np.argsort(perm)  # argsort gives inverse permutation
        state_multi_index = np.transpose(tdot, inv_perm)
        return np.reshape(state_multi_index, 2 ** self.num_wires)

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        super().reset()
        self._state = np.zeros(2 ** self.num_wires, dtype=complex)
        self._state[0] = 1
        self._pre_rotated_state = self._state

    def _analytic_probability(self, wires=None):
        """Return the (marginal) analytic probability of each computational basis."""
        if self._state is None:
            return None

        wires = wires or range(self.num_wires)

        prob = self.marginal_prob(np.abs(self._state) ** 2, wires)
        return prob

    def probability(self, wires=None):
        wires = wires or range(self.num_wires)

        if self.analytic:
            return self._analytic_probability(wires=wires)

        # non-analytic mode, estimate the probability from the generated samples

        # consider only the requested wires
        wires = np.hstack(wires)

        samples = self._samples[:, np.array(wires)]

        # convert samples from a list of 0, 1 integers, to base 10 representation
        unraveled_indices = [2] * len(wires)
        indices = np.ravel_multi_index(samples.T, unraveled_indices)

        # count the basis state occurrences, and construct the probability vector
        basis_states, counts = np.unique(indices, return_counts=True)
        prob = np.zeros([len(wires) ** 2], dtype=np.float64)
        prob[basis_states] = counts / self.shots
        return prob
