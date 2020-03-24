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

import numpy as onp

import jax.numpy as np
from jax import jit
from jax.ops import index, index_update
from jax.config import config

from pennylane import QubitDevice, DeviceError, QubitStateVector, BasisState

config.update("jax_enable_x64", True)


tolerance = 1e-10
C_DTYPE = np.complex128
R_DTYPE = np.float64


I = np.eye(2, dtype=C_DTYPE)
X = np.array([[0, 1], [1, 0]], dtype=C_DTYPE)     #: Pauli-X matrix
Y = np.array([[0, -1j], [1j, 0]], dtype=C_DTYPE)  #: Pauli-Y matrix
Z = np.array([[1, 0], [0, -1]], dtype=C_DTYPE)    #: Pauli-Z matrix

II = np.eye(4, dtype=C_DTYPE)
ZZ = np.kron(Z, Z)

IX = np.kron(I, X)
IY = np.kron(I, Y)
IZ = np.kron(I, Z)

ZI = np.kron(Z, I)
ZX = np.kron(Z, X)
ZY = np.kron(Z, Y)


@jit
def Rphi(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        array: unitary 2x2 phase shift matrix
    """
    return ((1 + np.exp(1j * phi)) * I + (1 - np.exp(1j * phi)) * Z) / 2


@jit
def Rotx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * X


@jit
def Roty(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y


@jit
def Rotz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z


@jit
def CRotx(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    return (
        np.cos(theta / 4) ** 2 * II
        - 1j * np.sin(theta / 2) / 2 * IX
        + np.sin(theta / 4) ** 2 * ZI
        + 1j * np.sin(theta / 2) / 2 * ZX
    )


@jit
def CRoty(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    return (
        np.cos(theta / 4) ** 2 * II
        - 1j * np.sin(theta / 2) / 2 * IY
        + np.sin(theta / 4) ** 2 * ZI
        + 1j * np.sin(theta / 2) / 2 * ZY
    )


@jit
def CRotz(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    return (
        np.cos(theta / 4) ** 2 * II
        - 1j * np.sin(theta / 2) / 2 * IZ
        + np.sin(theta / 4) ** 2 * ZI
        + 1j * np.sin(theta / 2) / 2 * ZZ
    )


def _mat_vec_product(mat, vec, wires, N):
    r"""Apply multiplication of a matrix to subsystems of the quantum state.

    Args:
        mat (array): matrix to multiply
        vec (array): state vector to multiply
        wires (Sequence[int]): target subsystems
        N (int): total number of subsystems

    Returns:
        array: output vector after applying ``mat`` to input ``vec`` on specified subsystems
    """
    mat = np.reshape(mat, [2] * len(wires) * 2)
    vec = np.reshape(vec, np.full([N], 2))
    axes = (list(range(len(wires), 2 * len(wires))), wires)
    tdot = np.tensordot(mat, vec, axes=axes)

    # tensordot causes the axes given in `wires` to end up in the first positions
    # of the resulting tensor. This corresponds to a (partial) transpose of
    # the correct output state
    # We'll need to invert this permutation to put the indices in the correct place
    unused_idxs = [idx for idx in range(N) if idx not in wires]
    perm = np.array(wires + unused_idxs)
    inv_perm = np.argsort(perm)  # argsort gives inverse permutation
    state_multi_index = np.transpose(tdot, inv_perm)
    return np.reshape(state_multi_index, 2 ** N)


mat_vec_product = jit(_mat_vec_product, static_argnums=(2, 3))


def _probability(state, wires, analytic, samples, shots):
    num_wires = int(np.log2(len(state)))

    if state is None:
        return None

    wires = wires or range(num_wires)

    if analytic:
        prob = marginal_prob(np.abs(state) ** 2, wires)
        return prob

    # non-analytic mode, estimate the probability from the generated samples

    # consider only the requested wires
    wires = np.hstack(wires)
    samples = samples[:, np.array(wires)]

    # convert samples from a list of 0, 1 integers,
    # to base 10 representation
    unraveled_indices = [2] * len(wires)
    indices = np.ravel_multi_index(samples.T, unraveled_indices)

    # count the basis state occurrences, and construct
    # the probability vector
    basis_states, counts = np.unique(indices, return_counts=True)
    prob = np.zeros([len(wires)**2], dtype=np.float64)
    prob = index_update(prob, index[basis_states], counts/shots)
    return prob


probability = jit(_probability, static_argnums=(1, 2))


def _marginal_prob(prob, wires):
    r"""Return the marginal probability of the computational basis
    states by summing the probabiliites on the non-specified wires.

    If no wires are specified, then all the basis states representable by
    the device are considered and no marginalization takes place.

    .. note::

        If the provided wires are not strictly increasing, the returned marginal
        probabilities take this permuation into account.

        For example, if ``wires=[2, 0]``, then the returned marginal
        probability vector will take this 'reversal' of the two wires
        into account:

        .. math::

            \mathbb{P}^{(2, 0)} = \[ |00\rangle, |10\rangle, |01\rangle, |11\rangle\]

    Args:
        prob: The probabilities to return the marginal probabilities
            for
        wires (Sequence[int]): Sequence of wires to return
            marginal probabilities for. Wires not provided
            are traced out of the system.

    Returns:
        array[float]: array of the resulting marginal probabilities.
    """
    num_wires = int(np.log2(len(prob)))

    if wires is None:
        # no need to marginalize
        return prob

    wires = np.hstack(wires)

    # determine which wires are to be summed over
    inactive_wires = list(set(range(num_wires)) - set(wires))

    # reshape the probability so that each axis corresponds to a wire
    prob = prob.reshape([2] * num_wires)

    # sum over all inactive wires
    # prob = apply_over_axes(np.sum, prob, inactive_wires).flatten()

    for axis in inactive_wires:
        if axis < 0:
            axis = num_wires + axis

        res = np.sum(prob, axis)

        if res.ndim == prob.ndim:
            prob = res
        else:
            res = np.expand_dims(res, axis)

            if res.ndim == prob.ndim:
                prob = res

    prob = prob.reshape([2] * len(wires))

    # The wires provided might not be in consecutive order (i.e., wires might be [2, 0]).
    # If this is the case, we must permute the marginalized probability so that
    # it corresponds to the orders of the wires passed.
    basis_states = np.array(list(itertools.product([0, 1], repeat=len(wires))))
    basis_states = basis_states[:, np.argsort(np.argsort(wires))]
    perm = np.sum(basis_states*(2**np.arange(len(wires), -1, -1)), axis=1)
    return prob[perm]


marginal_prob = jit(_marginal_prob, static_argnums=(1,))


class DefaultQubitJAX(QubitDevice):
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

    name = "Default qubit JAX PennyLane plugin"
    short_name = "default.qubit.jax"
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
        "CRX",
        "CRY",
        "CRZ",
    }

    parametric_ops = {
        "PhaseShift": Rphi,
        "RX": Rotx,
        "RY": Roty,
        "RZ": Rotz,
        "CRX": CRotx,
        "CRY": CRoty,
        "CRZ": CRotz,
    }

    observables = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian", "Identity"}

    _asarray = staticmethod(np.stack)

    def __init__(self, wires, *, shots=1000, analytic=True):
        self.eng = None
        self.analytic = analytic

        self._state = np.zeros(2 ** wires, dtype=C_DTYPE)
        self._state = index_update(self._state, index[0], 1)
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
                input_state = np.asarray(par[0], dtype=C_DTYPE)
                self.apply_state_vector(input_state, wires)

            elif isinstance(operation, BasisState):
                basis_state = par[0]
                self.apply_basis_state(basis_state, wires)

            else:
                if operation.name in self.parametric_ops:
                    matrix = self.parametric_ops[operation.name](*par)
                else:
                    matrix = operation.matrix

                self._state = mat_vec_product(matrix, self._state, wires, self.num_wires)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            wires = operation.wires
            par = operation.parameters

            if operation.name in self.parametric_ops:
                matrix = self.parametric_ops[operation.name](*par)
            else:
                matrix = operation.matrix

            self._state = self.mat_vec_product(matrix, self._state, wires, self.num_wires)

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
            self._state = index_update(self._state, index[ravelled_indices], input_state)
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
        self._state = index_update(self._state, index[num], 1.0)

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        super().reset()
        self._state = np.zeros(2 ** self.num_wires, dtype=complex)
        self._state = index_update(self._state, index[0], 1)
        self._pre_rotated_state = self._state

    def probability(self, wires=None):
        analytic = True if (self.analytic or self._samples is None) else False
        prob = probability(self._state, wires, analytic, self._samples, self.shots)
        return prob

    def expval(self, observable):
        wires = observable.wires

        if self.analytic:
            # exact expectation value
            eigvals = observable.eigvals
            prob = self.probability(wires=wires)
            return (eigvals @ prob).real

        # estimate the ev
        return np.mean(self.sample(observable))
