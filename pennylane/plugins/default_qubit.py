# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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

import numpy as np
from scipy.linalg import eigh

from pennylane import QubitDevice, DeviceError, QubitStateVector, BasisState
from pennylane.operation import Operation


# tolerance for numerical errors
tolerance = 1e-10


#========================================================
#  utilities
#========================================================

def spectral_decomposition(A):
    r"""Spectral decomposition of a Hermitian matrix.

    Args:
        A (array): Hermitian matrix

    Returns:
        (vector[float], list[array[complex]]): (a, P): eigenvalues and hermitian projectors
            such that :math:`A = \sum_k a_k P_k`.
    """
    d, v = eigh(A)
    P = []
    for k in range(d.shape[0]):
        temp = v[:, k]
        P.append(np.outer(temp, temp.conj()))
    return d, P


#========================================================
#  fixed gates
#========================================================

I = np.eye(2)
# Pauli matrices
X = np.array([[0, 1], [1, 0]]) #: Pauli-X matrix
Y = np.array([[0, -1j], [1j, 0]]) #: Pauli-Y matrix
Z = np.array([[1, 0], [0, -1]]) #: Pauli-Z matrix

H = np.array([[1, 1], [1, -1]])/np.sqrt(2) #: Hadamard gate
# Two qubit gates
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) #: CNOT gate
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) #: SWAP gate
CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]) #: CZ gate
S = np.array([[1, 0], [0, 1j]]) #: Phase Gate
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]) #: T Gate
# Three qubit gates
CSWAP = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1]]) #: CSWAP gate

Toffoli = np.diag([1 for i in range(8)])
Toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])

#========================================================
#  parametrized gates
#========================================================

def Rphi(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle
    Returns:
        array: unitary 2x2 phase shift matrix
    """
    return np.array([[1, 0], [0, np.exp(1j*phi)]])


def Rotx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return np.cos(theta/2) * I + 1j * np.sin(-theta/2) * X


def Roty(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return np.cos(theta/2) * I + 1j * np.sin(-theta/2) * Y


def Rotz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return np.cos(theta/2) * I + 1j * np.sin(-theta/2) * Z


def Rot3(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return Rotz(c) @ (Roty(b) @ Rotz(a))


def CRotx(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(theta/2), -1j*np.sin(theta/2)], [0, 0, -1j*np.sin(theta/2), np.cos(theta/2)]])


def CRoty(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(theta/2), -np.sin(theta/2)], [0, 0, np.sin(theta/2), np.cos(theta/2)]])


def CRotz(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.exp(-1j*theta/2), 0], [0, 0, 0, np.exp(1j*theta/2)]])


def CRot3(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.exp(-1j*(a+c)/2)*np.cos(b/2), -np.exp(1j*(a-c)/2)*np.sin(b/2)], [0, 0, np.exp(-1j*(a-c)/2)*np.sin(b/2), np.exp(1j*(a+c)/2)*np.cos(b/2)]])



#========================================================
#  Arbitrary states and operators
#========================================================

def unitary(*args):
    r"""Input validation for an arbitary unitary operation.

    Args:
        args (array): square unitary matrix

    Raises:
        ValueError: if the matrix is not unitary or square

    Returns:
        array: square unitary matrix
    """
    U = np.asarray(args[0])

    if U.shape[0] != U.shape[1]:
        raise ValueError("Operator must be a square matrix.")

    if not np.allclose(U @ U.conj().T, np.identity(U.shape[0])):
        raise ValueError("Operator must be unitary.")

    return U


def hermitian(*args):
    r"""Input validation for an arbitary Hermitian expectation.

    Args:
        args (array): square hermitian matrix

    Raises:
        ValueError: if the matrix is not Hermitian or square

    Returns:
        array: square hermitian matrix
    """
    A = np.asarray(args[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError("Expectation must be a square matrix.")

    if not np.allclose(A, A.conj().T):
        raise ValueError("Expectation must be Hermitian.")

    return A

def identity(*_):
    """Identity matrix observable.

    Returns:
        array: 2x2 identity matrix
    """
    return np.identity(2)

#========================================================
#  device
#========================================================


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
    name = 'Default qubit PennyLane plugin'
    short_name = 'default.qubit'
    pennylane_requires = '0.8'
    version = '0.8.0'
    author = 'Xanadu Inc.'
    _capabilities = {"model": "qubit", "tensor_observables": True, "inverse_operations": True}

    # Note: BasisState and QubitStateVector don't
    # map to any particular function, as they modify
    # the internal device state directly.
    _operation_map = {
        'BasisState': None,
        'QubitStateVector': None,
        'QubitUnitary': unitary,
        'PauliX': X,
        'PauliY': Y,
        'PauliZ': Z,
        'Hadamard': H,
        'S': S,
        'T': T,
        'CNOT': CNOT,
        'SWAP': SWAP,
        'CSWAP': CSWAP,
        'Toffoli': Toffoli,
        'CZ': CZ,
        'PhaseShift': Rphi,
        'RX': Rotx,
        'RY': Roty,
        'RZ': Rotz,
        'Rot': Rot3,
        'CRX': CRotx,
        'CRY': CRoty,
        'CRZ': CRotz,
        'CRot': CRot3
    }

    _observable_map = {
        'PauliX': X,
        'PauliY': Y,
        'PauliZ': Z,
        'Hadamard': H,
        'Hermitian': hermitian,
        'Identity': identity
    }

    def __init__(self, wires, *, shots=1000, analytic=True):
        super().__init__(wires, shots, analytic)
        self.eng = None
        self.analytic = analytic
        self._state = None

    def apply(self, operations, rotations):
        # apply the circuit operations
        for i, operation in enumerate(operations):
            # number of wires on device
            wires = operation.wires
            par = operation.parameters

            if isinstance(operation, QubitStateVector):
                self.apply_state_vector(wires, par)
                return

            if isinstance(operation, BasisState):
                self.apply_basis_state(wires, par)
                return

            A = self._get_operator_matrix(operation.name, par)
            self._state = self.mat_vec_product(A, self._state, wires)

        # store the pre-rotated state
        self.state = self._state

        # apply the circuit rotations
        for operation in rotations:
            wires = operation.wires
            par = operation.parameters
            A = self._get_operator_matrix(operation.name, par)
            self._state = self.mat_vec_product(A, self._state, wires)

    def apply_state_vector(wires, par):
        input_state = np.asarray(par[0], dtype=np.complex128)

        if not np.isclose(np.linalg.norm(input_state, 2), 1.0, atol=tolerance):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        n_state_vector = input_state.shape[0]

        if i > 0:
            raise DeviceError("Operation {} cannot be used after other Operations have already been applied "
                              "on a {} device.".format(operation.name, self.short_name))

        if input_state.ndim == 1 and n_state_vector == 2**len(wires):
            # generate basis states on subset of qubits via the cartesian product
            tuples = np.array(list(itertools.product([0, 1], repeat=len(wires))))

            # get basis states to alter on full set of qubits
            unravelled_nums = np.zeros((2 ** len(wires), self.num_wires), dtype=int)
            unravelled_nums[:, wires] = tuples

            # get indices for which the state is changed to input state vector elements
            nums = np.ravel_multi_index(unravelled_nums.T, [2] * self.num_wires)
            self._state = np.zeros_like(self._state)
            self._state[nums] = input_state
        else:
            raise ValueError("State vector must be of length 2**wires.")

    def apply_basis_state(wires, par):
        # length of basis state parameter
        n_basis_state = len(par[0])

        if not set(par[0]).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        if i > 0:
            raise DeviceError("Operation {} cannot be used after other Operations have already been applied "
                              "on a {} device.".format(operation.name, self.short_name))

        # get computational basis state number
        num = int(np.dot(par[0], 2**(self.num_wires - 1 - np.array(wires))))

        self._state = np.zeros_like(self._state)
        self._state[num] = 1.

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
        inv_perm = np.argsort(perm) # argsort gives inverse permutation
        state_multi_index = np.transpose(tdot, inv_perm)
        return np.reshape(state_multi_index, 2 ** self.num_wires)

    def get_operator_matrix_for_measurement(self, observable, par):
        """Get the operator matrix for a given observable before measurement.

        Args:
          observable (str or list[str]): name of the operation/observable
          par (tuple[float] or list[list[Any]]): parameter values
        Returns:
          array: matrix representation.
        """
        if isinstance(observable, list):
            return self._get_tensor_operator_matrix(observable, par)

        return self._get_operator_matrix(observable, par)

    def _get_operator_matrix(self, operation, par):
        """Get the operator matrix for a given operation or observable.

        If the inverse was defined for an operation, returns the
        conjugate transpose of the operator matrix.

        Args:
          operation    (str): name of the operation/observable
          par (tuple[float]): parameter values
        Returns:
          array: matrix representation.
        """

        operation_map = {**self._operation_map, **self._observable_map}

        if operation.endswith(Operation.string_for_inverse):
            A = operation_map[operation[:-len(Operation.string_for_inverse)]]
            return A.conj().T if not callable(A) else A(*par).conj().T

        A = operation_map[operation]
        return A if not callable(A) else A(*par)

    def _get_tensor_operator_matrix(self, obs, par):
        """Get the operator matrix for a given tensor product of operations.

        Args:
            obs (list[str]): list of observable names to tensor
            par (list[list[Any]]): parameter values

        Returns:
            array: matrix representation.
        """
        ops = [self._get_operator_matrix(o, p) for o, p in zip(obs, par)]
        return functools.reduce(np.kron, ops)

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        super().reset()
        self._state = np.zeros(2**self.num_wires, dtype=complex)
        self._state[0] = 1

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def observables(self):
        return set(self._observable_map.keys())

    def probability(self, wires=None):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. warning:: This method will have to be redefined for hardware devices, since it uses
            the ``device._state`` attribute. This attribute might not be available for such devices.

        Args:
            wires (Sequence[int]): Sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            List[float]: list of the probabilities
        """
        if self._state is None:
            return None

        wires = wires or range(self.num_wires)
        prob = self.marginal_prob(np.abs(self._state) ** 2, wires)
        return prob
