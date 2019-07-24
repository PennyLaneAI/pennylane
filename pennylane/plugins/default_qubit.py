# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Default qubit plugin
====================

**Module name:** :mod:`pennylane.plugins.default_qubit`

**Short name:** ``"default.qubit"``

.. currentmodule:: pennylane.plugins.default_qubit

The default plugin is meant to be used as a template for writing PennyLane device
plugins for new qubit-based backends.

It implements the necessary :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qubit operations <pennylane.ops.qubit>`, and provides a very simple pure state
simulation of a qubit-based quantum circuit architecture.

The following is the technical documentation of the implementation of the plugin. You will
not need to read and understand this to use this plugin.

Auxiliary functions
-------------------

.. autosummary::
    spectral_decomposition
    unitary
    hermitian

Gates and operations
--------------------

.. autosummary::
    Rphi
    Rotx
    Roty
    Rotz
    Rot3
    X
    Y
    Z
    H
    CNOT
    SWAP
    CZ
    CRotx
    CRoty
    CRotz
    CRot3

Observables
------------

.. autosummary::
    X
    Y
    Z

Classes
-------

.. autosummary::
    DefaultQubit

Code details
^^^^^^^^^^^^
"""
import warnings

import numpy as np
from scipy.linalg import expm, eigh

from pennylane import Device


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
    return expm(-1j * theta/2 * X)


def Roty(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return expm(-1j * theta/2 * Y)


def Rotz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return expm(-1j * theta/2 * Z)


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


class DefaultQubit(Device):
    """Default qubit device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. A value of 0 yields the exact result.
    """
    name = 'Default qubit PennyLane plugin'
    short_name = 'default.qubit'
    pennylane_requires = '0.5'
    version = '0.4.0'
    author = 'Xanadu Inc.'

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
        'CNOT': CNOT,
        'SWAP': SWAP,
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

    def __init__(self, wires, *, shots=0):
        super().__init__(wires, shots)
        self.eng = None
        self._state = None

    def pre_apply(self):
        self.reset()

    def apply(self, operation, wires, par):
        if operation == 'QubitStateVector':
            state = np.asarray(par[0], dtype=np.complex128)
            if state.ndim == 1 and state.shape[0] == 2**self.num_wires:
                self._state = state
            else:
                raise ValueError('State vector must be of length 2**wires.')
            if wires is not None and wires != [] and list(wires) != list(range(self.num_wires)):
                raise ValueError("The default.qubit plugin can apply QubitStateVector only to all of the {} wires.".format(self.num_wires))
            return
        if operation == 'BasisState':
            n = len(par[0])
            # get computational basis state number
            if n > self.num_wires or not (set(par[0]) == {0, 1} or set(par[0]) == {0} or set(par[0]) == {1}):
                raise ValueError("BasisState parameter must be an array of 0 or 1 integers of length at most {}.".format(self.num_wires))
            if wires is not None and wires != [] and list(wires) != list(range(self.num_wires)):
                raise ValueError("The default.qubit plugin can apply BasisState only to all of the {} wires.".format(self.num_wires))

            num = int(np.sum(np.array(par[0])*2**np.arange(n-1, -1, -1)))

            self._state = np.zeros_like(self._state)
            self._state[num] = 1.
            return

        A = self._get_operator_matrix(operation, par)
        self._state = self.mat_vec_product(A, self._state, wires)

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

    def expval(self, observable, wires, par):
        if self.shots == 0:
            # exact expectation value
            A = self._get_operator_matrix(observable, par)
            ev = self.ev(A, wires)
        else:
            # estimate the ev
            ev = np.mean(self.sample(observable, wires, par, self.shots))

        return ev

    def var(self, observable, wires, par):
        if self.shots == 0:
            # exact expectation value
            A = self._get_operator_matrix(observable, par)
            var = self.ev(A@A, wires) - self.ev(A, wires)**2
        else:
            # estimate the ev
            var = np.var(self.sample(observable, wires, par, self.shots))

        return var

    def sample(self, observable, wires, par, n=None):
        if n is None:
            n = self.shots

        if n == 0:
            raise ValueError("Calling sample with n = 0 is not possible.")
        if n < 0 or not isinstance(n, int):
            raise ValueError("The number of samples must be a positive integer.")

        A = self._get_operator_matrix(observable, par)
        a, P = spectral_decomposition(A)

        p = np.zeros(a.shape)
        for idx, Pi in enumerate(P):
            p[idx] = self.ev(Pi, wires)

        return np.random.choice(a, n, p=p)

    def _get_operator_matrix(self, operation, par):
        """Get the operator matrix for a given operation or observable.

        Args:
          operation    (str): name of the operation/observable
          par (tuple[float]): parameter values
        Returns:
          array: matrix representation.
        """
        A = {**self._operation_map, **self._observable_map}[operation]
        if not callable(A):
            return A
        return A(*par)

    def ev(self, A, wires):
        r"""Expectation value of observable on specified wires.

         Args:
          A (array[float]): the observable matrix as array
          wires (Sequence[int]): target subsystems
         Returns:
          float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        As = self.mat_vec_product(A, self._state, wires)
        expectation = np.vdot(self._state, As)

        if np.abs(expectation.imag) > tolerance:
            warnings.warn('Nonvanishing imaginary part {} in expectation value.'.format(expectation.imag), RuntimeWarning)
        return expectation.real

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        self._state = np.zeros(2**self.num_wires, dtype=complex)
        self._state[0] = 1

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def observables(self):
        return set(self._observable_map.keys())
