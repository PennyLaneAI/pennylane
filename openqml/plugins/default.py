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
"""This module contains the device class and context manager"""
import numpy as np
from scipy.linalg import expm, eigh

import openqml as qm
from openqml import Device, DeviceError, qfunc, QNode, Variable, __version__


# tolerance for numerical errors
tolerance = 1e-10


#========================================================
#  utilities
#========================================================

def spectral_decomposition_qubit(A):
    r"""Spectral decomposition of a 2*2 Hermitian matrix.

    Args:
      A (array): 2*2 Hermitian matrix

    Returns:
      (vector[float], list[array[complex]]): (a, P): eigenvalues and hermitian projectors
        such that :math:`A = \sum_k a_k P_k`.
    """
    d, v = eigh(A)
    P = []
    for k in range(2):
        temp = v[:, k]
        P.append(np.outer(temp.conj(), temp))
    return d, P


#========================================================
#  fixed gates
#========================================================

I = np.eye(2)
# Pauli matrices
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


#========================================================
#  parametrized gates
#========================================================


def frx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return expm(-1j * theta/2 * X)


def fry(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return expm(-1j * theta/2 * Y)


def frz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return expm(-1j * theta/2 * Z)


def fr3(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 2x2 rotation matrix rz(c) @ ry(b) @ rz(a)
    """
    return frz(c) @ (fry(b) @ frz(a))


#========================================================
#  Arbitrary states and operators
#========================================================

def ket(*args):
    r"""Input validation for an arbitary state vector.

    Args:
        args (array): NumPy array.

    Returns:
        array: normalised array.
    """
    state = np.asarray(args)
    return state/np.linalg.norm(state)


def unitary(*args):
    r"""Input validation for an arbitary unitary operation.

    Args:
        args (array): square unitary matrix.

    Returns:
        array: square unitary matrix.
    """
    U = np.asarray(args[0])

    if U.shape[0] != U.shape[1]:
        raise ValueError("Operator must be a square matrix.")

    if not np.allclose(U @ U.conj().T, np.identity(U.shape[0]), atol=tolerance):
        raise ValueError("Operator must be unitary.")

    return U


def hermitian(*args):
    r"""Input validation for an arbitary Hermitian observable.

    Args:
        args (array): square hermitian matrix.

    Returns:
        array: square hermitian matrix.
    """
    A = np.asarray(args[0])

    if A.shape[0] != A.shape[1]:
        raise ValueError("Observable must be a square matrix.")

    if not np.allclose(A, A.conj().T, atol=tolerance):
        raise ValueError("Observable must be Hermitian.")
    return A


#========================================================
#  operator map
#========================================================


operator_map = {
    'QubitStateVector': ket,
    'QubitUnitary': unitary,
    'Hermitian': hermitian,
    'Identity': I,
    'PauliX': X,
    'PauliY': Y,
    'PauliZ': Z,
    'CNOT': CNOT,
    'SWAP': SWAP,
    'RX': frx,
    'RY': fry,
    'RZ': frz,
    'Rot': fr3
}


#========================================================
#  device
#========================================================


class DefaultQubit(Device):
    """Default qubit device for OpenQML.

    wires (int): the number of modes to initialize the device in.
    cutoff (int): the Fock space truncation. Must be specified before
        applying a qfunc.
    hbar (float): the convention chosen in the canonical commutation
        relation [x, p] = i hbar. The default value is hbar=2.
    """
    name = 'Default OpenQML plugin'
    short_name = 'default.qubit'
    api_version = '0.1.0'
    version = '0.1.0'
    author = 'Xanadu Inc.'
    _gates = set(operator_map.keys())
    _observables = {}
    _circuits = {}

    def __init__(self, wires, *, shots=0):
        self.wires = wires
        self.eng = None
        self._state = None
        super().__init__(self.short_name, shots)

    def execute(self):
        """Apply the queued operations to the device, and measure the expectation."""
        if self._state is None:
            # init the state vector to |00..0>
            self._state = np.zeros(2**self.wires, dtype=complex)
            self._state[0] = 1
            self._out = np.full(self.wires, np.nan)

        # apply unitary operations U
        for operation in self._queue:
            if operation.name == 'QubitStateVector':
                state = np.asarray(operation.params[0])
                if state.ndim == 1 and state.shape[0] == 2**self.wires:
                    self._state = state
                else:
                    raise ValueError('State vector must be of length 2**wires.')
                continue

            U = DefaultQubit._get_operator_matrix(operation)

            if len(operation.wires) == 1:
                U = self.expand_one(U, operation.wires)
            elif len(operation.wires) == 2:
                U = self.expand_two(U, operation.wires)
            else:
                raise ValueError('This plugin supports only one- and two-qubit gates.')
            self._state = U @ self._state

        # measurement/expectation value <psi|A|psi>
        A = DefaultQubit._get_operator_matrix(self._observe)
        if self.shots == 0:
            # exact expectation value
            ev = self.ev(A, [self._observe.wires])
        else:
            # estimate the ev
            if 0:
                # use central limit theorem, sample normal distribution once, only ok if n_eval is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
                ev = self.ev(A, self._observe.wires)
                var = self.ev(A**2, self._observe.wires) - ev**2  # variance
                ev = np.random.normal(ev, np.sqrt(var / self.shots))
            else:
                # sample Bernoulli distribution n_eval times / binomial distribution once
                a, P = spectral_decomposition_qubit(A)
                p0 = self.ev(P[0], self._observe.wires)  # probability of measuring a[0]
                n0 = np.random.binomial(self.shots, p0)
                ev = (n0*a[0] +(self.shots-n0)*a[1]) / self.shots

        self._out = ev  # store the result

    @classmethod
    def _get_operator_matrix(cls, A):
        """Get the operator matrix for a given operation.

        Args:
            A (openqml.Operation or openqml.Expectation): operation/observable.

        Returns:
            array: matrix representation.
        """
        if A.name not in operator_map:
            raise DeviceError("{} not supported by device {}".format(A.name, cls.short_name))

        if not callable(operator_map[A.name]):
            return operator_map[A.name]

        # unpack variables
        p = [x.val if isinstance(x, Variable) else x for x in A.params]
        return operator_map[A.name](*p)

    def ev(self, A, wires):
        r"""Expectation value of a one-qubit observable in the current state.

        Args:
          A (array): 2*2 hermitian matrix corresponding to the observable
          wires (Sequence[int]): target subsystem

        Returns:
          float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        if A.shape != (2, 2):
            raise ValueError('2x2 matrix required.')

        A = self.expand_one(A, wires)
        expectation = np.vdot(self._state, A @ self._state)

        if np.abs(expectation.imag) > tolerance:
            log.warning('Nonvanishing imaginary part {} in expectation value.'.format(expectation.imag))
        return expectation.real

    def reset(self):
        """Reset the device"""
        self._state  = None  #: array: state vector
        self._out = None  #: array: measurement results

    def expand_one(self, U, wires):
        """Expand a one-qubit operator into a full system operator.

        Args:
          U (array): 2*2 matrix
          wires (Sequence[int]): target subsystem

        Returns:
          array: 2^n*2^n matrix
        """
        if U.shape != (2, 2):
            raise ValueError('2x2 matrix required.')
        if len(wires) != 1:
            raise ValueError('One target subsystem required.')
        wires = wires[0]
        before = 2**wires
        after  = 2**(self.wires-wires-1)
        U = np.kron(np.kron(np.eye(before), U), np.eye(after))
        return U

    def expand_two(self, U, wires):
        """Expand a two-qubit operator into a full system operator.

        Args:
          U (array): 4x4 matrix
          wires (Sequence[int]): two target subsystems (order matters!)

        Returns:
          array: 2^n*2^n matrix
        """
        if U.shape != (4, 4):
            raise ValueError('4x4 matrix required.')
        if len(wires) != 2:
            raise ValueError('Two target subsystems required.')
        wires = np.asarray(wires)
        if np.any(wires < 0) or np.any(wires >= self.wires) or wires[0] == wires[1]:
            raise ValueError('Bad target subsystems.')

        a = np.min(wires)
        b = np.max(wires)
        n_between = b-a-1  # number of qubits between a and b
        # dimensions of the untouched subsystems
        before  = 2**a
        after   = 2**(self.wires-b-1)
        between = 2**n_between

        U = np.kron(U, np.eye(between))
        # how U should be reordered
        if wires[0] < wires[1]:
            p = [0, 2, 1]
        else:
            p = [1, 2, 0]
        dim = [2, 2, between]
        p = np.array(p)
        perm = np.r_[p, p+3]
        # reshape U into another array which has one index per subsystem, permute dimensions, back into original-shape array
        temp = np.prod(dim)
        U = U.reshape(dim * 2).transpose(perm).reshape([temp, temp])
        U = np.kron(np.kron(np.eye(before), U), np.eye(after))
        return U


#====================
# Default circuits
#====================


dev = DefaultQubit(wires=2)

def node(x, y, z):
    qm.RX(x, [0])
    qm.CNOT([0, 1])
    qm.RY(-1.6, [0])
    qm.RY(y, [1])
    qm.CNOT([1, 0])
    qm.RX(z, [0])
    qm.CNOT([0, 1])
    qm.expectation.Hermitian(np.array([[0, 1], [1, 0]]), 0)

circuits = {'demo_ev': QNode(node, dev)}
