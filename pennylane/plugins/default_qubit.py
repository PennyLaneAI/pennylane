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

"""
Default qubit plugin
====================

**Module name:** :mod:`pennylane.plugins.default_qubit`

**Short name:** ``"default.qubit"``

.. currentmodule:: pennylane.plugins.default_qubit

The default plugin is meant to be used as a template for writing PennyLane device
plugins for new backends.

It implements all the :class:`~pennylane._device.Device` methods as well as all built-in
discrete-variable operations and expectations, and provides a very simple pure state
simulation of a qubit-based quantum circuit architecture.

Functions
---------

.. autosummary::
   spectral_decomposition_qubit
   Rphi
   frx
   fry
   frz
   fr3
   unitary
   hermitian

Classes
-------

.. autosummary::
   DefaultQubit

Code details
~~~~~~~~~~~~
"""
import logging as log

import numpy as np
from scipy.linalg import expm, eigh

from pennylane import Device

log.getLogger()

# tolerance for numerical errors
tolerance = 1e-10


#========================================================
#  utilities
#========================================================

def spectral_decomposition_qubit(A):
    r"""Spectral decomposition of a :math:`2\times 2` Hermitian matrix.

    Args:
        A (array): :math:`2\times 2` Hermitian matrix

    Returns:
        (vector[float], list[array[complex]]): (a, P): eigenvalues and hermitian projectors
        such that :math:`A = \sum_k a_k P_k`.
    """
    d, v = eigh(A)
    P = []
    for k in range(2):
        temp = v[:, k]
        P.append(np.outer(temp, temp.conj()))
    return d, P


#========================================================
#  fixed gates
#========================================================

I = np.eye(2)
# Pauli matrices
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
# Hadamard
H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
# Two qubit gates
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


#========================================================
#  parametrized gates
#========================================================

def Rphi(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle
    Returns:
        array: unitary 2x2 phase shift matrix.
    """
    return np.array([[1, 0], [0, np.exp(1j*phi)]])


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
        array: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return frz(c) @ (fry(b) @ frz(a))


#========================================================
#  Arbitrary states and operators
#========================================================

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

    if not np.allclose(U @ U.conj().T, np.identity(U.shape[0])):
        raise ValueError("Operator must be unitary.")

    return U


def hermitian(*args):
    r"""Input validation for an arbitary Hermitian expectation.

    Args:
        args (array): square hermitian matrix.

    Returns:
        array: square hermitian matrix.
    """
    A = np.asarray(args[0])

    if A.shape[0] != A.shape[1]:
        raise ValueError("Expectation must be a square matrix.")

    if not np.allclose(A, A.conj().T):
        raise ValueError("Expectation must be Hermitian.")
    return A


#========================================================
#  device
#========================================================


class DefaultQubit(Device):
    """Default qubit device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): How many times should the circuit be evaluated (or sampled) to estimate
            the expectation values. 0 yields the exact result.
    """
    name = 'Default qubit PennyLane plugin'
    short_name = 'default.qubit'
    api_version = '0.1.0'
    version = '0.1.0'
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
        'RX': frx,
        'RY': fry,
        'RZ': frz,
        'Rot': fr3
    }

    _expectation_map = {
        'PauliX': X,
        'PauliY': Y,
        'PauliZ': Z,
        'Hermitian': hermitian
    }

    def __init__(self, wires, *, shots=0):
        super().__init__(self.short_name, wires, shots)
        self.eng = None
        self._state = None

    def pre_apply(self):
        self.reset()

    def apply(self, op_name, wires, par):
        if op_name == 'QubitStateVector':
            state = np.asarray(par[0], dtype=np.float64)
            if state.ndim == 1 and state.shape[0] == 2**self.num_wires:
                self._state = state
            else:
                raise ValueError('State vector must be of length 2**wires.')
            return
        elif op_name == 'BasisState':
            # get computational basis state number
            if not (set(par[0]) == {0, 1} or set(par[0]) == {0} or set(par[0]) == {1}):
                raise ValueError("BasisState parameter must be an array of 0/1 integers.")

            n = len(par[0])
            num = int(np.sum(np.array(par[0])*2**np.arange(n-1, -1, -1)))

            self._state = np.zeros_like(self._state)
            self._state[num] = 1.
            return

        A = self._get_operator_matrix(op_name, par)

        # apply unitary operations
        if len(wires) == 1:
            U = self.expand_one(A, wires)
        elif len(wires) == 2:
            U = self.expand_two(A, wires)
        else:
            raise ValueError('This plugin supports only one- and two-qubit gates.')

        self._state = U @ self._state

    def expval(self, expectation, wires, par):
        # measurement/expectation value <psi|A|psi>
        A = self._get_operator_matrix(expectation, par)
        if self.shots == 0:
            # exact expectation value
            ev = self.ev(A, wires)
        else:
            # estimate the ev
            # sample Bernoulli distribution n_eval times / binomial distribution once
            a, P = spectral_decomposition_qubit(A)
            p0 = self.ev(P[0], wires)  # probability of measuring a[0]
            n0 = np.random.binomial(self.shots, p0)
            ev = (n0*a[0] +(self.shots-n0)*a[1]) / self.shots

        return ev

    def var(self, expectation, wires, par):
        # variance value <psi|A^2|psi> - <psi|A|psi>
        A = self._get_operator_matrix(expectation, par)

        if A.shape != (2, 2):
            raise ValueError('2x2 matrix required.')

        A = self.expand_one(A, wires)
        expectation = np.vdot(self._state, A @ self._state)
        expectationSq = np.vdot(self._state, (A @ A) @ self._state)

        variance = expectationSq - expectation**2
        return variance.real

    def _get_operator_matrix(self, op_name, par):
        """Get the operator matrix for a given operation or expectation.

        Args:
          op_name    (str): name of the operation/expectation
          par (tuple[float]): parameter values
        Returns:
          array: matrix representation.
        """
        A = {**self._operation_map, **self._expectation_map}[op_name]
        if not callable(A):
            return A
        return A(*par)

    def ev(self, A, wires):
        r"""Evaluates a one-qubit expectation in the current state.

        Args:
          A (array): :math:`2\times 2` Hermitian matrix corresponding to the expectation
          wires (Sequence[int]): target subsystem

        Returns:
          float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        if A.shape != (2, 2):
            raise ValueError('2x2 matrix required.')

        A = self.expand_one(A, wires)
        expectation = np.vdot(self._state, A @ self._state)

        if np.abs(expectation.imag) > tolerance:
            log.warning('Nonvanishing imaginary part % in expectation value.', expectation.imag)
        return expectation.real

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        self._state = np.zeros(2**self.num_wires, dtype=complex)
        self._state[0] = 1

    def expand_one(self, U, wires):
        r"""Expand a one-qubit operator into a full system operator.

        Args:
          U (array): :math:`2\times 2` matrix
          wires (Sequence[int]): target subsystem

        Returns:
          array: :math:`2^n\times 2^n` matrix
        """
        if U.shape != (2, 2):
            raise ValueError('2x2 matrix required.')
        if len(wires) != 1:
            raise ValueError('One target subsystem required.')
        wires = wires[0]
        before = 2**wires
        after = 2**(self.num_wires-wires-1)
        U = np.kron(np.kron(np.eye(before), U), np.eye(after))
        return U

    def expand_two(self, U, wires):
        r"""Expand a two-qubit operator into a full system operator.

        Args:
          U (array): :math:`4\times 4` matrix
          wires (Sequence[int]): two target subsystems (order matters!)

        Returns:
          array: :math:`2^n\times 2^n` matrix
        """
        if U.shape != (4, 4):
            raise ValueError('4x4 matrix required.')
        if len(wires) != 2:
            raise ValueError('Two target subsystems required.')
        wires = np.asarray(wires)
        if np.any(wires < 0) or np.any(wires >= self.num_wires) or wires[0] == wires[1]:
            raise ValueError('Bad target subsystems.')

        a = np.min(wires)
        b = np.max(wires)
        n_between = b-a-1  # number of qubits between a and b
        # dimensions of the untouched subsystems
        before = 2**a
        after = 2**(self.num_wires-b-1)
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
