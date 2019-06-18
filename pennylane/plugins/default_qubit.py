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
:mod:`qubit operations <pennylane.ops.qubit>` and
:mod:`expectations <pennylane.expval.qubit>`, and provides a very simple pure state
simulation of a qubit-based quantum circuit architecture.

The following is the technical documentation of the implementation of the plugin. You will
not need to read and understand this to use this plugin.

Auxillary functions
-------------------

.. autosummary::
    spectral_decomposition_qubit
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

Expectations
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
import itertools
import warnings

import numpy as np
from scipy.linalg import expm, eigh

from pennylane import Device


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
    """Identity matrix for expectations.

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
    pennylane_requires = '0.4'
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
        'Rot': Rot3
    }

    _expectation_map = {
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

        # apply unitary operations
        U = self.expand(A, wires)

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

    def _get_operator_matrix(self, operation, par):
        """Get the operator matrix for a given operation or expectation.

        Args:
          operation    (str): name of the operation/expectation
          par (tuple[float]): parameter values
        Returns:
          array: matrix representation.
        """
        A = {**self._operation_map, **self._expectation_map}[operation]
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
        A = self.expand(A, wires)

        expectation = np.vdot(self._state, A @ self._state)

        if np.abs(expectation.imag) > tolerance:
            warnings.warn('Nonvanishing imaginary part {} in expectation value.'.format(expectation.imag), RuntimeWarning)
        return expectation.real

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        self._state = np.zeros(2**self.num_wires, dtype=complex)
        self._state[0] = 1

    def expand(self, U, wires):
        r"""Expand a multi-qubit operator into a full system operator.

        Args:
          U (array): :math:`2^n \times 2^n` matrix where n = len(wires).
          wires (Sequence[int]): Target subsystems (order matters!)

        Returns:
          array: :math:`2^N\times 2^N` matrix. The full system operator.
        """
        if self.num_wires == 1:
            # total number of wires is 1, simply return the matrix
            return U

        N = self.num_wires
        wires = np.asarray(wires)

        if np.any(wires < 0) or np.any(wires >= N) or len(set(wires)) != len(wires):
            raise ValueError("Invalid target subsystems provided in 'wires' argument.")

        # generate N qubit basis states via the cartesian product
        tuples = np.array(list(itertools.product([0, 1], repeat=N)))

        # wires not acted on by the operator
        inactive_wires = list(set(range(N))-set(wires))

        # expand U to act on the entire system
        U = np.kron(U, np.identity(2**len(inactive_wires)))

        # move active wires to beginning of the list of wires
        rearanged_wires = np.array(list(wires)+inactive_wires)

        # convert to computational basis
        # i.e., converting the list of basis state bit strings into
        # a list of decimal numbers that correspond to the computational
        # basis state. For example, [0, 1, 0, 1, 1] = 2^3+2^1+2^0 = 11.
        perm = np.ravel_multi_index(tuples[:, rearanged_wires].T, [2]*N)

        # permute U to take into account rearranged wires
        return U[:, perm][perm]

    @property
    def operations(self):
        return set(self._operation_map.keys())

    @property
    def expectations(self):
        return set(self._expectation_map.keys())
