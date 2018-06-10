# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Example implementation for an OpenQML plugin
============================================

**Module name:** :mod:`openqml.plugins.dummy_plugin`

.. currentmodule:: openqml.plugins.dummy_plugin

The dummy plugin is meant to be used as a template for writing OpenQML plugin modules for new backends.
It implements all the API functions and provides a very simple simulation of a qubit-based quantum circuit architecture.
"""

import warnings

import numpy as np
from scipy.linalg import expm, eigh

import openqml.plugin
from openqml.circuit import (GateSpec, Command, ParRef, Circuit)


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
#  define the gate set
#========================================================

class Gate(GateSpec):
    """Implements the quantum gates and measurements.
    """
    def __init__(self, name, reg, par=[], func=None):
        super().__init__(name, reg, par)
        if not callable(func):
            # for fixed gates given e.g. as NumPy arrays
            self.func = lambda: func
        else:
            self.func = func  #: callable: function that returns the gate matrix

# gates
rx = Gate('rx', 1, 1, frx)
ry = Gate('ry', 1, 1, fry)
rz = Gate('rz', 1, 1, frz)
r3 = Gate('r3', 1, 3, fr3)
cnot = Gate('CNOT', 2, 0, CNOT)
swap = Gate('SWAP', 2, 0, SWAP)

# observable
ev_z = Gate('z', 1, 0, Z)

# circuit templates
_circuit_list = [
    Circuit([
        Command(rx, [0], [ParRef(0)]),
        Command(cnot, [0, 1]),
        Command(ry, [0], [-1.6]),
        Command(ry, [1], [ParRef(0)]),
        Command(cnot, [1, 0]),
        Command(rx, [0], [ParRef(1)]),
        Command(cnot, [0, 1])
    ], 'demo'),
    Circuit([
        Command(rx, [0], [ParRef(0)]),
        Command(cnot, [0, 1]),
        Command(ry, [0], [-1.6]),
        Command(ry, [1], [ParRef(0)]),
        Command(cnot, [1, 0]),
        Command(rx, [0], [ParRef(1)]),
        Command(cnot, [0, 1])
    ], 'demo_ev', obs=Command(ev_z, [0])),
    Circuit([
        Command(r3, [0], [ParRef(0), 0.3, -0.2]),
        Command(swap, [0, 1]),
    ], 'rubbish'),
]


class PluginAPI(openqml.plugin.PluginAPI):
    """Example implementation of an OpenQML plugin API class.

    Provides a very simple simulation of a qubit-based quantum circuit architecture.
    """
    plugin_name = 'dummy plugin'
    plugin_api_version = '0.1.0'
    plugin_version = '1.0.0'
    author = 'Xanadu Inc.'
    _gates = [rx, ry, rz, r3, cnot, swap]
    _observables = [ev_z]
    _circuits = {c.name: c for c in _circuit_list}

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.reset()

    def reset(self):
        self.n = None  #: int: number of qubits in the state
        self._state = None  #: array: state vector

    def _execute_gate(self, gate, par, reg):
        """Applies a single gate or measurement on the current system state.

        Args:
          gate           (Gate): gate type
          par (Sequence[float]): gate parameters
          reg   (Sequence[int]): subsystems to which the gate is applied
        """
        U = gate.func(*par)  # get the matrix
        if gate.n_sys == 1:
            U = self.apply_one(U, reg)
        elif gate.n_sys == 2:
            U = self.apply_two(U, reg)
        else:
            raise ValueError('This plugin supports only one- and two-qubit gates.')
        self._state = U @ self._state

    def apply_one(self, U, reg):
        """Expand a one-qubit gate into a full system propagator.

        Args:
          U (array): 2*2 unitary matrix
          reg (int): target subsystem

        Returns:
          array: 2^n*2^n unitary matrix
        """
        if U.shape != (2, 2):
            raise ValueError('2x2 unitary required.')
        reg = reg[0]
        before = 2**reg
        after  = 2**(self.n-reg-1)
        U = np.kron(np.kron(np.eye(before), U), np.eye(after))
        return U


    def apply_two(self, U, reg):
        """Expand a two-qubit gate into a full system propagator.

        Args:
          U (array): 4x4 unitary matrix
          reg (Sequence[int]): two target subsystems (order matters!)

        Returns:
          array: 2^n*2^n unitary matrix
        """
        if U.shape != (4, 4):
            raise ValueError('4x4 unitary required.')
        if len(reg) != 2:
            raise ValueError('Two target subsystems required.')
        reg = np.asarray(reg)
        if np.any(reg < 0) or np.any(reg >= self.n) or reg[0] == reg[1]:
            raise ValueError('Bad target subsystems.')

        a = np.min(reg)
        b = np.max(reg)
        n_between = b-a-1  # number of qubits between a and b
        # dimensions of the untouched subsystems
        before  = 2**a
        after   = 2**(self.n-b-1)
        between = 2**n_between

        U = np.kron(U, np.eye(between))
        # how U should be reordered
        if reg[0] < reg[1]:
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

    def ev(self, A, reg):
        """Expectation value of an observable in the current state.

        Args:
          A (array): 2*2 hermitian matrix corresponding to the observable
          reg (int): target subsystem

        Returns:
          float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        if A.shape != (2, 2):
            raise ValueError('2x2 matrix required.')
        A = self.apply_one(A, [reg])
        temp = np.vdot(self._state, A @ self._state)
        if np.abs(temp.imag) > tolerance:
            warnings.warn('Nonvanishing imaginary part {} in expectation value.'.format(temp.imag))
        return temp.real

    def measure(self, A, reg, n_eval=None):
        A = A.func()
        ev  = self.ev(A, reg)
        var = self.ev(A**2, reg) -ev**2
        if n_eval is not None:
            if 0:
                # use central limit theorem, sample normal distribution once, only ok if n_eval is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
                ev = np.random.normal(ev, np.sqrt(var / n_eval))
            else:
                # sample Bernoulli distribution n_eval times / binomial distribution once
                a, P = spectral_decomposition_qubit(A)
                p0 = self.ev(P[0], reg)  # probability of measuring a[0]
                n0 = np.random.binomial(n_eval, p0)
                ev = (n0*a[0] +(n_eval-n0)*a[1]) / n_eval
        return ev, var

    def _execute_circuit(self, circuit, params=[], **kwargs):
        if self._state is None:
            # init the state vector to |00..0>
            self.n = circuit.n_sys
            self._state = np.zeros(2**self.n, dtype=complex)
            self._state[0] = 1
        elif self.n != circuit.n_sys:
            raise ValueError("Trying to execute a {}-qubit circuit '{}' on a {}-qubit state.".format(circuit.n_sys, circuit.name, self.n))

        def parmap(p):
            "Mapping function for gate parameters. Replaces ParRefs with the corresponding parameter values."
            if isinstance(p, ParRef):
                return params[p.idx]
            return p

        for c in circuit.seq:
            # prepare the parameters
            par = map(parmap, c.par)
            self._execute_gate(c.gate, par, c.reg)

        if circuit.obs is not None:
            ev, var = self.measure(circuit.obs.gate, circuit.obs.reg[0], **kwargs)
            return ev



def init_plugin(**kwargs):
    """Every plugin must define this function.

    It should perform whatever initializations are necessary, and then return the API class.

    Returns:
      class: plugin API class
    """
    return PluginAPI
