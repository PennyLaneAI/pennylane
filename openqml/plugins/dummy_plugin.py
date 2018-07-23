# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Example OpenQML plugin
======================

**Module name:** :mod:`openqml.plugins.dummy_plugin`

.. currentmodule:: openqml.plugins.dummy_plugin

The dummy plugin is meant to be used as a template for writing OpenQML plugin modules for new backends.
It implements all the :class:`~openqml.plugin.PluginAPI` methods and provides a very simple pure state
simulation of a qubit-based quantum circuit architecture.

Functions
---------

.. autosummary::
   init_plugin
   spectral_decomposition_qubit
   frx
   fry
   frz
   fr3

Classes
-------

.. autosummary::
   Gate
   Observable
   PluginAPI

----
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
    """Implements the quantum gates.
    """
    def __init__(self, name, n_sys, n_par, func=None):
        super().__init__(name, n_sys, n_par)
        if not callable(func):
            # for fixed gates given e.g. as NumPy arrays
            self.func = lambda: func
        else:
            self.func = func  #: callable: function that returns the gate matrix

    def execute(self, par, reg, sim):
        """Applies a single gate or measurement on the current system state.

        Args:
          par (Sequence[float]): gate parameters
          reg   (Sequence[int]): subsystems to which the gate is applied
          sim (~openqml.plugin.PluginAPI): simulator instance keeping track of the system state and measurement results

        Returns:
          vector[complex]: evolved system state vector
        """
        U = self.func(*par)  # get the matrix
        if self.n_sys == 1:
            U = sim.expand_one(U, reg)
        elif self.n_sys == 2:
            U = sim.expand_two(U, reg)
        else:
            raise ValueError('This plugin supports only one- and two-qubit gates.')
        return U @ sim._state


class Observable(Gate):
    """Implements single-qubit hermitian observables.

    We assume that all the observables in the circuit are consequtive, and commute.
    Since we are only interested in the expectation values, there is no need to project the state after the measurement.
    See :ref:`measurements`.
    """
    def execute(self, par, reg, sim):
        """Estimates the expectation value of the observable in the current system state.

        The arguments and return value are the same as for :meth:`Gate.execute`.
        """
        if self.n_sys != 1:
            raise ValueError('This plugin supports only one-qubit observables.')

        A = self.func(*par)  # get the matrix
        n_eval = sim.n_eval
        if n_eval == 0:
            # exact expectation value
            ev = sim.ev(A, reg)
        else:
            # estimate the ev
            if 0:
                # use central limit theorem, sample normal distribution once, only ok if n_eval is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
                ev = sim.ev(A, reg)
                var = sim.ev(A**2, reg) -ev**2  # variance
                ev = np.random.normal(ev, np.sqrt(var / n_eval))
            else:
                # sample Bernoulli distribution n_eval times / binomial distribution once
                a, P = spectral_decomposition_qubit(A)
                p0 = sim.ev(P[0], reg)  # probability of measuring a[0]
                n0 = np.random.binomial(n_eval, p0)
                ev = (n0*a[0] +(n_eval-n0)*a[1]) / n_eval

        sim._result[reg[0]] = ev  # store the result
        return sim._state  # no change to state


# gates
rx = Gate('rx', 1, 1, frx)
ry = Gate('ry', 1, 1, fry)
rz = Gate('rz', 1, 1, frz)
r3 = Gate('r3', 1, 3, fr3)
cnot = Gate('CNOT', 2, 0, CNOT)
swap = Gate('SWAP', 2, 0, SWAP)

# observables
ev_z = Observable('z', 1, 0, Z)

demo = [
    Command(rx, [0], [ParRef(0)]),
    Command(cnot, [0, 1]),
    Command(ry, [0], [-1.6]),
    Command(ry, [1], [ParRef(0)]),
    Command(cnot, [1, 0]),
    Command(rx, [0], [ParRef(1)]),
    Command(cnot, [0, 1])
]

# circuit templates
_circuit_list = [
    Circuit(demo, 'demo'),
    Circuit(demo +[Command(ev_z, [0])], 'demo_ev', out=[0]),
    Circuit([
        Command(r3, [0], [ParRef(0), 0.3, -0.2]),
        Command(swap, [0, 1]),
    ], 'rubbish'),
    Circuit([  # data classifier circuit, ParRef(0) represents the data
        Command(rx, [0], [ParRef(0)]),
        Command(cnot, [0, 1]),
        Command(rx, [0], [ParRef(1)]),
        Command(rz, [1], [2.7]),
        Command(cnot, [0, 1]),
        Command(rx, [0], [-1.8]),
        Command(rz, [1], [ParRef(2)]),
        Command(rx, [1], [ParRef(6)]),
        Command(rz, [1], [ParRef(7)]),
        Command(cnot, [0, 1]),
        Command(rx, [0], [ParRef(3)]),
        Command(rz, [1], [ParRef(4)]),
        Command(cnot, [0, 1]),
        Command(rx, [0], [ParRef(5)]),
        Command(ev_z, [0])
    ], 'opt_ev', out=[0]),
]


class PluginAPI(openqml.plugin.PluginAPI):
    """Example implementation of an OpenQML plugin API class.

    Provides a very simple simulation of a qubit-based quantum circuit architecture.
    """
    plugin_name = 'Dummy OpenQML plugin'
    plugin_api_version = '0.1.0'
    plugin_version = '1.0.0'
    author = 'Xanadu Inc.'
    _gates = {g.name: g for g in [rx, ry, rz, r3, cnot, swap]}
    _observables = {g.name: g for g in [ev_z]}
    _circuits = {c.name: c for c in _circuit_list}

    def __init__(self, name='default', **kwargs):
        super().__init__(name, **kwargs)
        self.reset()

    def reset(self):
        self.n = None  #: int: number of qubits in the state
        self._state  = None  #: array: state vector
        self._result = None  #: array: measurement results

    def expand_one(self, U, reg):
        """Expand a one-qubit operator into a full system operator.

        Args:
          U (array): 2*2 matrix
          reg (Sequence[int]): target subsystem

        Returns:
          array: 2^n*2^n matrix
        """
        if U.shape != (2, 2):
            raise ValueError('2x2 matrix required.')
        if len(reg) != 1:
            raise ValueError('One target subsystem required.')
        reg = reg[0]
        before = 2**reg
        after  = 2**(self.n-reg-1)
        U = np.kron(np.kron(np.eye(before), U), np.eye(after))
        return U

    def expand_two(self, U, reg):
        """Expand a two-qubit operator into a full system operator.

        Args:
          U (array): 4x4 matrix
          reg (Sequence[int]): two target subsystems (order matters!)

        Returns:
          array: 2^n*2^n matrix
        """
        if U.shape != (4, 4):
            raise ValueError('4x4 matrix required.')
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
        r"""Expectation value of a one-qubit observable in the current state.

        Args:
          A (array): 2*2 hermitian matrix corresponding to the observable
          reg (Sequence[int]): target subsystem

        Returns:
          float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        if A.shape != (2, 2):
            raise ValueError('2x2 matrix required.')
        A = self.expand_one(A, reg)
        temp = np.vdot(self._state, A @ self._state)
        if np.abs(temp.imag) > tolerance:
            warnings.warn('Nonvanishing imaginary part {} in expectation value.'.format(temp.imag))
        return temp.real

    def measure(self, A, reg, par=[], n_eval=0):
        temp = self.n_eval  # store the original
        self.n_eval = n_eval
        A.execute(par, [reg], self)
        self.n_eval = temp  # restore it
        return self._result[reg]

    def execute_circuit(self, circuit, params=[], *, reset=True, **kwargs):
        super().execute_circuit(circuit, params, reset=reset, **kwargs)
        circuit = self.circuit

        if self._state is None:
            # init the state vector to |00..0>
            self.n = circuit.n_sys
            self._state = np.zeros(2**self.n, dtype=complex)
            self._state[0] = 1
            self._result = np.full(self.n, np.nan)
        elif self.n != circuit.n_sys:
            raise ValueError("Trying to execute a {}-qubit circuit '{}' on a {}-qubit state.".format(circuit.n_sys, circuit.name, self.n))

        def parmap(p):
            "Mapping function for gate parameters. Replaces ParRefs with the corresponding parameter values."
            if isinstance(p, ParRef):
                return params[p.idx]
            return p

        for cmd in circuit.seq:
            # prepare the parameters
            par = map(parmap, cmd.par)
            # apply the gate to the current state
            self._state = cmd.gate.execute(par, cmd.reg, self)

        if circuit.out is not None:
            # return the measurement results for the requested modes
            return self._result[circuit.out]




def init_plugin():
    """Initialize the plugin.

    Every plugin must define this function.
    It should perform whatever initializations are necessary, and then return the API class.

    Returns:
      class: plugin API class
    """
    return PluginAPI
