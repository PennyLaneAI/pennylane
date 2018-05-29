# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Dummy implementation for a plugin
=================================

**Module name:** :mod:`openqml.plugins.dummy_plugin`

.. currentmodule:: openqml.plugins.dummy_plugin

The dummy plugin is meant to be used as a template for writing OpenQML plugins for new backends.
It implements all the API functions and provides a very simple simulation of a qubit-based quantum circuit architecture.
"""

import sys
import numpy as np
from scipy.linalg import expm

from openqml.plugin import Plugin
from openqml.circuit import (GateSpec, Command, ParRef, Circuit)


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
    return rz(c) @ (ry(b) @ rz(a))


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


rx = Gate('rx', 1, 1, frx)
ry = Gate('ry', 1, 1, fry)
rz = Gate('rz', 1, 1, frz)
r3 = Gate('r3', 1, 3, fr3)
cnot = Gate('CNOT', 2, 0, CNOT)
swap = Gate('SWAP', 2, 0, SWAP)

_gates = [rx, ry, rz, r3, cnot, swap]


# circuit templates
_circuits = [
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
    Command(r3, [0], [ParRef(0), 0.3, -0.2]),
    Command(swap, [0, 1]),
  ], 'rubbish'),
]


class DummyPlugin(Plugin):
    """Dummy implementation for OpenQML plugins.
    """
    plugin_name = 'dummy plugin'
    plugin_api_version = '0.1.0'
    plugin_version = '1.0.0'
    author = 'Xanadu Inc.'

    def __init__(self):
        super().__init__()
        self.reset()
        for c in _circuits:
            self.define_circuit(c)

    def reset(self):
        self.state = None

    def get_gateset(self):
        return _gates

    def _execute_gate(self, gate, par, reg):
        """Applies a single gate or measurement on the current system state.
        """
        # get the matrix
        print(gate.name)
        U = gate.func(*par)
        if gate.n_sys == 1:
            self.apply_one(U, reg)
        elif gate.n_sys == 2:
            self.apply_two(U, reg)
        else:
            raise ValueError('This plugin supports only one- and two-qubit gates.')

    def apply_one(self, U, reg):
        """Apply a one-qubit gate to the state vector.

        Args:
          U (array): 2x2 unitary matrix
          reg (int): target subsystem
        """
        if U.shape != (2, 2):
            raise ValueError('2x2 unitary required.')
        reg = reg[0]
        before = 2**reg
        after  = 2**(self.n-reg-1)
        U = np.kron(np.kron(np.eye(before), U), np.eye(after))
        self._state = U @ self._state


    def apply_two(self, U, reg):
        """Apply a two-qubit gate to the state vector.

        Args:
          U (array): 4x4 unitary matrix
        reg (Sequence[int]): two target subsystems (order matters!)
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
        self._state = U @ self._state


    def _execute_circuit(self, circuit, params=[], **kwargs):
        # init the state vector

        self.n = circuit.n_sys  #: int: number of qubits in the state
        self._state = np.zeros(2**self.n, dtype=complex)  #: array: state vector

        def parmap(p):
            "Mapping function for gate parameters. Replaces ParRefs with the corresponding parameter values."
            if isinstance(p, ParRef):
                return params[p.idx]
            return p

        for c in circuit.seq:
            # prepare the parameters
            par = map(parmap, c.par)
            self._execute_gate(c.gate, par, c.reg)



def init_plugin():
    """Every plugin must define this function.

    It should perform whatever initializations are necessary, and then return an API class instance.

    Returns:
      Plugin: plugin API class instance
    """
    return DummyPlugin()
