# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Photon redirection demo
=======================

In this demo we optimize an optical quantum circuit to redirect a photon from mode 1 to mode 2.

----
"""

import os
import sys

import autograd
import autograd.numpy as np
from autograd.numpy.random import (randn,)

# Make sure openqml is always imported from the same source distribution where the tests reside, not e.g. from site-packages.
# See https://docs.python-guide.org/en/latest/writing/structure/#test-suite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import openqml

from openqml.plugin import (load_plugin,)
from openqml.circuit import *
from openqml.optimize import (Optimizer,)



if __name__ == '__main__':
    print('Optimized photon redirection demo.')

    # load the Strawberry Fields CV quantum circuit plugin
    plugin = load_plugin('strawberryfields')

    # instantiate a Fock-basis capable backend
    p = plugin('node1', backend='fock')

    # define some gates and measurements
    g = p.gates
    BS   = g['BS']    # beamsplitter
    Fock = g['Fock']  # Fock state preparation
    o = p.observables
    MFock = o['MFock']  # Fock basis measurement

    BS.grad_method = 'F'  # TODO FIXME: we do not yet support analytic CV circuit differentiation with order-2 observables like n

    # gate sequence for our two-mode circuit with one free parameter
    seq = [
        Command(Fock, [0], [1]),                # prepare mode 0 in the |1> state
        Command(BS, [0, 1], [ParRef(0), 0.0]),  # apply a parametrized beamsplitter between the modes
        Command(MFock, [1]),                    # measure mode 1 (this is the circuit output)
    ]
    circuit = Circuit(seq, 'photon redirection', out=[1])

    # construct a quantum node out of the circuit and the backend
    q = QNode(circuit, p)

    def cost(x):
        """Cost (error) function to be minimized.

        Args:
          x (array[float]): optimization parameters
        """
        temp = q.evaluate(x)  # photon number expectation value <n> in mode 1
        return (temp[0] -1) ** 2  # <n> should be 1

    x0 = randn(q.circuit.n_par)    # initial parameter values
    grad = autograd.grad(cost, 0)  # gradient of the cost function with respect to the parameters
    o = Optimizer(cost, grad, x0, optimizer='BFGS')  # set up an optimizer using the BFGS algorithm
    res = o.train()
    print('Optimized beamsplitter angle: pi *', o.weights / np.pi)
