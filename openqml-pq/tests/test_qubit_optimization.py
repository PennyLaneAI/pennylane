"""Qubit optimization example for the ProjectQ plugin.

In this demo, we perform rotation on one qubit, entangle it via a CNOT
gate to a second qubit, and then measure the second qubit projected onto PauliZ.
We then optimize the circuit such the resulting expectation value is 1.
"""
import unittest
from unittest_data_provider import data_provider
import os, sys
sys.path.append(os.getcwd())
from defaults import BaseTest
import openqml as qm
from openqml import numpy as np

class QubitOptimizationTests(BaseTest):
    """Test a simple one qubit rotation gate optimization."""
    num_subsystems = 2
    def setUp(self):
        super().setUp()
        self.dev1 = qm.device('projectq.'+self.args.backend, wires=2, **vars(self.args))


    def all_optimizers():
        return tuple([(optimizer,) for optimizer in qm.optimizer.OPTIMIZER_NAMES])

    @data_provider(all_optimizers)
    def test_qubit_optimization(self, optimizer):
        """ """
        if optimizer in ["dogleg", "trust-ncg", "trust-exact", "trust-krylov"]:
            return #these optimizers need a Hessian, so we don't test against them

        @qm.qfunc(self.dev1)
        def circuit(x, y, z):
            """QNode"""
            qm.RZ(z, [0])
            qm.RY(y, [0])
            qm.RX(x, [0])
            qm.CNOT([0, 1])
            return qm.expectation.PauliZ(1)


        def cost(x, batched):
            """Cost (error) function to be minimized."""
            return np.abs(circuit(x)-1)

        # initialize x with "random" value
        x0 = np.array([0.2,-0.1,0.5])
        o = qm.Optimizer(cost, x0, optimizer=optimizer)

        # train the circuit
        c = o.train(max_steps=100)


        self.assertAllAlmostEqual(circuit(*o.weights), 1, delta=0.002, msg="Optimizer "+optimizer+" failed to achieve the optimal value.")
        self.assertAllAlmostEqual(o.weights[0], 0, delta=0.002, msg="Optimizer "+optimizer+" failed to find the optimal x angles.")
        self.assertAllAlmostEqual(o.weights[1], 0, delta=0.002, msg="Optimizer "+optimizer+" failed to find the optimal y angles.")
