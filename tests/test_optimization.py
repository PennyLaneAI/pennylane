"""
Unit tests for the :mod:`openqml` :class:`Optimizer` class.
"""

import unittest
import logging as log
log.getLogger()


import autograd
import autograd.numpy as np
from autograd.numpy.random import (randn,)

from matplotlib.pyplot import figure

from defaults import openqml as qm, BaseTest
from openqml import Optimizer, QNode


def circuit(*args):
    qm.RX(args[0], 0)
    qm.CNOT([0, 1])
    qm.RX(args[1], 0)
    qm.RZ(2.7, 1)
    qm.CNOT([0, 1])
    qm.RX(-1.8, 0)
    qm.RZ(args[2], 1)
    qm.RX(args[6], 1)
    qm.RZ(args[7], 1)
    qm.CNOT([0, 1])
    qm.RX(args[3], 0)
    qm.RZ(args[4], 1)
    qm.CNOT([0, 1])
    qm.RX(args[5], 0)
    qm.expectation.PauliZ(0)


class OptTest(BaseTest):
    """Optimizer tests.
    """
    def setUp(self):
        # arbitrary classification data
        self.data = np.array([[-0.8, -0.4],
                              [-0.5, 0.3],
                              [-0.2, -0.2],
                              [0.1, 0.1],
                              [0.4, 0.5],
                              [1.0, 0.9]], dtype=float)


    def map_data(self, data_in, weights):
        """Maps input data using a parametrized quantum circuit.

        Args:
          data_in (float): input data
          weights (array[float]): optimization parameters
        Returns:
          float: mapped data
        """
        par = np.concatenate((np.array([0.5*np.pi * data_in]), weights))
        return self.qnode(par)


    def cost(self, weights, data_sample=None):
        """Cost (error) function to be minimized.

        Implements a quantum classifier, trying to map input data to output data.

        Args:
          weights (array[float]): optimization parameters
          data_sample (array[int], None): For stochastic gradient methods, indices of the data samples to use in the error calculation.
            If None, all data samples are used.
        Returns:
          float: cost
        """
        if data_sample is None:
            data = self.data
        else:
            data = self.data[data_sample]

        cost = 0
        for d in data:
            temp = self.map_data(d[0], weights) -d[1]
            cost = cost + temp ** 2
        return cost


    def plot_result(self, weights):
        """Plot the classification function."""
        xx = np.linspace(-1, 1, 100)
        yy = np.array([self.map_data(x, weights) for x in xx], dtype=float)
        fig = figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(xx, yy, 'k-')
        ax.plot(self.data[:,0], self.data[:,1], 'rs')
        ax.legend(['classification', 'data'])
        ax.grid(True)
        ax.set_title('Quantum classifier')
        fig.show()


    def test_opt_errors(self):
        "Make sure faulty input raises an exception."
        weights = np.array([0])
        def f(p):
            return 0

        # initial weights must be given as an array
        self.assertRaises(TypeError, Optimizer, *(f,0), optimizer='SGD')
        self.assertRaises(TypeError, Optimizer, *(f,[0]), optimizer='SGD')

        # optimizer has to be a callable or a known name
        self.assertRaises(ValueError, Optimizer, *(f,weights), optimizer='Unknown')


    def test_opt(self):
        "Test all supported optimization algorithms on a simple optimization task."

        self.dev = qm.device('default.qubit', wires=2)
        qnode = QNode(circuit, self.dev)
        self.qnode = qnode

        x0 = np.array([-0.71690972, -0.55632194,  0.74297438, -1.15401698,  0.62766983,  2.55008079, -0.27567698]) #fix "random" values to make the test pass/fail deterministically

        temp = 0.30288  # expected minimal cost
        tol = 0.001

        o = Optimizer(self.cost, x0, n_data=self.data.shape[0], optimizer='SGD')
        o.set_hp(batch_size=4)
        c = o.train(100)
        self.plot_result(o.weights)
        self.assertAlmostLess(c, temp, delta=0.2)  # SGD requires more iterations to converge well

        opts = ['BFGS', 'CG', 'L-BFGS-B', 'TNC', 'SLSQP']
        opts_nograd = ['Nelder-Mead', 'Powell']

        for opt in opts+opts_nograd:
            o = Optimizer(self.cost, x0, optimizer=opt)
            c = o.train()
            self.assertAlmostEqual(c, temp, delta=tol)


if __name__ == '__main__':
    print('Testing OpenQML version ' + qm.version() + ', Optimizer class.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (OptTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
