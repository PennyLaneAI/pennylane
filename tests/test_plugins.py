"""
Unit tests for the :mod:`openqml` plugin interface.
"""

import unittest

from numpy.random import (randn,)
#from numpy import array, pi

from defaults import openqml, BaseTest
from openqml.plugin import (list_plugins, load_plugin, PluginAPI)
from openqml.circuit import (Command, Circuit, QNode)


class BasicTest(BaseTest):
    """ABC for tests.
    """
    def setUp(self):
        #self.plugin = openqml.load_plugin('strawberryfields')
        pass

    def test_load_plugin(self):
        "Plugin discovery and loading, circuit execution."

        plugins = list_plugins()
        print('Found plugins: ', plugins, '\n')

        # nonexistent plugin exception
        self.assertRaises(ValueError, load_plugin, 'nonexistent_plugin')

        def test_plugin(p):
            "Tests for each individual plugin."
            self.assertTrue(isinstance(p, PluginAPI))
            print(p)
            print('Gates:')
            for g in p.gates():
                # try running each gate with random parameters
                print(g)
                cmd = Command(g, list(range(g.n_sys)), randn(g.n_par))
                print(cmd)
                p.execute_circuit(Circuit([cmd], g.name))
            print('\nCircuit templates:')
            for c in p.templates():
                # try running each circuit template with random parameters
                print(c)
                p.execute_circuit(c, randn(c.n_par))

        # try loading all the discovered plugins, test them
        for name in plugins:
            try:
                p = load_plugin(name)
            except ImportError:
                continue
            test_plugin(p())
            print()


    def test_measurement(self):
        "Basic expectation value measurement."

        p = load_plugin('dummy_plugin')
        p = p('test instance')
        obs = p.observables()
        p.execute_circuit('demo', params=[1.0, 2.0])
        temp = p.measure(obs[0], 0, n_eval=5000)
        print('Measured:', temp)


    def test_qnode(self):
        "Quantum node and node gradient evaluation."

        p = load_plugin('dummy_plugin')
        c = p.get_circuit('demo_ev')
        p1 = p('node 1')
        qn = QNode(c, p1)
        params = [1.0, 2.0]
        x0 = qn.evaluate(params)
        grad1 = qn.gradient_finite_diff(params)
        grad2 = qn.gradient_angle(params)
        # gradients computed with different methods must agree
        self.assertAllAlmostEqual(grad1, grad2, self.tol)


    def test_autograd(self):
        "Automatic differentiation of quantum nodes."

        p = load_plugin('dummy_plugin')
        c = p.get_circuit('demo_ev')
        p1 = p('node 1')
        qn = QNode(c, p1)
        params = [1.0, 2.0]
        x0 = qn.evaluate(params)
        grad1 = qn.gradient_finite_diff(params)
        grad2 = qn.gradient_angle(params)
        # gradients computed with different methods must agree
        self.assertAllAlmostEqual(grad1, grad2, self.tol)

        def loss(p):
            ret = 1
            ret += circuit1(2 * p[:3])  #__call__: re-initializes linked plugin, propagates, returns ev:s, deletes state to save memory?
            # we have defined circuit.grad(p) as the gradient func...
            # one plugin per circuit instance???
            # or maybe circuit is just the program, and qnode is the node object? qnode contains a circuit and a plugin ref
            ret += circuit2(p[3:] ** 2)
            return norm(ret)


if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', plugin API.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
