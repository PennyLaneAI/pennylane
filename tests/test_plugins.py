"""
Unit tests for the :mod:`openqml` plugin interface.
"""

import unittest

from numpy.random import (randn,)
#from numpy import array, pi

from defaults import openqml, BaseTest
from openqml.plugin import (list_plugins, load_plugin, Plugin)
from openqml.circuit import (Command, Circuit)


class BasicTest(BaseTest):
    """ABC for tests.
    """
    def setUp(self):
        #self.plugin = openqml.load_plugin('strawberryfields')
        pass

    def test_load_plugin(self):
        "Plugin discovery and loading."

        plugins = list_plugins()
        print('Found plugins: ', plugins, '\n')

        # nonexistent plugin exception
        self.assertRaises(ValueError, load_plugin, 'nonexistent_plugin')

        def test_plugin(p):
            "Tests for each individual plugin."
            self.assertTrue(isinstance(p, Plugin))
            print(p)
            print('Gates:')
            gates = p.get_gateset()
            for g in gates:
                # try running each gate with random parameters
                print(g)
                cmd = Command(g, list(range(g.n_sys)), randn(g.n_par))
                print(cmd)
                p.reset()
                p.execute_circuit(Circuit([cmd]))
            print('\nCircuit templates:')
            templates = p.get_templates()
            for c in templates:
                # try running each circuit template with random parameters
                print(c)
                p.execute_circuit(c, randn(c.n_par))


        # try loading all the discovered plugins, test them
        for name in plugins:
            try:
                p = load_plugin(name)
            except ImportError:
                continue
            test_plugin(p)
            print()


    def test_usage(self):
        "Simple use case."
        p = load_plugin('dummy_plugin')
        obs = p.get_observables()
        p.execute_circuit('demo', params=[1.0, 2.0])
        temp = p.measure(obs[0], 0, n_eval=5000)
        print(temp)


if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', plugin API.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
