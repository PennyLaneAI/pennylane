"""
Unit tests for the :mod:`openqml` plugin interface.
"""

import unittest

from numpy.random import (randn,)
#from numpy import array, pi

from defaults import openqml, BaseTest
from openqml.plugin import (list_plugins, load_plugin, PluginAPI)
from openqml.circuit import (Command, Circuit)


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

        def test_backend(p):
            """Tests for each individual plugin/backend.

            Args:
              p (PluginAPI): backend to test
            """
            self.assertTrue(isinstance(p, PluginAPI))
            print()
            print(p)
            print('Gates:')
            for g in p.gates.values():
                # try running each gate with random parameters
                print(g)
                cmd = Command(g, list(range(g.n_sys)), randn(g.n_par))
                print(cmd)
                p.execute_circuit(Circuit([cmd], g.name))

            print('\nObservables:')
            for g in p.observables.values():
                print(g)
                cmd = Command(g, list(range(g.n_sys)), randn(g.n_par))
                print(cmd)
                p.execute_circuit(Circuit([cmd], g.name))

            print('\nCircuit templates:')
            for c in p.templates.values():
                # try running each circuit template with random parameters
                print(c)
                p.execute_circuit(c, randn(c.n_par))

            # nonexistent template
            self.assertRaises(KeyError, p.get_circuit, 'nonexistent_template')
            self.assertRaises(KeyError, p.execute_circuit, 'nonexistent_template')

            c = p.get_circuit('demo')

            # wrong number of parameters
            self.assertRaises(ValueError, p.execute_circuit, c, params=randn(c.n_par-1))
            self.assertRaises(ValueError, p.execute_circuit, c, params=randn(c.n_par+1))

            print('\nInteractive measurements:')
            for g in p.observables.values():
                print(g)
                # execute the demo circuit without measurement, then measure the observable manually
                temp = p.execute_circuit(c, randn(c.n_par))
                temp = p.measure(g, 0, par=randn(g.n_par), n_eval=100)

        # try loading all the discovered plugins, test them
        for name in plugins:
            try:
                p = load_plugin(name)
            except ImportError:
                continue
            temp = p.capabilities().get('backend')
            if temp is None:
                # test the default version
                test_backend(p())
            else:
                # test all backends
                for k in temp:
                    test_backend(p(backend=k))
                    print('-' * 80)



if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', plugin API.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
