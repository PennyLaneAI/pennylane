"""
Unit tests for the :mod:`openqml` plugin interface.
"""

import unittest

#from numpy.random import (randn, uniform, randint)
#from numpy import array, pi

from defaults import openqml, BaseTest
from openqml.plugin import (list_plugins, load_plugin, Plugin)



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
            self.assertTrue(issubclass(p, Plugin))
            print('Name:', p.plugin_name)
            print('API version:', p.plugin_api_version)
            print('Plugin version:', p.plugin_version)

        # try loading all the discovered plugins, test them
        for name in plugins:
            try:
                p = load_plugin(name)
            except ImportError:
                continue
            print(p)
            test_plugin(p)
            print()



if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', plugin API.')
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BasicTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
