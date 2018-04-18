"""
Unit tests for the :mod:`openqml` plugin interface.
"""

import unittest
# HACK to import the openqml modules from this source distribution, not the one in site-packages
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

#from numpy.random import (randn, uniform, randint)
#from numpy import array, pi

import openqml
from openqml.plugin import (list_plugins, load_plugin, Plugin)



class BaseTest(unittest.TestCase):
    """ABC for tests.
    TODO Encapsulates the user-given commandline parameters for the test run.
    """
    def setUp(self):
        #self.plugin = openqml.load_plugin('strawberryfields')
        pass

    def test_load_plugin(self):
        "Plugin discovery and loading."

        plugins = list_plugins()
        print('Found plugins: ', plugins)

        # nonexistent plugin exception
        self.assertRaises(ValueError, load_plugin, 'nonexistent_plugin')

        def test_plugin(p):
            "Tests for each individual plugin."
            self.assertTrue(issubclass(p, Plugin))

        # try loading all the discovered plugins, test them
        for name in plugins:
            p = load_plugin(name)
            print(p)
            test_plugin(p)



if __name__ == '__main__':
    print('Testing OpenQML version ' + openqml.version() + ', plugin API.')

    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (BaseTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
