"""
Unit tests for all installed plugins.
"""

import unittest
from unittest_data_provider import data_provider
import logging as log
log.getLogger()

from pkg_resources import iter_entry_points
import inspect

import openqml as qm
import numpy as np

# import autograd
# import autograd.numpy as np
# from autograd.numpy.random import (randn,)

# from matplotlib.pyplot import figure

from defaults import openqml as qm, BaseTest
from openqml import Device

class PluginTest(BaseTest):
    """Plugin test.
    """
    def all_plugins():
        return tuple([ (entry,) for entry in iter_entry_points('openqml.plugins')])

    @data_provider(all_plugins)
    def test_resolve_all_plugins(self, plugin):
        obj = plugin.resolve()
        self.assertIsNotNone(obj, msg="Plugin "+plugin.name+" advertised entry point "+str(plugin)+" but it could not be resolved.")

    @data_provider(all_plugins)
    def test_load_all_plugins(self, plugin):
        device = plugin.load()
        self.assertIsNotNone(device, msg="Plugin "+plugin.name+" advertised device "+str(plugin)+" but it could not be loaded.")

    @data_provider(all_plugins)
    def test_plugin_device(self, plugin):
        print(plugin.name)
        obj = plugin.resolve()
        device = plugin.load()

        #fullargspec = inspect.getfullargspec(obj)
        #print(fullargspec)
        #print(fullargspec.args[1::])

        sig = inspect.signature(obj)
        bind = sig.bind_partial(wires=3)
        bind.apply_defaults()
        #print(bind)

        try:
            dev = device(*bind.args, **bind.kwargs)
        except (ValueError, TypeError) as e:
            print("The device "+plugin.name+" could not be instantiated with only the standard parameters because ("+str(e)+"). Skipping automatic test.")
            return

        # run all single gate circuits
        for gate in dev.gates:
            for observable in dev.observables:
                print("circuit: "+gate+", "+observable)

                # gate_sig = inspect.signature(getattr(qm, gate))
                # print(gate_sig)

                #observable_fullargspec = inspect.getfullargspec(getattr(qm.expectation, observable))
                #print(observable_fullargspec)

                #from openqml import Expectation

                @qm.qfunc(dev)
                def circuit():

                    observable_class = getattr(qm.expectation, observable)
                    observable_fullargspec = inspect.getfullargspec(observable_class.__init__)
                    observable_num_par_args = len(observable_fullargspec.args)-2
                    observable_pars = np.random.randn(observable_num_par_args)

                    gate_class = getattr(qm, gate)
                    gate_fullargspec = inspect.getfullargspec(gate_class.__init__)
                    gate_num_par_args = len(gate_fullargspec.args)-2
                    gate_pars = np.random.randn(gate_num_par_args)

                    gate_class(*gate_pars, [0,1,3])#todo: find out how to know the number of subsystems a gate is supposed to act on...
                    return observable_class(*observable_pars, [0])

                circuit()


if __name__ == '__main__':
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (PluginTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
