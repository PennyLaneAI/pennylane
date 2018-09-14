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
        wires = 3 #todo: calculate that form the size of the largest gate

        #fullargspec = inspect.getfullargspec(obj)
        #print(fullargspec)
        #print(fullargspec.args[1::])

        sig = inspect.signature(obj)

        if 'cutoff_dim' in sig.parameters:
            bind = sig.bind_partial(wires=wires, cutoff_dim=5)
        else:
            bind = sig.bind_partial(wires=wires)
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
                print(plugin.name+": "+gate+", "+observable)

                #observable_fullargspec = inspect.getfullargspec(observable_class.__init__)
                #observable_num_par_args = observable_class.n_params#len(observable_fullargspec.args)-2
                #gate_fullargspec = inspect.getfullargspec(gate_class.__init__)
                #gate_num_par_args = len(gate_fullargspec.args)-2
                #gate_n_wires = gate_class.n_wires

                @qm.qfunc(dev)
                def circuit():
                    observable_class = getattr(qm.expectation, observable)
                    gate_class = getattr(qm, gate)

                    gate_pars = np.abs(np.random.randn(gate_class.n_params))
                    observable_pars = np.abs(np.random.randn(observable_class.n_params)) #todo: some operations fails when parameters are negative (e.g. thermal state) but par_domain is not fine grained enough to capture this

                    gate_wires = gate_class.n_wires if gate_class.n_wires != 0 else wires
                    observable_wires = observable_class.n_wires if observable_class.n_wires != 0 else wires

                    gate_class(*gate_pars, list(range(gate_class.n_wires)))
                    return observable_class(*observable_pars, list(range(observable_class.n_wires)))

                try:
                    circuit()
                except Exception as e:
                    print(e)


if __name__ == '__main__':
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (PluginTest,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
