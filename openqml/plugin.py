# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
ABC for plugins
===============

**Module name:** :mod:`openqml.plugin`

.. currentmodule:: openqml.plugin


Functions
---------

.. autosummary::
   list_plugins
   load_plugin
   _load_from_namespace_pkg


Classes
-------

.. autosummary::
   Plugin
"""


import os
import sys
import importlib
import pkgutil

import openqml.plugins



def _load_from_namespace_pkg(ns, name):
    """Iterate namespace package contents, looking for the named module.

    Args:
      ns (module): namespace package
      name (str): module name

    Returns:
      module: loaded module
    """
    for finder, modname, ispkg in pkgutil.iter_modules(ns.__path__):
        if modname == name:
            # load the module
            try:
                mod = importlib.import_module(ns.__name__ +'.' +modname)
                return mod
            except ImportError as err:
                raise err
            break


def list_plugins():
    """Lists all the OpenQML plugins found.

    Returns:
      list[str]: names of all the plugins found in the openqml.plugins namespace package
    """
    return [modname for finder, modname, ispkg in pkgutil.iter_modules(openqml.plugins.__path__)]


def load_plugin(name, plugin_dir=None):
    """Loads an OpenQML plugin.

    Args:
      name (str): plugin filename without the .py suffix
      plugin_dir (None, str): path to the directory storing the plugin modules.
        If None, load the plugins from the openqml.plugins namespace package.
        If '', try using the OPENQML_PLUGINS environmental variable as the path.

    Returns:
      Plugin: loaded plugin class object
    """
    mod = None
    try:
        if plugin_dir is None:
            # load plugin from the namespace package (recommended)
            mod = _load_from_namespace_pkg(openqml.plugins, name)
        else:
            if plugin_dir == '':
                # try reading the plugin dir from the environment
                plugin_dir = os.environ['OPENQML_PLUGINS']

            # load the plugin by appending the python module path
            sys.path.append(os.path.abspath(plugin_dir))
            mod = importlib.import_module(name)
            # TODO moving one module path element from plugin_dir to name might be safer, or not
    except ImportError as err:
        print("Error loading the plugin '{}':".format(name), err)
        raise err

    if mod is None:
        raise ValueError('Plugin {} not found.'.format(name))
    print(mod)
    return mod.init_plugin()



class Plugin:
    """ABC for OpenQML plugins.
    """
    plugin_name = ''         #: str: official plugin name
    plugin_api_version = ''  #: str: version of OpenQML for which the plugin was made
    plugin_version = ''      #: str: version of the plugin itself

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        """String representation."""
        # defaults to the class name
        return self.__class__.__name__

    def get_gateset(self):
        """Get the supported gate set.

        Returns:
          list[GateSpec]: gate specifications for supported gates
        """
        raise NotImplementedError

    def get_capabilities(self):
        """Get the other capabilities of the plugin.

        Measurements, batching etc.

        Returns:
          dict: results
        """
        raise NotImplementedError

    def reset(self):
        """Reset the plugin.

        After the reset the plugin should be as if it was just loaded.
        """
        raise NotImplementedError

    def shutdown(self):
        """Shut down the hardware backend.

        Called to notify a hardware backend that it is no longer needed.
        """
        raise NotImplementedError

    def execute_circuit(self, circuit, params, evals=1):
        """Execute a parametrized quantum circuit with the specified parameter values.

        Args:
          circuit (Sequence[Command]):
          params    (Sequence[float]):
          evals (int): how many times should the circuit be evaluated to estimate the measurement results?
        """
        raise NotImplementedError

    def define_circuit(self, circuit):
        """Define a parametrized quantum circuit for later execution.

        Args:
          circuit (Sequence[Command]):
        """
        raise NotImplementedError

    def eval_circuit(self, params, evals=1):
        """Evaluate a pre-defined quantum circuit using the given parameter values.

        Args:
          params    (Sequence[float]):
          evals (int): how many times should the circuit be evaluated to estimate the measurement results?
        """
        raise NotImplementedError





# gate set: (name, #modes, #params, how to diffenrentiate the gate wrt. param (generator, numeric?), other info?)
