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
import logging as log
import pkgutil
import warnings

import openqml
from .circuit import Circuit
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
                temp = ns.__name__ +'.' +modname
                mod = importlib.import_module(temp)
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
    #print(mod)
    print(mod.__file__)
    p = mod.init_plugin()
    temp = openqml.version()
    if p.plugin_api_version != temp:
        warnings.warn('Plugin API version {} does not match OpenQML version {}.'.format(p.plugin_api_version, temp))
    return p



class Plugin:
    """ABC for OpenQML plugins.

    This class implements some utility methods that the child classes can either use or override as they wish.
    The other methods are simply to define the plugin API, raising NotImplementedError if called.
    The child classes *must* override them.
    """
    plugin_name = ''         #: str: official plugin name
    plugin_api_version = ''  #: str: version of OpenQML for which the plugin was made
    plugin_version = ''      #: str: version of the plugin itself
    author = ''              #: str: plugin author(s)

    def __init__(self):
        self._circuits = {}  #: dict[str->Circuit]: known circuit templates

    def __repr__(self):
        """String representation."""
        return self.__class__.__name__

    def __str__(self):
        """Verbose string representation."""
        return self.__repr__() +'\nName: ' +self.plugin_name +'\nAPI version: ' +self.plugin_api_version\
            +'\nPlugin version: ' +self.plugin_version +'\nAuthor: ' +self.author +'\n'

    def get_gateset(self):
        """Get the supported gate set.

        Returns:
          list[GateSpec]: gate specifications for supported gates
        """
        raise NotImplementedError

    def get_templates(self):
        """Get the predefined circuit templates.

        Returns:
          list[Circuit]: circuit templates
        """
        return list(self._circuits.values())

    def get_capabilities(self):
        """Get the other capabilities of the plugin.

        Measurements, batching etc.

        Returns:
          dict[str->*]: results
        """
        raise NotImplementedError

    def reset(self):
        """Reset the plugin.

        After the reset the plugin should be as if it was just loaded.
        Most importantly the quantum state is reset to its initial value.
        """
        raise NotImplementedError

    def shutdown(self):
        """Shut down the hardware backend.

        Called to notify a hardware backend that it is no longer needed.
        """
        raise NotImplementedError

    def _execute_circuit(self, circuit, params=[], **kwargs):
        raise NotImplementedError

    def execute_circuit(self, circuit, params=[], **kwargs):
        """Execute a parametrized quantum circuit with the specified parameter values.

        Note: The state of the backend is not automatically reset.

        This function is a thin wrapper around :meth:`~Plugin._execute_circuit`
        that mostly checks argument validity.

        Args:
          circuit (Circuit, str): circuit to execute, or the name of a predefined circuit
          params  (Sequence[float]): values of the non-fixed parameters

        Keyword Args:
          evals (int): how many times should the circuit be evaluated to estimate the measurement results?
        """
        if not isinstance(circuit, Circuit):
            # look it up by name
            try:
                circuit = self._circuits[circuit]
            except KeyError:
                raise KeyError("Unknown circuit '{}'".format(circuit))

        temp = len(params)
        if temp != circuit.n_par:
            raise ValueError('Wrong number of circuit parameters: {} given, {} required.'.format(temp, circuit.n_par))

        log.info('Executing {}'.format(str(circuit)))
        return self._execute_circuit(circuit, params, **kwargs)

    def define_circuit(self, circuit, name=None):
        """Define a parametrized quantum circuit for later execution.

        Args:
          circuit (Circuit): quantum circuit
          name (str, None): Name given to the circuit. If None, circuit.name is used.
        """
        if name is None:
            name = circuit.name
        else:
            warnings.warn('Circuit stored under a different name.')
        if name in self._circuits:
            warnings.warn('Stored circuit replaced.')
        self._circuits[name] = circuit

    def measure(self, A, reg, n_eval=None):
        """Measure the expectation value of an observable.

        Args:
          A  (Gate): Hermitian observable
          reg (int): target subsystem
          n_eval (int, None): If None return the exact expectation value,
            otherwise estimate it by averaging n_eval measurements.
            Returned variance is always exact.

        Returns:
          (float, float): expectation value, variance
        """
        raise NotImplementedError
