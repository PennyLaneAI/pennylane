# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Plugin system
=============

**Module name:** :mod:`openqml.plugin`

.. currentmodule:: openqml.plugin

OpenQML is designed to be extended by plugins that provide new backends for executing the quantum circuits,
either by simulation or on quantum hardware.


Terminology
-----------

By an :dfn:`OpenQML plugin` we mean an individual Python module in the openqml.plugins namespace package.
Each OpenQML plugin defines the function :func:`init_plugin` which returns the :dfn:`API class` for that plugin,
a subclass of :class:`PluginAPI`.
Each instance of the plugin API class is an independent backend for executing quantum circuits.
Multiple instances of the same class can exist simultaneously and they must not interfere with each other.

Typically each :class:`quantum node <openqml.circuit.QNode>` in the computational graph uses its own PluginAPI instance.


Functions
---------

.. autosummary::
   list_plugins
   load_plugin
   _load_from_namespace_pkg


Classes
-------

.. autosummary::
   PluginAPI


PluginAPI methods
-----------------

.. currentmodule:: openqml.plugin.PluginAPI

.. autosummary::
   gates
   observables
   templates
   capabilities
   register_circuit
   get_circuit
   reset
   shutdown
   execute_circuit
   measure

----
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
      class: loaded plugin API class
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



class PluginAPI:
    """ABC for OpenQML plugins.

    This class implements some utility methods that the child classes can either use or override as they wish.
    The other methods are simply to define the plugin API, raising NotImplementedError if called.
    The child classes *must* override them.

    Args:
      name (str): name for the plugin instance

    Keyword Args:
      n_eval (int): How many times should the circuit be evaluated (or sampled) to estimate the measurement results?
        For simulator backends, zero yields the exact result.
    """
    plugin_name = ''         #: str: official plugin name
    plugin_api_version = ''  #: str: version of OpenQML for which the plugin was made
    plugin_version = ''      #: str: version of the plugin itself
    author = ''              #: str: plugin author(s)
    _capabilities = {}       #: dict[str->*]: plugin capabilities
    _gates = {}              #: dict[str->GateSpec]: specifications for supported gates
    _observables = {}        #: dict[str->GateSpec]: specifications for supported observables
    _circuits = {}           #: dict[str->Circuit]: circuit templates associated with this API class

    def __init__(self, name='default', *, n_eval=0, **kwargs):
        self.name   = name    #: str: name of the plugin instance
        self.n_eval = n_eval  #: int: number of circuit evaluations used to estimate expectation values of observables. 0 means the exact ev is returned.

    def __repr__(self):
        """String representation."""
        return self.__module__ +'.' +self.__class__.__name__ +'\nInstance: ' +self.name

    def __str__(self):
        """Verbose string representation."""
        return self.__repr__() +'\nName: ' +self.plugin_name +'\nAPI version: ' +self.plugin_api_version\
            +'\nPlugin version: ' +self.plugin_version +'\nAuthor: ' +self.author +'\n'

    @property
    def gates(self):
        """Get the supported gate set.

        Returns:
          dict[str->GateSpec]:
        """
        return self._gates

    @property
    def observables(self):
        """Get the supported observables.

        Returns:
          dict[str->GateSpec]:
        """
        return self._observables

    @property
    def templates(self):
        """Get the predefined circuit templates.

        .. todo:: rename to circuits?

        Returns:
          dict[str->Circuit]: circuit templates
        """
        return self._circuits

    @classmethod
    def capabilities(cls):
        """Get the other capabilities of the plugin.

        Measurements, batching etc.

        Returns:
          dict[str->*]: results
        """
        return cls._capabilities

    @classmethod
    def register_circuit(cls, circuit, name=None):
        """Register a parametrized quantum circuit for later execution.

        Args:
          circuit (Circuit): quantum circuit
          name (str, None): Name given to the circuit. If None, circuit.name is used.
        """
        if name is None:
            name = circuit.name
        else:
            warnings.warn('Circuit stored under a different name.')
        if name in cls._circuits:
            warnings.warn('Stored circuit replaced.')
        cls._circuits[name] = circuit

    @classmethod
    def get_circuit(cls, name):
        """Find a register quantum circuit by name.

        Args:
          name (str): name given to the circuit
        Returns:
          Circuit: quantum circuit
        """
        if name not in cls._circuits:
            raise KeyError("Unknown circuit '{}'.".format(name))
        return cls._circuits[name]

    def reset(self):
        """Reset the backend state.

        After the reset the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        raise NotImplementedError

    def shutdown(self):
        """Shut down the hardware backend.

        Called to notify a hardware backend that it is no longer needed. The plugin instance is rendered unusable.
        """
        raise NotImplementedError

    def execute_circuit(self, circuit, params=[], *, reset=True, **kwargs):
        """Execute a parametrized quantum circuit with the specified parameter values.

        :meth:`PluginAPI.execute_circuit` mostly just checks argument validity and sets certain instance variables,
        the subclasses are expected to provide the actual functionality by overriding this method.

        Args:
          circuit (Circuit, str): circuit to execute, or the name of a predefined circuit
          params  (Sequence[float]): values of the non-fixed parameters
          reset   (bool): should the backend state be reset before the execution?

        Keyword Args:
          n_eval (int): How many times should the circuit be evaluated (or sampled) to estimate the measurement results?
            For simulator backends, zero yields the exact result.

        Returns:
          vector[float], None: If the circuit has output observable(s) defined return the expectation value(s), otherwise None.
        """
        if not isinstance(circuit, Circuit):
            # look it up by name
            try:
                circuit = self._circuits[circuit]
            except KeyError:
                raise KeyError("Unknown circuit '{}'".format(circuit))
        self.circuit = circuit  #: Circuit: quantum circuit to execute

        temp = len(params)
        if temp != circuit.n_par:
            raise ValueError('Wrong number of circuit parameters: {} given, {} required.'.format(temp, circuit.n_par))

        # keyword arguments may override initialization arguments
        self.n_eval = kwargs.get('n_eval', self.n_eval)

        log.debug('Executing {}'.format(str(circuit)))
        if reset:
            self.reset()

    def measure(self, A, reg, par=[], n_eval=0):
        """Measure the expectation value of an observable in the current state of the circuit.

        Args:
          A  (GateSpec): Hermitian observable
          reg (int): target subsystem
          par (Sequence[float]): parameters for the observable
          n_eval (int): If zero return the exact expectation value,
            otherwise estimate it by averaging n_eval measurements.

        Returns:
          float: (estimated) expectation value
        """
        raise NotImplementedError
