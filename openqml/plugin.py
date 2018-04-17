# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
ABC for plugins
===============

**Module name:** :mod:`openqml.plugin`

.. currentmodule:: openqml.plugin


Functions
---------

.. autosummary::
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
    if plugin_dir is None:
        # load plugin from the namespace package (recommended)
        mod = _load_from_namespace_pkg(openqml.plugins, name)
    else:
        if plugin_dir == '':
            # try reading the plugin dir from the environment
            plugin_dir = os.environ['OPENQML_PLUGINS']

        # load the plugin by append the python module path
        sys.path.append(os.path.abspath(plugin_dir))
        mod = importlib.import_module(name)
        # TODO moving one module path element from plugin_dir to name might be safer, or not

    if mod is None:
        raise ValueError('Plugin {} not found.'.format(name))
    print(mod)
    return mod.init_plugin()



class Plugin:
    """ABC for OpenQML plugins.
    """
    def __init__(self):
        self.a = set()  #: set[]: sdfdsf

    def __str__(self):
        """String representation."""
        # defaults to the class name
        return self.__class__.__name__

    def func(self, reg):
        """A function.
        """
        raise NotImplementedError('Not implemented: {}.func'.format(self))
