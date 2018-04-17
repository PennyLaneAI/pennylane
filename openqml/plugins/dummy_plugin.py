# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Dummy implementation for a plugin
=================================

**Module name:** :mod:`openqml.dummy_plugin`

.. currentmodule:: openqml.dummy_plugin


"""

from openqml.plugin import Plugin


class DummyPlugin(Plugin):
    def __init__(self):
        super().__init__()
        print('Dummy plugin instance created.')

    def func(self, reg):
        print('Dummy plugin func called.')



def init_plugin():
    """Every plugin must define this function.

    It should perform whatever initializations are necessary, and then return the API class object.

    Returns:
      Plugin: plugin class object
    """
    return DummyPlugin
