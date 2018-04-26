# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Dummy implementation for a plugin
=================================

**Module name:** :mod:`openqml.dummy_plugin`

.. currentmodule:: openqml.dummy_plugin


"""

from openqml.plugin import Plugin


class DummyPlugin(Plugin):
    plugin_name = 'dummy plugin'
    plugin_api_version = '0.0.0'
    plugin_version = '1.0.0'

    def __init__(self):
        super().__init__()
        print('Dummy plugin instance created.')



def init_plugin():
    """Every plugin must define this function.

    It should perform whatever initializations are necessary, and then return the API class object.

    Returns:
      Plugin: plugin class object
    """
    return DummyPlugin
