# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Strawberry Fields plugin for OpenQML
====================================

**Module name:** :mod:`openqml.plugins.strawberryfields`

.. currentmodule:: openqml.plugins.strawberryfields

This plugin provides the interface between OpenQML and Strawberry Fields.
It enables OpenQML to optimize continuous variable quantum circuits.
"""

from openqml.plugin import Plugin

#import strawberryfields as sf
#from strawberryfields.ops import *



class SFPlugin(Plugin):
    plugin_name = 'Strawberry Fields OpenQML plugin'
    plugin_api_version = '0.1.0'
    plugin_version = '0' #sf.version()
    author = 'Xanadu Inc.'

    def __init__(self):
        super().__init__()

    def get_gateset(self):
        return []


def init_plugin():
    """Every plugin must define this function.

    It should perform whatever initializations are necessary, and then return an API class instance.

    Returns:
      Plugin: plugin API class instance
    """
    return SFPlugin()
