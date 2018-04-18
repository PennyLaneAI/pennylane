# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
Strawberry Fields plugin for OpenQML
====================================

**Module name:** :mod:`openqml.sf_plugin`

.. currentmodule:: openqml.sf_plugin

This plugin provides the interface between OpenQML and Strawberry Fields.
It enables OpenQML to optimize continuous variable quantum circuits.
"""

from openqml.plugin import Plugin

import strawberryfields as sf
from strawberryfields.ops import *



class SFPlugin(Plugin):
    def __init__(self):
        super().__init__()
        print('Strawberry Fields plugin instance created.')



def init_plugin():
    """Every plugin must define this function.

    It should perform whatever initializations are necessary, and then return the API class object.

    Returns:
      Plugin: plugin class object
    """
    return SFPlugin
