"""
Submodule for performing mixed-qutrit-based simulations of quantum circuits.

This submodule is internal and subject to change without a deprecation cycle. Use
at your own discretion.

.. currentmodule:: pennylane.devices.qutrit
.. autosummary::
    :toctree: api

    create_initial_state
    apply_operation
"""

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
