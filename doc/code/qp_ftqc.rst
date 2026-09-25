qp.ftqc
========

.. currentmodule:: pennylane.ftqc

.. warning::

    This module is currently experimental and will not maintain API stability between releases.

.. automodapi:: pennylane.ftqc
    :no-heading:
    :include-all-objects:

Overview
--------

Fault-tolerant gadgets
^^^^^^^^^^^^^^^^^^^^^^

The :doc:`qp.ftqc.gadget </code/qp_ftqc_gadget>` submodule contains functionality for writing
fault-tolerant gadgets on CSS codes, checking them, and compiling them with Catalyst.

Pauli Tracker
^^^^^^^^^^^^^

This module contains functions for tracking, commuting Pauli operations in a Clifford circuit as well as getting measurement corrections.

.. currentmodule:: pennylane.ftqc

.. autosummary::
    :toctree: api

    ~pauli_to_xz
    ~xz_to_pauli
    ~pauli_prod
    ~commute_clifford_op
    ~get_byproduct_corrections
