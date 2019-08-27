.. _operations:

Quantum operations
==================

**Module name:** :mod:`pennylane.ops`

.. currentmodule:: pennylane.ops

This module contains core quantum operations supported by PennyLane - such as gates, state preparations and observables.

PennyLane supports a collection of built-in quantum operations,
including both discrete-variable (DV) operations as used in the qubit model,
and continuous-variable (CV) operations as used in the qumode model of quantum
computation.

Here, we summarize the built-in operations and observables supported by PennyLane,
as well as the conventions chosen for their implementation.

.. note::

    When writing a plugin device for PennyLane, make sure that your plugin
    supports as many of the PennyLane built-in operations defined here as possible.

    If the convention differs between the built-in PennyLane operation
    and the corresponding operation in the targeted framework, ensure that the
    conversion between the two conventions takes places automatically
    by the plugin device.

.. raw:: html

    <style>
    div.topic.contents > ul {
        max-height: 100px;
    }
    </style>

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    ops/qubit
    ops/cv


Shared operations
------------------

These operations can be used on both qubit and CV devices:

.. automodule:: pennylane.ops
	:members: Identity