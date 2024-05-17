qml.noise
=========

This module contains the functionality for building and manipulating noise models.

Overview
--------

In PennyLane, noise models are defined via a mapping of conditions defined as ``Conditionals``
to functions specified as ``Callables``, along with some additional noise-related metadata.

::

    NoiseModel: ({Conditionals --> Callables},  metadata)

Each ``Conditional`` evaluates the gate operations in the quantum circuit based on some
condition on its attributes (e.g., type, parameters, wires, etc.) and use the corresponding
``Callable`` to queue the noise operations using the user-provided metadata (e.g., hardware
topologies or relaxation times) whenever the condition results true.

.. currentmodule:: pennylane.noise

Conditional Constructors
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.noise.conditionals

.. autosummary::
    :toctree: api

    ~op_eq
    ~op_in
    ~wires_eq
    ~wires_in

Callable Constructor
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.noise.conditionals

.. autosummary::
    :toctree: api

    ~partial_wires

Conditional Classes
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.noise.conditionals

.. autosummary::
    :toctree: api

    ~OpEq
    ~OpIn
    ~WiresEq
    ~WiresIn

Class Inheritence Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: pennylane.noise.conditionals
    :parts: 1
