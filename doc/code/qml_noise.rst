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
condition of its attributes (e.g., type, parameters, wires, etc.) and use the corresponding
``Callable`` to apply the noise operations, using the user-provided metadata (e.g., hardware
topologies or relaxation times), whenever the condition results true.

Noise Model
^^^^^^^^^^^

.. currentmodule:: pennylane.noise.noise_model

.. autosummary::
    :toctree: api

    ~NoiseModel

Conditional Constructors
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.noise.conditionals

.. autosummary::
    :toctree: api

    ~op_eq
    ~op_in
    ~wires_eq
    ~wires_in

Arbitrary conditionals can also be defined by wrapping the functional form
of custom conditions with the following decorator:

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api
 
    ~BooleanFn

.. note::

    Conditionals built via these constructors or decorator can be combined using
    standard bit-wise operators, such as ``&``, ``|``, ``^``, or ``~``. The resulting
    combination will still behave like a single conditional and store the individual
    components in the ``operands`` attribute. As Python will evaluate the expression
    in the same order, i.e., left to right, the order of composition could matter,
    even though bitwise operations are symmetric by definition.

Callable Constructor
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    ~partial_wires

.. note::

    The signature of any user-defined callable must be -
    ``callable(op: Operation, **kwargs) -> None``, i.e., it accepts
    an operation and some metadata-based keyword arguments. It should
    then let one queue the noisy gates corresponding to that operation
    similar to a quantum function but without returning any measurements.

Conditional Classes
^^^^^^^^^^^^^^^^^^^

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
