qml.noise
=========

This module contains the functionality for building and manipulating insertion-based noise models,
where noisy gates and channels are inserted based on the target operations.

Overview
--------

Insertion-based noise models in PennyLane are defined via a mapping from conditionals, specified
as :class:`~.BooleanFn` objects, to :ref:`quantum function <intro_vcirc_qfunc>`-like callables
that contain the noisy operations to be applied, but without any return statements. Additional
noise-related metadata can also be supplied to construct a noise model using:

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~NoiseModel

Each conditional in the ``model_map`` evaluates the gate operations in the quantum circuit based on
some condition of its attributes (e.g., type, parameters, wires, etc.) and use the corresponding
callable to apply the noise operations, using the user-provided metadata (e.g., hardware topologies
or relaxation times), whenever the condition results true.

.. _intro_boolean_fn:

Boolean functions
^^^^^^^^^^^^^^^^^

Each :class:`~.BooleanFn` in the noise model is evaluated on the operations of a given
quantum circuit. One can construct standard Boolean functions using the following helpers:

.. currentmodule:: pennylane.noise

.. autosummary::
    :toctree: api

    ~op_eq
    ~op_in
    ~wires_eq
    ~wires_in

For example, a Boolean function that checks of an operation is on wire ``0`` can be created
like so:

>>> fn = wires_eq(0)
>>> op1, op2 = qml.PauliX(0), qml.PauliX(1)
>>> fn(op1)
False
>>> fn(op2)
True

Arbitrary Boolean functions can also be defined by wrapping the functional form
of custom conditions with the following decorator:

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api
 
    ~BooleanFn

For example, a Boolean function can be created to identify an :class:`~.RX` gate
with a maximum parameter value:

.. code-block:: python

    @qml.BooleanFn
    def rx_condition(op, **metadata):
        return isinstance(op, qml.RX) and op.parameters[0] < 1.0

Boolean functions can be combined using standard bit-wise operators, such as
``&``, ``|``, ``^``, or ``~``. The result will be another Boolean function. It
is important to note that as Python will evaluate the expression in the order
of their combination, i.e., left to right, the order of composition could matter,
even though bitwise operations are symmetric by definition.

Noisy quantum functions
^^^^^^^^^^^^^^^^^^^^^^^

If a Boolean function evaluates to ``True`` on a given operation in the quantum circuit,
the corresponding quantum function is evaluated that inserts the noise directly after
the operation. The quantum function should have signature ``fn(op, **metadata)``,
allowing for dependency on both the preceding operation and metadata specified in
the noise model. For example, the following noise model adds an over-rotation
to the ``RX`` gate:

.. code-block:: python

    def noisy_rx(op, **metadata):
        qml.RX(op.parameters[0] * 0.05, op.wires)

    noise_model = qml.NoiseModel({rx_condition: noisy_rx})

A common use case is to have a single-operation :ref:`noise channel <intro_ref_ops_channels>`
whose wires are the same as the preceding operation. This can be constructed using:

.. currentmodule:: pennylane.noise

.. autosummary::
    :toctree: api

    ~partial_wires

For example, a constant-valued over-rotation can be created using:

>>> rx_constant = qml.noise.partial_wires(qml.RX(0.1, wires=[0]))
>>> rx_constant(2)
RX(0.1, 2)
>>> qml.NoiseModel({rx_condition: rx_constant})
NoiseModel({
    BooleanFn(rx_condition): RX(phi=0.1)
})

Example noise model
^^^^^^^^^^^^^^^^^^^

The following example shows how to set up an artificial noise model in PennyLane:

.. code-block:: python

  # Set up the conditions
  c0 = qml.noise.op_eq(qml.PauliX) | qml.noise.op_eq(qml.PauliY)
  c1 = qml.noise.op_eq(qml.Hadamard) & qml.noise.wires_in([0, 1])
  c2 = qml.noise.op_eq(qml.RX)

  @qml.BooleanFn
  def c3(op, **metadata):
      return isinstance(op, qml.RY) and op.parameters[0] >= 0.5

  # Set up noisy ops
  n0 = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

  def n1(op, **metadata):
      ThermalRelaxationError(0.4, metadata["t1"], 0.2, 0.6, op.wires)

  def n2(op, **metadata):
      qml.RX(op.parameters[0] * 0.05, op.wires)

  n3 = qml.noise.partial_wires(qml.PhaseDamping, 0.9)
  
  # Set up noise model
  noise_model = qml.NoiseModel({c0: n0, c1: n1, c2: n2}, t1=0.04)
  noise_model += {c3: n3}  # One-at-a-time construction

>>> noise_model
NoiseModel({
    OpEq(PauliX) | OpEq(PauliY): AmplitudeDamping(gamma=0.4)
    OpEq(Hadamard) & WiresIn([0, 1]): n1
    OpEq(RX): n2
    BooleanFn(c3): PhaseDamping(gamma=0.9)
}, t1 = 0.04)

API overview
^^^^^^^^^^^^

The following are the ``BooleanFn`` objects created by calling the helper functions
above, such as :func:`~.op_eq`. These objects do not need to be instantiated directly

.. currentmodule:: pennylane.noise.conditionals

.. autosummary::
    :toctree: api

    ~OpEq
    ~OpIn
    ~WiresEq
    ~WiresIn

Bitwise operations like `And` and `Or` are represented with the following classes in the
:mod:`boolean_fn` module:

.. currentmodule:: pennylane.boolean_fn

.. autosummary::
    :toctree: api

    ~And
    ~Or
    ~Xor
    ~Not

Class Inheritence Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: pennylane.noise.conditionals
    :parts: 1

.. inheritance-diagram:: pennylane.boolean_fn
    :parts: 1
