qml.operation
=============

.. currentmodule:: pennylane.operation

.. warning::

    Unless you are a PennyLane or plugin developer, you likely do not need
    to use these classes directly.

    See the :doc:`main operations page <../introduction/operations>` for
    details on available operations and observables.

.. automodapi:: pennylane.operation
    :no-heading:
    :include-all-objects:
    :skip: IntEnum, ClassPropertyDescriptor, multi_dot, pauli_eigs, Wires, eye, kron, coo_matrix


Operation attributes
^^^^^^^^^^^^^^^^^^^^

PennyLane contains a mechanism for storing lists of operations with similar
attributes and behaviour (for example, those that are their own inverses).
The attributes below are already included, and are used primarily for the
purpose of compilation transforms. New attributes can be added by instantiating
new :class:`~pennylane.ops.qubit.attributes.Attribute` objects.

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~ops.qubit.attributes.Attribute
    ~ops.qubit.attributes.composable_rotations
    ~ops.qubit.attributes.diagonal_in_z_basis
    ~ops.qubit.attributes.has_unitary_generator
    ~ops.qubit.attributes.self_inverses
    ~ops.qubit.attributes.supports_broadcasting
    ~ops.qubit.attributes.symmetric_over_all_wires
    ~ops.qubit.attributes.symmetric_over_control_wires
