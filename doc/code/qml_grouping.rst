qml.grouping
============

This subpackage defines functions and classes for Pauli-word partitioning
functionality used in measurement optimization.

A Pauli word is defined as :math:`P_J = \prod_{i=1}^{N}\sigma_i^{(J)}`, where
:math:`\sigma_i^{(J)}` is one of the Pauli operators (:class:`~pennylane.PauliX`,
:class:`~pennylane.PauliY`, :class:`~pennylane.PauliZ`) or identity
(:class:`~pennylane.Identity`) acting on the :math:`i^{th}` qubit.

Pauli words can be used for expressing a qubit :class:`~pennylane.Hamiltonian`.
A qubit Hamiltonian has the form :math:`H_{q} = \sum_{J} C_J P_J` where
:math:`C_{J}` are numerical coefficients, and :math:`P_J` are Pauli words.

A list of Pauli words can be partitioned according to certain grouping
strategies. As an example, the :func:`~.group_observables` function partitions
a list of observables (Pauli operations and tensor products thereof) into
groupings according to a binary relation (e.g., qubit-wise commuting):

>>> observables = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
>>> obs_groupings = group_observables(observables)
>>> obs_groupings
[[Tensor(PauliX(wires=[0]), PauliX(wires=[1]))],
 [PauliY(wires=[0]), PauliZ(wires=[1])]]

The :math:`C_{J}` coefficients for each :math:`P_J` Pauli word making up a
Hamiltonian can also be specified along with further options, such as the
Pauli-word grouping method (e.g., *qubit-wise commuting*) and the underlying
graph-colouring algorithm (e.g., *recursive largest first*) used for creating
the groups of observables:

>>> obs = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
>>> coeffs = [1.43, 4.21, 0.97]
>>> obs_groupings, coeffs_groupings = group_observables(obs, coeffs, 'qwc', 'rlf')
>>> obs_groupings
[[Tensor(PauliX(wires=[0]), PauliX(wires=[1]))],
 [PauliY(wires=[0]), PauliZ(wires=[1])]]
>>> coeffs_groupings
[[4.21], [1.43, 0.97]]

.. currentmodule:: pennylane.grouping


.. automodapi:: pennylane.grouping
    :no-inheritance-diagram:
    :no-inherited-members:


.. automodapi:: pennylane.grouping.graph_colouring
    :no-inheritance-diagram:
    :no-inherited-members:
