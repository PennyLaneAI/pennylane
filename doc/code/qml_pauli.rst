qml.pauli
============

Overview
--------

This module defines functions and classes for generating and manipulating
elements of the Pauli group. It also contains a subpackage :mod:`pauli/grouping`
for Pauli-word partitioning functionality used in measurement optimization.

.. currentmodule:: pennylane.pauli

.. automodapi:: pennylane.pauli
    :no-heading:
    :no-main-docstr:
    :no-inheritance-diagram:
    :no-inherited-members:

Graph colouring
---------------

.. automodapi:: pennylane.pauli.grouping.graph_colouring
    :no-heading:
    :no-inheritance-diagram:
    :no-inherited-members:


Pauli group
-----------

The single-qubit Pauli group consists of the four single-qubit Pauli operations
:class:`~pennylane.Identity`, :class:`~pennylane.PauliX`,
:class:`~pennylane.PauliY` , and :class:`~pennylane.PauliZ`. The :math:`n`-qubit
Pauli group is constructed by taking all possible :math:`N`-fold tensor products
of these elements. Elements of the :math:`n`-qubit Pauli group are often known
as Pauli words, and have the form :math:`P_J = \otimes_{i=1}^{n}\sigma_i^{(J)}`,
where :math:`\sigma_i^{(J)}` is one of the Pauli operators
(:class:`~pennylane.PauliX`, :class:`~pennylane.PauliY`,
:class:`~pennylane.PauliZ`) or identity (:class:`~pennylane.Identity`) acting on
the :math:`i^{th}` qubit. The full :math:`n`-qubit Pauli group has size
:math:`4^n` (neglecting the four possible global phases that may arise from
multiplication of its elements).

The Pauli group can be constructed using the :func:`~.pauli_group`
function. To construct the group, it is recommended to provide a wire map in
order to indicate the names and indices of the wires. (If none is provided, a
default mapping of integers will be used.)

>>> from pennylane.pauli import pauli_group
>>> pg_3 = list(pauli_group(3))

Multiplication of Pauli group elements can be performed using
:func:`~.pauli_mult` or
:func:`~.pauli_mult_with_phase`:

>>> from pennylane.pauli import pauli_mult
>>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
>>> pg = list(pauli_group(3, wire_map=wire_map))
>>> pg[3]
PauliZ(wires=['b']) @ PauliZ(wires=['c'])
>>> pg[55]
PauliY(wires=['a']) @ PauliY(wires=['b']) @ PauliZ(wires=['c'])
>>> pauli_mult(pg[3], pg[55], wire_map=wire_map)
PauliY(wires=['a']) @ PauliX(wires=['b'])

Pauli observables can be converted to strings (and vice versa):

>>> from pennylane.pauli import pauli_word_to_string, string_to_pauli_word
>>> pauli_word_to_string(pg[55], wire_map=wire_map)
'YYZ'
>>> string_to_pauli_word('ZXY', wire_map=wire_map)
PauliZ(wires=['a']) @ PauliX(wires=['b']) @ PauliY(wires=['c'])

The matrix representation for arbitrary Paulis and wire maps can also be performed.

>>> pennylane.pauli import pauli_word_to_matrix
>>> wire_map = {'a' : 0, 'b' : 1}
>>> pauli_word = qml.PauliZ('b')  # corresponds to Pauli 'IZ'
>>> pauli_word_to_matrix(pauli_word, wire_map=wire_map)
array([[ 1.,  0.,  0.,  0.],
       [ 0., -1.,  0., -0.],
       [ 0.,  0.,  1.,  0.],
       [ 0., -0.,  0., -1.]])

Grouping observables
--------------------

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
