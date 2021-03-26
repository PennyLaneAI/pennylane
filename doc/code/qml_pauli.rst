qml.pauli
=========

This subpackage defines functions and classes for generating and manipulating
elements of the Pauli group. It also contains Pauli-word partitioning
functionality used in measurement optimization.

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

.. currentmodule:: pennylane.pauli

.. automodapi:: pennylane.pauli.pauli
    :no-inheritance-diagram:
    :no-inherited-members:
    :skip: binary_to_pauli, pauli_to_binary, are_identical_pauli_words

.. automodapi:: pennylane.pauli.pauli_utils
    :no-inheritance-diagram:
    :no-inherited-members:
    :skip: reduce

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

.. currentmodule:: pennylane.pauli

.. automodapi:: pennylane.pauli.graph_colouring
    :no-inheritance-diagram:
    :no-inherited-members:

.. automodapi:: pennylane.pauli.group_observables
    :no-inheritance-diagram:
    :no-inherited-members:
    :skip: binary_to_pauli, are_identical_pauli_words

.. automodapi:: pennylane.pauli.grouping_utils
    :no-inheritance-diagram:
    :no-inherited-members:
    :skip: pauli_to_binary

.. automodapi:: pennylane.pauli.optimize_measurements
    :no-inheritance-diagram:
    :no-inherited-members:
    :skip: diagonalize_qwc_groupings

.. automodapi:: pennylane.pauli.transformations
    :no-inheritance-diagram:
    :no-inherited-members:
    :skip: are_identical_pauli_words, is_pauli_word, is_qwc, pauli_to_binary, template
