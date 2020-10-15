qml.grouping
============

This subpackage defines functions and classes for Pauli word partitioning
functionality used in measurement optimization.

A Pauli word is defined as :math:`P_I = \prod_{i=1}^{N}\sigma_i^{(I)}` where
:math:`\sigma_i^{(I)}` is one of the Pauli operators (:class:`~pennylane.PauliX`,
:class:`~pennylane.PauliY`, :class:`~pennylane.PauliZ`) or identity
(:class:`~pennylane.Identity`) for the :math:`i^{th}` qubit.

Pauli words can be used for expressing a qubit :class:`~pennylane.Hamiltonian`.
A qubit Hamiltonian has the form :math:`H_{q} = \sum_{I} C_I P_I` where
:math:`C_{I}` are numerical coefficients, and :math:`P_I` are Pauli words.

A list of Pauli words can be partitioned according to certain grouping
strategies. As an example, the :func:`~.group_observables` function partitions
a list of observables (Pauli operations and tensor products thereof) into
groupings according to a binary relation (e.g. qubit-wise commuting):

.. code-block:: python

    >>> obs = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
    >>> obs_groupings = group_observables(obs)
    >>> obs_groupings
    [[Tensor(PauliX(wires=[0]), PauliX(wires=[1]))],
     [PauliY(wires=[0]), PauliZ(wires=[1])]]

The :math:`C_{I}` coefficients for each :math:`P_I` Pauli word making up a
Hamiltonian can also be specified along with further options such as the Pauli
word grouping method (``qubit-wise commutativity``) and the underlying graph
coloring algorithm (``recursive largest first``) used for creating the groups
of observables:

.. code-block:: python

    >>> obs = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
    >>> coeffs = [1.43, 4.21, 0.97]
    >>> obs_groupings, coeffs_groupings = group_observables(obs, coeffs, 'qwc', 'rlf')
    >>> obs_groupings
    [[Tensor(PauliX(wires=[0]), PauliX(wires=[1]))],
     [PauliY(wires=[0]), PauliZ(wires=[1])]]
    >>> coeffs_groupings
    [[4.21], [1.43, 0.97]]

.. currentmodule:: pennylane.grouping

Graph colouring
---------------

.. automodapi:: pennylane.grouping.graph_colouring
    :no-heading:
    :no-inheritance-diagram:
    :no-inherited-members:

Group observables
-----------------

.. automodapi:: pennylane.grouping.group_observables
    :no-heading:
    :no-inheritance-diagram:
    :no-inherited-members:
    :skip: are_identical_pauli_words, binary_to_pauli, observables_to_binary_matrix, qwc_complement_adj_matrix, largest_first, recursive_largest_first, Wires

Optimize measurements
---------------------

.. automodapi:: pennylane.grouping.optimize_measurements
    :no-heading:
    :no-inheritance-diagram:
    :no-inherited-members:
    :skip: group_observables, diagonalize_qwc_groupings

Transformations
---------------

.. automodapi:: pennylane.grouping.transformations
    :no-heading:
    :no-inheritance-diagram:
    :no-inherited-members:
    :skip: are_identical_pauli_words, is_pauli_word, is_qwc, pauli_to_binary, template

Utils
-----

.. automodapi:: pennylane.grouping.utils
    :no-heading:
    :no-inheritance-diagram:
    :no-inherited-members:
