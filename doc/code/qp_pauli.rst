qp.pauli
============

Overview
--------

.. currentmodule:: pennylane.pauli

This module defines functions and classes for generating and manipulating
elements of the Pauli group. It also contains a subpackage :mod:`pauli/grouping`
for Pauli-word partitioning functionality used in measurement optimization.

.. currentmodule:: pennylane.pauli

.. automodapi:: pennylane.pauli
    :no-heading:
    :no-main-docstr:
    :no-inherited-members:
    :skip:

PauliWord and PauliSentence
---------------------------

The single-qubit Pauli group consists of the four single-qubit Pauli operations
:class:`~pennylane.Identity`, :class:`~pennylane.PauliX`,
:class:`~pennylane.PauliY` , and :class:`~pennylane.PauliZ`. The :math:`n`-qubit
Pauli group is constructed by taking all possible :math:`N`-fold tensor products
of these elements. Elements of the :math:`n`-qubit Pauli group are known as
Pauli words, and have the form :math:`P_J = \otimes_{i=1}^{n}\sigma_i^{(J)}`,
where :math:`\sigma_i^{(J)}` is one of the Pauli operators
(:class:`~pennylane.PauliX`, :class:`~pennylane.PauliY`,
:class:`~pennylane.PauliZ`) or identity (:class:`~pennylane.Identity`) acting on
the :math:`i^{th}` qubit. The full :math:`n`-qubit Pauli group has size
:math:`4^n` (neglecting the four possible global phases that may arise from
multiplication of its elements).

:class:`~pennylane.pauli.PauliWord` is a lightweight class which uses a dictionary
approach to represent Pauli words. A :class:`~pennylane.pauli.PauliWord` can be
instantiated by passing a dictionary of wires and their associated Pauli operators.

>>> from pennylane.pauli import PauliWord
>>> pw1 = PauliWord({0:"X", 1:"Z"})  # X@Z
>>> pw2 = PauliWord({0:"Y", 1:"Z"})  # Y@Z
>>> pw1, pw2
(X(0) @ Z(1), Y(0) @ Z(1))

The purpose of this class is to efficiently compute products of Pauli words and
obtain the matrix representation.

>>> pw1 @ pw2
1j * Z(0)
>>> pw1.to_mat(wire_order=[0, 1])
array([[ 0,  0,  1,  0],
       [ 0,  0,  0, -1],
       [ 1,  0,  0,  0],
       [ 0, -1,  0,  0]])

The :class:`~pennylane.pauli.PauliSentence` class represents linear combinations of
Pauli words. Using a similar dictionary based approach we can efficiently add, multiply
and extract the matrix of operators in this representation.

>>> ps1 = PauliSentence({pw1: 1.2, pw2: 0.5j})
>>> ps2 = PauliSentence({pw1: -1.2})
>>> ps1
1.2 * X(0) @ Z(1)
+ 0.5j * Y(0) @ Z(1)

>>> ps1 + ps2
0.0 * X(0) @ Z(1)
+ 0.5j * Y(0) @ Z(1)

>>> ps1 @ ps2
-1.44 * I
+ (-0.6+0j) * Z(0)

>>> (ps1 + ps2).to_mat(wire_order=[0, 1])
array([[ 0. +0.j,  0. +0.j,  0.5+0.j,  0. +0.j],
       [ 0. +0.j,  0. +0.j,  0. +0.j, -0.5+0.j],
       [-0.5+0.j,  0. +0.j,  0. +0.j,  0. +0.j],
       [ 0. +0.j,  0.5+0.j,  0. +0.j,  0. +0.j]])

We can intuitively use Pauli arithmetic to construct Hamiltonians consisting of :class:`~pennylane.pauli.PauliWord`
and :class:`~pennylane.pauli.PauliSentence` objects like the spin-1/2 XXZ model Hamiltonian,

.. math:: H_\text{XXZ} = \sum_j [J^\bot (X_j X_{j+1} + Y_j Y_{j+1}) + J^\text{ZZ} Z_j Z_{j+1} + h Z_j].

Here we look at the simple topology of a one-dimensional chain with periodic boundary conditions
(i.e. qubit number :math:`n \equiv 0` for pythonic numbering of wires, e.g. ``[0, 1, 2, 3]`` for ``n=4``).
In code we can do this via the following example with 4 qubits.

.. code-block:: python3

    n = 4
    J_orthogonal = 1.5
    ops = [
        J_orthogonal * (PauliWord({i:"X", (i+1)%n:"X"}) + PauliWord({i:"Y", (i+1)%n:"Y"}))
        for i in range(n)
    ]

    J_zz = 0.5
    ops += [J_zz * PauliWord({i:"Z", (i+1)%n:"Z"}) for i in range(n)]

    h = 2.
    ops += [h * PauliWord({i:"Z"}) for i in range(n)]

    H = sum(ops)

We can also displace the Hamiltonian by an arbitrary amount. Here, for example, such that the ground state energy is 0.

>>> H = H - np.min(np.linalg.eigvalsh(H.to_mat()))

.. _graph_colouring:

Graph colouring
---------------

.. automodapi:: pennylane.pauli.grouping.graph_colouring
    :no-heading:
    :no-inherited-members:


Grouping observables
--------------------

Pauli words can be used for expressing a qubit :class:`~pennylane.Hamiltonian`.
A qubit Hamiltonian has the form :math:`H_{q} = \sum_{J} C_J P_J` where
:math:`C_{J}` are numerical coefficients, and :math:`P_J` are Pauli words.

A list of Pauli words can be partitioned according to certain grouping
strategies. As an example, the :func:`~.group_observables` function partitions
a list of observables (Pauli operations and tensor products thereof) into
groupings according to a binary relation (e.g., qubit-wise commuting):

>>> observables = [qp.PauliY(0), qp.PauliX(0) @ qp.PauliX(1), qp.PauliZ(1)]
>>> obs_groupings = group_observables(observables)
>>> obs_groupings
[[PauliX(wires=[0]) @ PauliX(wires=[1])],
 [PauliY(wires=[0]), PauliZ(wires=[1])]]

The :math:`C_{J}` coefficients for each :math:`P_J` Pauli word making up a
Hamiltonian can also be specified along with further options, such as the
Pauli-word grouping method (e.g., *qubit-wise commuting*) and the underlying
graph-colouring algorithm (e.g., *recursive largest first*) used for creating
the groups of observables:

>>> obs = [qp.PauliY(0), qp.PauliX(0) @ qp.PauliX(1), qp.PauliZ(1)]
>>> coeffs = [1.43, 4.21, 0.97]
>>> obs_groupings, coeffs_groupings = group_observables(obs, coeffs, 'qwc', 'rlf')
>>> obs_groupings
[[PauliX(wires=[0]) @ PauliX(wires=[1])],
 [PauliY(wires=[0]), PauliZ(wires=[1])]]
>>> coeffs_groupings
[[4.21], [1.43, 0.97]]

For a larger example of how grouping can be used with PennyLane, check out the
`Measurement Optimization demo <https://pennylane.ai/qml/demos/tutorial_measurement_optimize/>`_.
