qml.pauli
============

Overview
--------

This module contains functions and classes for creating and manipulating fermionic operators.

.. currentmodule:: pennylane.fermi

.. automodapi:: pennylane.fermi
    :no-heading:
    :no-main-docstr:
    :no-inherited-members:

FermiC and FermiA
-----------------

The fermionic creation and annihilation operators are constructed with :class:`~pennylane.FermiC`
and :class:`~pennylane.FermiA`, respectively, by passing the index of the orbital that the fermionic
operator acts on. For instance, the operators :math:`a^{\dagger}_0` :math:`a_3` are constructed as

>>> qml.FermiC(0)
>>> qml.FermiA(3)

These operators can be multiplied by each other to create :math:`n`-orbital fermionic operators such
as :math:`a^{\dagger}_0 a_0 a^{\dagger}_3 a_3` that we call a Fermi word.

>>> qml.FermiC(0) * qml.FermiA(0) * qml.FermiC(3) * qml.FermiA(3)
<FermiWord = '0+ 0- 3+ 3-'>

The Fermi words can be linearly combined to create fermionic operators that we call a Fermi
Sentence. For instance, a fermionic Hamiltonian such as
:math:`H = 1.2 a^{\dagger}_0 a_0  + 2.3 a^{\dagger}_3 a_3` can be constructed as

>>> h = 1.2 * qml.FermiC(0) * qml.FermiA(0) + 2.3 * qml.FermiC(3) * qml.FermiA(3)
>>> h
1.2 * '0+ 0-'
+ 2.3 * '3+ 3-'

These fermionic objects can be mapped to the qubit basis by using the function
:func:`jordan_wigner`

>>> qml.jordan_wigner(h)
((-1.75+0j)*(Identity(wires=[0]))) + ((0.6+0j)*(PauliZ(wires=[0]))) + ((1.15+0j)*(PauliZ(wires=[3])))

A Fermi word or a Fermi Sentence can be also constructed directly

FermiWord and FermiSentence
---------------------------

Fermi words and Fermi Sentences can also be constructed directly with :class:`~pennylane.FermiWord`
and :class:`~pennylane.FermiSentence` by passing dictionaries that define the fermionic operators.


FermiWord and FermiSentence
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
>>> pw1 = qml.pauli.PauliWord({0:"X", 1:"Z"})  # X@Z
>>> pw2 = qml.pauli.PauliWord({0:"Y", 1:"Z"})  # Y@Z
>>> pw1, pw2
(X(0) @ Z(1), Y(0) @ Z(1))

The purpose of this class is to efficiently compute products of Pauli words and
obtain the matrix representation.

>>> pw1 * pw2
(Z(0), 1j)
>>> pw1.to_mat(wire_order=[0, 1])
array([[ 0,  0,  1,  0],
       [ 0,  0,  0, -1],
       [ 1,  0,  0,  0],
       [ 0, -1,  0,  0]])


The :class:`~pennylane.pauli.PauliSentence` class represents linear combinations of
Pauli words. Using a similar dictionary based approach we can efficiently add, multiply
and extract the matrix of operators in this representation.

>>> ps1 = qml.pauli.PauliSentence({pw1: 1.2, pw2: 0.5j})
>>> ps2 = qml.pauli.PauliSentence({pw1: -1.2})
>>> ps1
1.2 * X(0) @ Z(1)
+ 0.5j * Y(0) @ Z(1)
>>> ps1 + ps2
0.0 * X(0) @ Z(1)
+ 0.5j * Y(0) @ Z(1)
>>> ps1 * ps2
-1.44 * I
+ (-0.6+0j) * Z(0)
>>> (ps1 + ps2).to_mat(wire_order=[0, 1])
array([[ 0. +0.j,  0. +0.j,  0.5+0.j,  0. +0.j],
       [ 0. +0.j,  0. +0.j,  0. +0.j, -0.5+0.j],
       [-0.5+0.j,  0. +0.j,  0. +0.j,  0. +0.j],
       [ 0. +0.j,  0.5+0.j,  0. +0.j,  0. +0.j]])
