qml.fermi
=========

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
and :class:`~pennylane.FermiA`, respectively. We pass in the index of the orbital that the
operator acts on. For example, the operators :math:`a^{\dagger}_0` and :math:`a_3`, acting on the
:math:`0\text{th}` and :math:`3\text{rd}` orbitals, are constructed as

>>> qml.FermiC(0)
a⁺(0)
>>> qml.FermiA(3)
a(3)

These operators can be multiplied by each other to create :math:`n`-orbital fermionic operators such
as :math:`a^{\dagger}_0 a_0 a^{\dagger}_3 a_3`. We call these :math:`n`-orbital fermionic operators
Fermi words.

>>> qml.FermiC(0) * qml.FermiA(0) * qml.FermiC(3) * qml.FermiA(3)
a⁺(0) a(0) a⁺(3) a(3)

The Fermi words can be linearly combined to create fermionic operators that we call Fermi
sentences. For instance, a fermionic Hamiltonian such as
:math:`H = 1.2 a^{\dagger}_0 a_0  + 2.3 a^{\dagger}_3 a_3` can be constructed from Fermi words with

>>> h = 1.2 * qml.FermiC(0) * qml.FermiA(0) + 2.3 * qml.FermiC(3) * qml.FermiA(3)
>>> h
1.2 * a⁺(0) a(0)
+ 2.3 * a⁺(3) a(3)

Mapping to qubit operators
--------------------------

The fermionic operators can be mapped to the qubit basis by using the
:func:`~pennylane.jordan_wigner` function. This function can be used to map
:class:`~pennylane.FermiC` and :class:`~pennylane.FermiA` operators as well as Fermi words and
Fermi sentences.

>>> qml.jordan_wigner(qml.FermiA(1))
0.5 * (Z(0) @ X(1)) + 0.5j * (Z(0) @ Y(1))

>>> qml.jordan_wigner(qml.FermiC(1) * qml.FermiA(1))
(0.5+0j) * I(1) + (-0.5+0j) * Z(1)

>>> f = 0.5 * qml.FermiC(1) * qml.FermiA(1) + 0.75 * qml.FermiC(2) * qml.FermiA(2)
>>> qml.jordan_wigner(f)
(
    (0.625+0j) * I(1)
  + (-0.25+0j) * Z(1)
  + (-0.375+0j) * Z(2)
)

FermiWord and FermiSentence
---------------------------

Fermi words and Fermi sentences can also be constructed directly with
:class:`~pennylane.fermi.FermiWord` and :class:`~pennylane.fermi.FermiSentence` by passing
dictionaries that define the fermionic operators.

For Fermi words, the dictionary items define the fermionic creation and annihilation operators.
The keys of the dictionary are tuples of two integers. The first integer represents the
position of the creation/annihilation operator in the Fermi word and the second integer represents
the orbital it acts on. The values of the dictionary are one of ``'+'`` or ``'-'`` symbols that
denote creation and annihilation operators, respectively. The operator
:math:`a^{\dagger}_0 a_3 a^{\dagger}_1` can then be constructed with

>>> qml.FermiWord({(0, 0): '+', (1, 3): '-', (2, 1): '+'})
a⁺(0) a(3) a⁺(1)

A Fermi sentence can be constructed directly by passing a dictionary of Fermi words and their
corresponding coefficients to the :class:`~pennylane.fermi.FermiSentence` class. For instance, the
Fermi sentence :math:`1.2 a^{\dagger}_0 a_0  + 2.3 a^{\dagger}_3 a_3` can be constructed as

>>> fw1 = qml.FermiWord({(0, 0): '+', (1, 0): '-'})
>>> fw2 = qml.FermiWord({(0, 3): '+', (1, 3): '-'})
>>> qml.FermiSentence({fw1: 1.2, fw2: 2.3})
1.2 * a⁺(0) a(0)
+ 2.3 * a⁺(3) a(3)
