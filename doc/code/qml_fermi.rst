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
and :class:`~pennylane.FermiA`, respectively, by passing the index of the orbital that the fermionic
operator acts on. For instance, the operators :math:`a^{\dagger}_0` :math:`a_3` are constructed as

>>> qml.FermiC(0)
a⁺(0)
>>> qml.FermiA(3)
a(3)

These operators can be multiplied by each other to create :math:`n`-orbital fermionic operators such
as :math:`a^{\dagger}_0 a_0 a^{\dagger}_3 a_3` that we call a Fermi word.

>>> qml.FermiC(0) * qml.FermiA(0) * qml.FermiC(3) * qml.FermiA(3)
a⁺(0) a(0) a⁺(3) a(3)

The Fermi words can be linearly combined to create fermionic operators that we call a Fermi
Sentence. For instance, a fermionic Hamiltonian such as
:math:`H = 1.2 a^{\dagger}_0 a_0  + 2.3 a^{\dagger}_3 a_3` can be constructed as

>>> h = 1.2 * qml.FermiC(0) * qml.FermiA(0) + 2.3 * qml.FermiC(3) * qml.FermiA(3)
>>> h
1.2 * a⁺(0) a(0)
+ 2.3 * a⁺(3) a(3)

These fermionic objects can be mapped to the qubit basis by using the function
:func:`jordan_wigner`

>>> qml.jordan_wigner(h)
((1.75+0j)*(Identity(wires=[0]))) + ((-0.6+0j)*(PauliZ(wires=[0]))) + ((-1.15+0j)*(PauliZ(wires=[3])))

FermiWord and FermiSentence
---------------------------

Fermi words and Fermi Sentences can also be constructed directly with
:class:`~pennylane.fermi.FermiWord` and :class:`~pennylane.fermi.FermiSentence` by passing
dictionaries that define the fermionic operators.

For Fermi words, the dictionary items identify the fermionic creation and annihilation operators.
Each operators is identified by a tuple of two integers representing its position in the Fermi word
the orbital it acts on, and a symbol identifying its type. We use ``'+'`` and ``'-'`` to denote
creation and annihilation operators, respectively. The operator
:math:`a^{\dagger}_0 a_0 a^{\dagger}_3 a_3` can then be constructed with

>>> qml.fermi.FermiWord({(0, 0): '+', (1, 3): '-'})
a⁺(0) a(3)

A Fermi sentence can be constructed directly by passing a dictionary of Fermi words and their
corresponding coefficients to the :class:`~pennylane.fermi.FermiSentence` class. For instance, the
Fermi sentence :math:`1.2 a^{\dagger}_0 a_0  + 2.3 a^{\dagger}_3 a_3` can be constructed as

>>> fw1 = qml.fermi.FermiWord({(0, 0): '+', (1, 0): '-'})
>>> fw2 = qml.fermi.FermiWord({(0, 3): '+', (1, 3): '-'})
>>> qml.fermi.FermiSentence({fw1: 1.2, fw2: 2.3})
1.2 * a⁺(0) a(0)
+ 2.3 * a⁺(3) a(3)

Mapping to qubit operators
--------------------------

The fermionic operators can be mapped to the qubit basis by using the
:func:`~pennylane.jordan_wigner` function. This function can be used to map creation and
annihilation operators as well as Fermi words and Fermi Sentences.

>>> qml.jordan_wigner(qml.FermiA(1))
(0.5*(PauliZ(wires=[0]) @ PauliX(wires=[1]))) + (0.5j*(PauliZ(wires=[0]) @ PauliY(wires=[1])))

>>> qml.jordan_wigner(qml.FermiC(1) * qml.FermiA(1))
((0.5+0j)*(Identity(wires=[1]))) + ((-0.5+0j)*(PauliZ(wires=[1])))

>>> qml.jordan_wigner(0.5 * qml.FermiC(1) * qml.FermiA(1) + 0.75 * qml.FermiC(2) * qml.FermiA(2))
((0.625+0j)*(Identity(wires=[1]))) + ((-0.25+0j)*(PauliZ(wires=[1]))) + ((-0.375+0j)*(PauliZ(wires=[2])))
