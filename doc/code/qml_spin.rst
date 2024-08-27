qml.fermi
=========

Overview
--------

This module contains functions and classes for creating and manipulating spin Hamiltonian operators.

.. currentmodule:: pennylane.spin

.. automodapi:: pennylane.spin
    :no-heading:
    :no-main-docstr:
    :no-inherited-members:

Transverse-Field Ising Hamiltonian
----------------------------------

The transverse-field Ising Hamiltonian for a given physical arrangement of spins, i.e. a lattice can
be obtained by using the :func:`~pennylane.spin.transverse_ising` function. We pass in the shape of
the lattice and number of lattice sites along :math:`x`, :math:`y` and :math:`z` axes. For example,
the Hamiltonian for a square lattice of size :math:`2x2` can be constructed with

>>> qml.spin.transverse_ising(lattice="square", n_cells=[2, 2], coupling=0.5, h=0.2)
    -0.5 * (Z(0) @ Z(1))
  + -0.5 * (Z(0) @ Z(2))
  + -0.5 * (Z(1) @ Z(3))
  + -0.5 * (Z(2) @ Z(3))
  + -0.2 * X(0)
  + -0.2 * X(1)
  + -0.2 * X(2)
  + -0.2 * X(3)


Heisenberg Hamiltonian
----------------------------------

The Heisenberg Hamiltonian for a given physical arrangement of spins, i.e. a lattice can
be obtained by using the :func:`~pennylane.spin.heisenberg` function. We pass in the shape of
the lattice and number of lattice sites along :math:`x`, :math:`y` and :math:`z` axes. For example,
the Hamiltonian for a triangular lattice of size :math:`2x2` can be constructed with

>>> qml.spin.heisenberg(lattice="triangle", n_cells=[2, 2], coupling=[0.5, 0.1, 0.2])
    0.5 * (X(0) @ X(1))
  + 0.1 * (Y(0) @ Y(1))
  + 0.2 * (Z(0) @ Z(1))
  + 0.5 * (X(0) @ X(2))
  + 0.1 * (Y(0) @ Y(2))
  + 0.2 * (Z(0) @ Z(2))
  + 0.5 * (X(1) @ X(2))
  + 0.1 * (Y(1) @ Y(2))
  + 0.2 * (Z(1) @ Z(2))
  + 0.5 * (X(1) @ X(3))
  + 0.1 * (Y(1) @ Y(3))
  + 0.2 * (Z(1) @ Z(3))
  + 0.5 * (X(2) @ X(3))
  + 0.1 * (Y(2) @ Y(3))
  + 0.2 * (Z(2) @ Z(3))


Fermi-Hubbard Hamiltonian
-------------------------

The Fermi-Hubbard Hamiltonian for a given physical arrangement of spins, i.e. a lattice can
be obtained by using the :func:`~pennylane.spin.fermi_hubbard` function. We pass in the shape
of the lattice and number of lattice sites along :math:`x`, :math:`y` and :math:`z` axes. For
example, the Hamiltonian for a chain of size :math:`2` can be constructed with

>>> qml.spin.fermi_hubbard(lattice="chain", n_cells=[2], hopping=0.1, coulomb=0.0)
    -0.05 * (Y(0) @ Z(1) @ Y(2))
  + -0.05 * (X(0) @ Z(1) @ X(2))
  + -0.05 * (Y(1) @ Z(2) @ Y(3))
  + -0.05 * (X(1) @ Z(2) @ X(3))
