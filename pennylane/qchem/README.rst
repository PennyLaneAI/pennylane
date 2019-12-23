PennyLane-Qchem
===============

PennyLane-Qchem is a Python package that can be used to generate and decompose the many-electron Hamiltonian of a molecular system to perform quantum chemistry simulations using PennyLane.  

Features
--------

- *Focused on near-term hardware*. Currently written to be used in the framework of Variational Quantum Eigensolver (VQE) algorithms to estimate the ground state energy of molecules using quantum simulators and/or near-term quantum computers.

- *Standardized input data*. As most of the electronic structure packages, PennLane-Qchem just requires the atomic structure of the molecule to be simulated, its net charge, the spin-multiplicity of the ground state and the atomic basis set used to solve the *meanfield* electronic structure problem.

- *Low barrier to entry*. Designed to require little prior knowledge of quantum computing from
  the user. It's our job to know how to encode problems into quantum hardware!

- *Integration*. PennyLane-Qchem makes use of `OpenFermion <https://github.com/quantumlib/OpenFermion>`__ and the electronic structure package plugins `OpenFermion-Psi4 <https://github.com/quantumlib/OpenFermion-Psi4>`__ and `OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-PySCF>`__.

Getting started
---------------

To get started with PennyLane-QChem install it by running `pip install -e .` from the top level directory.


Authors
-------

`Alain Delgado Gran <https://github.com/agran2018>`__, `Josh Izaac <https://github.com/josh146>`__,
`Zeyue Niu <https://github.com/zeyueN>`__, and `Nathan Killoran <https://github.com/co9olguy>`__.
