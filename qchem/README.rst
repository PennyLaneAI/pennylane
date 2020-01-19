PennyLane-Qchem
===============

PennyLane-Qchem is a PennyLane package that provides tools to generate and decompose the
many-electron Hamiltonian of a molecular system. Once generated, the Hamiltonian can
be used to perform quantum chemistry simulations in PennyLane.

See the `documentation <https://pennylane.readthedocs.io/en/stable/introduction/chemistry.html>`_
for more details.

Features
--------

- *Focused on near-term hardware*. Currently written to be used with
  Variational Quantum Eigensolver (VQE) algorithms to estimate the ground state energy
  of molecules using quantum simulators and/or near-term quantum computers.

- *Standardized input data*. As most of the electronic structure packages, PennLane-Qchem
  just requires the atomic structure of the molecule to be simulated, its net charge,
  the spin-multiplicity of the ground state and the atomic basis set used to solve the
  *meanfield* electronic structure problem.

- *Low barrier to entry*. Designed to require little prior knowledge of quantum computing from
  the user. It's our job to know how to encode problems into quantum hardware!

- *Integration*. PennyLane-Qchem makes use of `OpenFermion <https://github.com/quantumlib/OpenFermion>`__
  and the electronic structure package plugins `OpenFermion-Psi4 <https://github.com/quantumlib/OpenFermion-Psi4>`__
  and `OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-PySCF>`__.

Installation
------------

PennyLane-Qchem requires Python version 3.5 and above, and the following dependencies:

* `pySCF <https://sunqm.github.io/pyscf>`__
  and `OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-pyscf>`__ >= 0.4

* (optional) `Psi4 <http://www.psicode.org/>`__
  and `OpenFermion-Psi4 <https://github.com/quantumlib/OpenFermion-Psi4>`__ >= 0.4

  The easiest way to install Psi4 is via Ananconda:

  .. code-block:: bash

    conda install psi4 psi4-rt -c psi4

* `OpenFermion <https://github.com/quantumlib/OpenFermion>`__ >= 0.10

* `Open Babel <https://openbabel.org>`__ (optional)

  Open Babel can be installed using ``apt`` if on Ubuntu/Debian:

  .. code-block:: bash

      sudo apt install openbabel

  or using Anaconda:

  .. code-block:: bash

      conda install -c conda-forge openbabel

Once the requirements are installed, PennyLane-Qchem can be installed using pip:

.. code-block:: bash

    pip install pennylane-qchem

Once installed, it is available via

.. code-block:: python

	from pennylane import qchem

Authors
-------

`Alain Delgado Gran <https://github.com/agran2018>`__, `Josh Izaac <https://github.com/josh146>`__,
`Zeyue Niu <https://github.com/zeyueN>`__, `Soran Jahangiri <https://github.com/soranjh>`__,
`Juan Miguel Arrazola <https://github.com/ixfoduap>`__ and `Nathan Killoran <https://github.com/co9olguy>`__.
