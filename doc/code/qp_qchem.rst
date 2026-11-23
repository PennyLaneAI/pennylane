qp.qchem
=========

Overview
--------

.. currentmodule:: pennylane.qchem

The quantum chemistry module provides the functionality to perform Hartree-Fock calculations
and construct observables such as molecular Hamiltonians as well as dipole moment, spin and particle
number observables. It also includes functionalities to convert to and from Openfermion's 
`QubitOperator <https://quantumai.google/reference/python/openfermion/ops/QubitOperator>`__ and 
`FermionOperator <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`__.

.. note::

    The function :func:`~pennylane.math.decomposition.givens_decomposition` has been moved to the
    :mod:`~pennylane.math` module. The function is still available in the
    :mod:`~pennylane.qchem` module for backward compatibility, but it is recommended to import it
    from the new location in future code.

.. currentmodule:: pennylane.qchem

.. automodapi:: pennylane.qchem
    :no-heading:
    :include-all-objects:
    :skip: taper, symmetry_generators, paulix_ops, import_operator, givens_decomposition

External Hartree-Fock backends
------------------------------

The :func:`~.molecular_hamiltonian` function uses
`PySCF <https://github.com/pyscf/pyscf>`_ by default. Select ``method='openfermion'`` to use the
`OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-PySCF>`_ plugin instead.
The :func:`~.molecular_dipole` function uses OpenFermion-PySCF by default.
Both backends require separate installation and do not support automatic differentiation of
molecular parameters.

.. code-block:: python

    import pennylane as qp
    import numpy as np

    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    molecule = qp.qchem.Molecule(symbols, geometry, charge=0, mult=1, basis_name='sto-3g')
    hamiltonian, qubits = qp.qchem.molecular_hamiltonian(molecule, method='pyscf')

The non-differentiable backends require either ``OpenFermion-PySCF`` or ``PySCF`` to be installed by
the user with

.. code-block:: bash

    pip install openfermionpyscf
    pip install pyscf
