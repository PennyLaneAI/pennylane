.. role:: html(raw)
   :format: html

.. _intro_ref_chm:

Quantum Chemistry
=================

PennyLane provides the :mod:`~.qchem` module to perform quantum chemistry simulations. It
contains a differentiable Hartree-Fock solver and the functionality to construct a
fully-differentiable molecular Hamiltonian that can be used as input to quantum algorithms
such as the variational quantum eigensolver (VQE) algorithm. The :mod:`~.qchem` module
also provides tools for building other observables such as molecular dipole moment, spin
and particle number observables. The theoretical foundation of the quantum chemistry functionality
in PennyLane is explained in our `white paper <https://arxiv.org/abs/2111.09967>`_.

Building the electronic Hamiltonian
-----------------------------------

The :mod:`~.qchem` module provides access to a driver function :func:`~.molecular_hamiltonian`
to generate the electronic Hamiltonian in a single call. For example,

.. code-block:: python

    import pennylane as qp
    from pennylane import numpy as np

    symbols = ["H", "H"]
    geometry = np.array([[0., 0., -0.66140414], [0., 0., 0.66140414]])
    molecule = qml.qchem.Molecule(symbols, geometry)
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule)

where:

* ``hamiltonian`` is the qubit Hamiltonian of the molecule represented as a PennyLane Hamiltonian and

* ``qubits`` is the number of qubits needed to perform the quantum simulation.

The :func:`~.molecular_hamiltonian` function can also be used to construct the molecular Hamiltonian
with external backends that use the
`OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-PySCF>`_ plugin or the
electronic structure package `PySCF <https://github.com/pyscf/pyscf>`_, which requires separate
installation. These backends are non-differentiable and can be selected by setting
``method='openfermion'`` or ``method='pyscf'`` in ``molecular_hamiltonian``.

Furthermore, the net charge,
the `spin multiplicity <https://en.wikipedia.org/wiki/Multiplicity_(chemistry)>`_, the
`atomic basis functions <https://www.basissetexchange.org/>`_, the mapping method and the active
space can also be specified for each backend.

.. code-block:: python

    molecule = qml.qchem.Molecule(
        symbols,
        geometry,
        charge=0,
        mult=1,
        basis_name='sto-3g')
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        molecule,
        mapping='jordan_wigner',
        active_electrons=2,
        active_orbitals=2
    )

Importing the Hamiltonian and ansatz state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the electronic Hamiltonian is built independently using 
`OpenFermion <https://github.com/quantumlib/OpenFermion>`_ tools, it can be readily converted 
to a PennyLane observable using the :func:`~.pennylane.import_operator` function. There is also 
capability to import wavefunctions (states) that have been pre-computed by traditional quantum chemistry methods
from `PySCF <https://github.com/pyscf/pyscf>`_, which could be used to for example provide a better
starting point to a quantum algorithm. State import can be accomplished using the :func:`~pennylane.qchem.import_state` 
utility function.


Importing molecular structure data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The atomic structure of a molecule can be either defined as an array or imported from an external
file using the :func:`~.read_structure` function:

.. code-block:: python

    symbols, geometry = qml.qchem.read_structure('h2.xyz')


VQE simulations
---------------

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical computational scheme,
where a quantum computer is used to prepare the trial wave function of a molecule and to measure
the expectation value of the *electronic Hamiltonian*, while a classical optimizer is used to
find its ground state.

PennyLane supports treating Hamiltonians just like any other observable, and the
expectation value of a Hamiltonian can be calculated using ``qml.expval``:

.. code-block:: python

    dev = qml.device('default.qubit', wires=4)

    symbols = ["H", "H"]
    geometry = np.array([[0., 0., -0.66140414], [0., 0., 0.66140414]])
    molecule = qml.qchem.Molecule(symbols, geometry)
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule)

    @qml.qnode(dev)
    def circuit(params):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        qml.DoubleExcitation(params, wires=[0, 1, 2, 3])
        return qml.expval(hamiltonian)

    params = np.array(0.20885146442480412, requires_grad=True)
    circuit(params)

.. code-block:: text

    tensor(-1.13618912, requires_grad=True)

The circuit parameter can be optimized using the interface of choice.

.. note::

    For more details on VQE and the quantum chemistry functionality available in
    :mod:`~pennylane.qchem`, check out the PennyLane `quantum chemistry tutorials <https://pennylane.ai/search/?contentType=DEMO&categories=quantum%20chemistry>`_.


Quantum chemistry functions and classes
---------------------------------------

PennyLane supports the following quantum chemistry functions and classes.

Molecular integrals and matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qchem.attraction_integral
    ~pennylane.qchem.attraction_matrix
    ~pennylane.qchem.contracted_norm
    ~pennylane.qchem.core_matrix
    ~pennylane.qchem.dipole_integrals
    ~pennylane.qchem.electron_integrals
    ~pennylane.qchem.electron_repulsion
    ~pennylane.qchem.expansion
    ~pennylane.qchem.gaussian_kinetic
    ~pennylane.qchem.gaussian_moment
    ~pennylane.qchem.gaussian_overlap
    ~pennylane.qchem.hermite_moment
    ~pennylane.qchem.kinetic_integral
    ~pennylane.qchem.kinetic_matrix
    ~pennylane.qchem.mol_density_matrix
    ~pennylane.qchem.moment_integral
    ~pennylane.qchem.moment_matrix
    ~pennylane.qchem.nuclear_attraction
    ~pennylane.qchem.overlap_integral
    ~pennylane.qchem.overlap_matrix
    ~pennylane.qchem.primitive_norm
    ~pennylane.qchem.repulsion_integral
    ~pennylane.qchem.repulsion_tensor

:html:`</div>`


Differentiable Hartree-Fock
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qchem.hf_energy
    ~pennylane.qchem.nuclear_energy
    ~pennylane.qchem.scf

:html:`</div>`


Hartree-Fock with external packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qchem.decompose
    ~pennylane.qchem.meanfield
    ~pennylane.qchem.one_particle
    ~pennylane.qchem.two_particle

:html:`</div>`


Differentiable observables
^^^^^^^^^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qchem.diff_hamiltonian
    ~pennylane.qchem.dipole_moment
    ~pennylane.qchem.fermionic_dipole
    ~pennylane.qchem.fermionic_hamiltonian
    ~pennylane.qchem.fermionic_observable
    ~pennylane.jordan_wigner
    ~pennylane.qchem.molecular_hamiltonian
    ~pennylane.qchem.qubit_observable

:html:`</div>`


Other observables
^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qchem.dipole_of
    ~pennylane.qchem.observable
    ~pennylane.qchem.particle_number
    ~pennylane.qchem.spin2
    ~pennylane.qchem.spinz

:html:`</div>`


Qubit tapering
^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qchem.clifford
    ~pennylane.qchem.optimal_sector
    ~pennylane.paulix_ops
    ~pennylane.symmetry_generators
    ~pennylane.taper
    ~pennylane.qchem.taper_hf
    ~pennylane.qchem.taper_operation

:html:`</div>`

Utility functions
^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qchem.active_space
    ~pennylane.qchem.excitations
    ~pennylane.qchem.excitations_to_wires
    ~pennylane.qchem.factorize
    ~pennylane.math.decomposition.givens_decomposition
    ~pennylane.qchem.hf_state
    ~pennylane.import_operator
    ~pennylane.qchem.import_state
    ~pennylane.qchem.mol_data
    ~pennylane.qchem.read_structure

:html:`</div>`

Molecule class and basis functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.qchem.atom_basis_data
    ~pennylane.qchem.BasisFunction
    ~pennylane.qchem.Molecule
    ~pennylane.qchem.mol_basis_data

:html:`</div>`
