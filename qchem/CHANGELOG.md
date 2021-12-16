# Release 0.21.0-dev

<h3>New features</h3>

<h3>Improvements</h3>

<h3>Bug fixes</h3>

<h3>Breaking changes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

# Release 0.19.0

<h3>New features</h3>

* The ``dipole`` function has been added to the ``obs`` module
  to construct the electric dipole operator of a molecule.
  Currently, the implemented function relies on a PySCF functionality
  to load the dipole matrix elements in the atomic basis.
  [(#1698)](https://github.com/PennyLaneAI/pennylane/pull/1698)

<h3>Improvements</h3>

* The ``meanfield`` function has been modified to avoid creating
  a directory tree to the HF data file. Now the filename output by
  the function encodes the qchem package and basis set
  used to run the HF calculations. This ensures compatibility
  with multiprocessing environment
  [(#1854)](https://github.com/PennyLaneAI/pennylane/pull/1854)

<h3>Bug fixes</h3>

* Pins PySCF to version `>=1.7.2`, `<2.0` to ensure that features,
  tests and documentation continue to work.
  [(#1827)](https://github.com/PennyLaneAI/pennylane/pull/1827)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Alain Delgado Gran, Josh Izaac, Soran Jahangiri,
Lee James O'Riordan, Romain Moyard.

# Release 0.17.0

<h3>Bug fixes</h3>

* The types of the Hamiltonian terms built from an OpenFermion ``QubitOperator`` using the
  ``convert_observable`` function, are the same with respect to the analogous observable
  built directly using PennyLane operations.
  [(#1525)](https://github.com/PennyLaneAI/pennylane/pull/1525)

* Requires the H5Py dependency to be `H5Py<=3.2.1` due to incompatibilities between `pyscf>=1.7.2` and `H5Py==3.3.0`.
  [(#1430)](https://github.com/PennyLaneAI/pennylane/pull/1430)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Josh Izaac, Romain Moyard.

# Release 0.16.0

<h3>Improvements</h3>

* Eases the PySCF dependency to ``pyscf>=1.7.2``.
  [(#1254)](https://github.com/PennyLaneAI/pennylane/pull/1254)

<h3>Bug fixes</h3>

* Include tolerance in the``convert_observable`` function to check if the input QubitOperator
  contains complex coefficients. This avoid raising an error if the coefficient's imaginary part is less than `2.22e-08`.
  [(#1309)](https://github.com/PennyLaneAI/pennylane/pull/1309)

* An error message is raised if a QubitOperator with complex coefficients is passed
  to the ``convert_observable`` function. At present, the ``vqe.Hamiltonian`` class does not
  support complex coefficients.
  [(#1277)](https://github.com/PennyLaneAI/pennylane/pull/1277)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Alain Delgado Gran, Zeyue Niu.

# Release 0.15.1

<h3>Bug fixes</h3>

* The version requirement for PySCF has been modified to allow for `pyscf>=1.7.2`.
  [(#1254)](https://github.com/PennyLaneAI/pennylane/pull/1254)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Antal Száva.

# Release 0.15.0

<h3>Breaking changes</h3>

* The molecular geometry is now stored by a list containing the atomic symbols and
  a 1D array with the position of the atoms in atomic units.

  - The `read_structure` function returns a list with the symbols of the atoms and
    the array with the atomic positions.

  - The `meanfield` and `molecular_hamiltonian` functions take separately the
    list of atomic symbols and the array with the atomic coordinates.

  - Labelling the molecule is now optional as we have made `name` a keyword argument
    in the `meanfield` and `molecular_hamiltonian` functions.

    For example:

    ```pycon
    >>> symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.661, 0., 0., 0.661]))
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    ```

  This allows users to more easily build parametrized electronic Hamiltonians
  [(#1078)](https://github.com/PennyLaneAI/pennylane/pull/1078)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Alain Delgado Gran, Soran Jahangiri.


# Release 0.13.1

<h3>Bug fixes</h3>

* Updates `PennyLane-QChem` to support the new OpenFermion v1.0 release.
  [(#973)](https://github.com/PennyLaneAI/pennylane/pull/973)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Josh Izaac

# Release 0.13.0

<h3>Improvements</h3>

* Many-body observables are now built from the OpenFermion `FermionOperator` representation of
  one-particle and two-particle second-quantized operators.
  [(#854)](https://github.com/PennyLaneAI/pennylane/pull/854)

  This improvement brings the following advantages:

  - Extra tables to store the indices of the orbitals and the corresponding
    matrix elements are not needed.

  - The functions `observable`, `one_particle` and `two_particle` are
    significantly simplified.

  - The methodology to build many-body observables in PL-QChem is more consistent.

  - There is no longer a need to keep track of the contribution due to core orbitals
    when an active space is defined. This is now handled internally.

<h3>Bug fixes</h3>

* The `qchem._terms_to_qubit_operator` function is now updated to handle tensor products with
  `Identity` observables.
  [(#928)](https://github.com/PennyLaneAI/pennylane/pull/928)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Alain Delgado Gran, Soran Jahangiri, Zeyue Niu.

# Release 0.12.0

<h3>New features since last release</h3>

* The functions `one_particle` and `two_particle` have been implemented
  to extend PennyLane-QChem capabilities to construct observables of many-body
  quantum systems. These functions can be used in conjunction with the
  `observable` function to construct electronic structure hamiltonians
  involving one- and two-particle operators.
  [(#809)](https://github.com/PennyLaneAI/pennylane/pull/809)

* The function `observable` in the `obs` module has been generalized to build
  many-body observables combining one- and two-particle operators (e.g., Hamiltonians)
  [(#791)](https://github.com/PennyLaneAI/pennylane/pull/791)

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fix calculation of the contribution of core orbitals to two-particle operators in the
  function two_particle.
  [(#825)](https://github.com/PennyLaneAI/pennylane/pull/825)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Thomas Bromley, Alain Delgado, Josh Izaac, Soran Jahangiri.

# Release 0.11.0

<h3>New features since last release</h3>

* Adds the function `qml.qchem.particle_number`, which computes the particle
  number operator of a given molecule.
  [(#698)](https://github.com/XanaduAI/pennylane/pull/698)

* The function ``get_spinZ_matrix_elements`` has been added to the
  ``obs`` module to generate the matrix elements required to build
  the total-spin projection operator by using the generic function
  ``observable`` as implemented in the same module.
  [(#696)](https://github.com/XanaduAI/pennylane/pull/696)

* The new module ``obs`` has been added to build many-body operators
  whose expectation values can be computed in PennyLane to simulate
  properties of interest of quantum systems. In particular, this PR adds
  the required functions to build the total-spin operator S^2. The adopted
  methodology is very general and is not restricted to molecular systems.
  [(#689)](https://github.com/XanaduAI/pennylane/pull/689)

* The new function ``excitations_to_wires`` has been implemented to map the particle-hole
  representations ph and pphh, generated by ``sd_excitations``, to the wires that the
  qchem templates act on. This implementation enables compliance with the
  generalized PennyLane templates required to build the UCCSD VQE ansatz.
  [(#679)](https://github.com/XanaduAI/pennylane/pull/679)

  For example:

  ```pycon
  >>> n_electrons = 2
  >>> n_spinorbitals = 4
  >>> ph_confs, pphh_confs = sd_excitations(n_electrons, n_spinorbitals)
  >>> print(ph_confs)
  [[0, 2], [1, 3]]
  >>> print(pphh_confs)
  [[0, 1, 2, 3]]

  >>> wires=['a0', 'b1', 'c2', 'd3']
  >>> ph, pphh = excitations_to_wires(ph_confs, pphh_confs, wires=wires)
  >>> print(ph)
  [['a0', 'b1', 'c2'], ['b1', 'c2', 'd3']]
  >>> print(pphh)
  [[['a0', 'b1'], ['c2', 'd3']]]
  ```

<h3>Improvements</h3>

* The naming convention used in the `structure` module has been propagated
  to the `obs` module.
  [(#759)](https://github.com/PennyLaneAI/pennylane/pull/759)

  The changes include:

  - `n_electrons` renamed to `electrons`.
  - `n_orbitals` renamed to `orbitals`.

  In addition, the argument `orbitals` is used now to pass the number of *spin*
  orbitals.

* The functions involved in observable conversions from/to OpenFermion now accept
  a new `wires` argument that can be used to specify the qubits-to-wires mapping
  when custom wires are used in Pennylane ansatz.
  [(#750)](https://github.com/PennyLaneAI/pennylane/pull/750)

* The functions involved in generating the single and double excitations from a
  a Hartree-Fock state and mapping them to the wires that the Unitary
  Coupled-Cluster (UCCSD) ansatz act on have been improved, with a more
  consistent naming convention and improved docstrings.
  [(#742)](https://github.com/PennyLaneAI/pennylane/pull/742)

  The changes include:

  - `sd_excitations` has been renamed to `excitations`.

  - The names of internal variables and arguments have been standardized
    to avoid using different languages mixing the terminologies
    "single/double excitations" and "particle-hole excitations".

  - The arguments of the function `excitations_to_wires` have been renamed.
    `ph_confs` → `singles`, `pphh_confs` → `doubles`

* The functions involved in the construction of the electronic Hamiltonian have been
  improved, with shorter names and improved docstrings, including adding complementary
  information to better explain the basics of quantum chemistry.
  [(#735)](https://github.com/PennyLaneAI/pennylane/pull/735)

  The changes include:

  - `active_space` is now independent of the OpenFermion `MolecularData` data structure.

  - `meanfield_data` has been renamed to `meanfield`, and modified to return
    the absolute path to the file with the meanfield electronic structure of the molecule.

  - `decompose_hamiltonian` has been renamed to `decompose`,
    due to the new `qml.utils.decompose_hamiltonian` function. This function has also been
    marked for deprecation.

  - `generate_hamiltonian` has been renamed to `molecular_hamiltonian`.
    The modified function contains an extended docstring that outlines the main steps to build the Hamiltonian.

  In addition to the above changes, function arguments have also been modified and improved; please
  see relevant function docstrings for more details.

* The total spin observable S^2 can be built straightforwardly using the
  function `spin2` as implemented in the `obs` module.
  [(#714)](https://github.com/XanaduAI/pennylane/pull/714)

* The total-spin projection observable S_z can be built straightforwardly using the
  function `spin_z` as implemented in the `obs` module.
  [(#711)](https://github.com/XanaduAI/pennylane/pull/711)

<h3>Breaking changes</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Alain Delgado, Josh Izaac, Soran Jahangiri, Maria Schuld

# Release 0.10.0

<h3>New features since last release</h3>

* The function ``hf_state`` outputs an array with the occupation-number
  representation of the Hartree-Fock (HF) state. This function can be used to
  set the qubit register to encode the HF state which is the typical starting
  point for quantum chemistry simulations using the VQE algorithm.
  [(#629)](https://github.com/XanaduAI/pennylane/pull/629)

<h3>Improvements</h3>

* The function ``convert_hamiltonian`` has been renamed to ``convert_observable``
  since it can be used to convert any OpenFermion QubitOperator to a PennyLane
  Observable. ``convert_observable`` will be used in the ``obs`` module to build
  observables linked to molecular properties.
  [(#677)](https://github.com/XanaduAI/pennylane/pull/677)

<h3>Breaking changes</h3>

* Removes support for Python 3.5.
  [(#639)](https://github.com/XanaduAI/pennylane/pull/639)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Alain Delgado, Josh Izaac, Soran Jahangiri, Maria Schuld

# Release 0.9.0

Initial release.
