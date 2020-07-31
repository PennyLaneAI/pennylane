# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the core functions for electronic structure calculations,
and converting the resulting data structures to forms understood by PennyLane."""
import os
import subprocess
from shutil import copyfile

import numpy as np
from openfermion.hamiltonians import MolecularData
from openfermion.ops._qubit_operator import QubitOperator
from openfermion.transforms import bravyi_kitaev, get_fermion_operator, jordan_wigner
from openfermionpsi4 import run_psi4
from openfermionpyscf import run_pyscf

import pennylane as qml
from pennylane import Hamiltonian


def _exec_exists(prog):
    r"""Checks whether the executable program ``prog`` exists in any of the directories
    set in the ``PATH`` environment variable.

    Args:
        prog (str): name of the executable program

    Returns:
        boolean: ``True`` if the executable ``prog`` is found; ``False`` otherwise
    """
    for dir_in_path in os.environ["PATH"].split(os.pathsep):
        path_to_prog = os.path.join(dir_in_path, prog)
        if os.path.exists(path_to_prog):
            try:
                subprocess.call([path_to_prog], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except OSError:
                return False
            return True
    return False


def read_structure(filepath, outpath="."):
    r"""Reads the molecular structure from a file and creates a list containing the
    symbol and Cartesian coordinates of the atomic species.

    The `xyz <https://en.wikipedia.org/wiki/XYZ_file_format>`_ format is supported out of the box.
    If `Open Babel <https://openbabel.org/>`_ is installed,
    `any format recognized by Open Babel <https://openbabel.org/wiki/Category:Formats>`_
    is also supported. Additionally, the new file ``structure.xyz``,
    containing the geometry of the molecule, is created in a directory with path given by
    ``outpath``.


    Open Babel can be installed using ``apt`` if on Ubuntu:

    .. code-block:: bash

        sudo apt install openbabel

    or using Anaconda:

    .. code-block:: bash

        conda install -c conda-forge openbabel

    See the Open Babel documentation for more details on installation.

    **Example usage:**

    >>> read_structure('h2_ref.xyz')
    [['H', (0.0, 0.0, -0.35)], ['H', (0.0, 0.0, 0.35)]]

    Args:
        filepath (str): name of the molecular structure file in the working directory
            or the full path to the file if it is located in a different folder
        outpath (str): path to the output directory

    Returns:
        list: for each atomic species, a list containing the symbol and the Cartesian coordinates
    """

    obabel_error_message = (
        "Open Babel converter not found:\n"
        "If using Ubuntu or Debian, try: 'sudo apt install openbabel' \n"
        "If using openSUSE, try: 'sudo zypper install openbabel' \n"
        "If using CentOS or Fedora, try: 'sudo snap install openbabel' "
        "Open Babel can also be downloaded from http://openbabel.org/wiki/Main_Page, \n"
        "make sure you add it to the PATH environment variable. \n"
        "If Anaconda is installed, try: 'conda install -c conda-forge openbabel'"
    )

    extension = filepath.split(".")[-1].strip().lower()

    file_in = filepath.strip()
    file_out = os.path.join(outpath, "structure.xyz")

    if extension != "xyz":
        if not _exec_exists("obabel"):
            raise TypeError(obabel_error_message)
        try:
            subprocess.run(
                ["obabel", "-i" + extension, file_in, "-oxyz", "-O", file_out], check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Open Babel error. See the following Open Babel "
                "output for details:\n\n {}\n{}".format(e.stdout, e.stderr)
            )
    else:
        copyfile(file_in, file_out)

    geometry = []
    with open(file_out) as f:
        for line in f.readlines()[2:]:
            species, x, y, z = line.split()
            geometry.append([species, (float(x), float(y), float(z))])
    return geometry


def meanfield(
    name, geometry, charge=0, mult=1, basis="sto-3g", package="pyscf", outpath="."
):  # pylint: disable=too-many-arguments
    r"""Generates a file from which the mean field electronic structure
    of the molecule can be retrieved.

    This function uses OpenFermion-PySCF and OpenFermion-Psi4 plugins to
    perform the Hartree-Fock (HF) calculation for the polyatomic system using the quantum
    chemistry packages ``PySCF`` and ``Psi4``, respectively. The mean field electronic
    structure is saved in an hdf5-formatted output file in the directory
    ``os.path.join(outpath, package, basis)``.

    Args:
        name (str): String used to label the molecule
        geometry (list): List containing the symbol and Cartesian coordinates for each atom
        charge (int): Net charge of the system
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` for
            :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values for ``mult`` are :math:`1, 2, 3, \ldots`. If not specified,
            a closed-shell HF state is assumed.
        basis (str): Atomic basis set used to represent the HF orbitals. Basis set
            availability per element can be found
            `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement>`_
        package (str): Quantum chemistry package used to solve the Hartree-Fock equations.
            Either ``'pyscf'`` or ``'psi4'`` can be used.
        outpath (str): Path to output directory

    Returns:
        str: full path to the file containing the mean field electronic structure

    **Example**

    >>> name = 'h2'
    >>> geometry = [['H', (0.0, 0.0, -0.35)], ['H', (0.0, 0.0, 0.35)]]
    >>> meanfield(name, geometry)
    ./pyscf/sto-3g/h2
    """

    package = package.strip().lower()

    if package not in ("psi4", "pyscf"):
        error_message = (
            "Integration with quantum chemistry package '{}' is not available. \n Please set"
            " 'package' to 'pyscf' or 'psi4'.".format(package)
        )
        raise TypeError(error_message)

    package_dir = os.path.join(outpath.strip(), package)
    basis_dir = os.path.join(package_dir, basis.strip())

    if not os.path.isdir(package_dir):
        os.mkdir(package_dir)
        os.mkdir(basis_dir)
    elif not os.path.isdir(basis_dir):
        os.mkdir(basis_dir)

    path_to_file = os.path.join(basis_dir, name.strip())

    molecule = MolecularData(geometry, basis, mult, charge, filename=path_to_file)

    if package == "psi4":
        run_psi4(molecule, run_scf=1, verbose=0, tolerate_error=1)

    if package == "pyscf":
        run_pyscf(molecule, run_scf=1, verbose=0)

    return path_to_file


def active_space(n_electrons, n_orbitals, mult=1, nact_els=None, nact_orbs=None):
    r"""Builds the active space by partitioning the set of Hartree-Fock (HF) orbitals.

    Post-Hartree-Fock electron correlation methods expand the many-body wave function
    as a linear combination of Slater determinants, commonly referred to as configurations.
    This configurations are generated by exciting electrons from the occupied to the
    unoccupied HF orbitals. Since the number of configurations increases combinatorially
    with the number of electrons and orbitals this expansion can be truncated by defining
    an active space. 

    The active space is created by classifying the HF orbitals as core, active and
    external orbitals:

    - Core orbitals are always occupied by two electrons
    - Active orbitals can be occupied by zero, one, or two electrons
    - The external orbitals are never occupied

    .. figure:: ../../_static/qchem/sketch_active_space.png
        :align: center
        :width: 50%

    .. note::
        The number of active *spin*-orbitals ``2*nact_orbs`` determines the number of
        qubits required to perform the quantum simulations of the electronic structure
        of the many-electron system.

    Args:
        n_electrons (int): Total number of electrons
        n_orbitals (int): Total number of orbitals
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` for
            :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values for ``mult`` are :math:`1, 2, 3, \ldots`. If not specified,
            a closed-shell HF state is assumed.
        nact_els (int): Number of active electrons. If not specified, all electrons
            are treated as active.
        nact_orbs (int): Number of active orbitals. If not specified, all orbitals
            are treated as active.

    Returns:
        tuple: lists of indices for core and active orbitals

    **Example**
    
    >>> n_electrons = 4
    >>> n_orbitals = 4
    >>> core, active = active_space(n_electrons, n_orbitals, nact_els=2, nact_orbs=2)
    >>> print(core) # core orbitals
    [0]
    >>> print(active) # active orbitals
    [1, 2]
    """
    # pylint: disable=too-many-branches

    if nact_els is None:
        ncore_orbs = 0
        core = []
    else:
        if nact_els <= 0:
            raise ValueError(
                "The number of active electrons ({}) " "has to be greater than 0.".format(nact_els)
            )

        if nact_els > n_electrons:
            raise ValueError(
                "The number of active electrons ({}) "
                "can not be greater than the total "
                "number of electrons ({}).".format(nact_els, n_electrons)
            )

        if nact_els < mult - 1:
            raise ValueError(
                "For a reference state with multiplicity {}, "
                "the number of active electrons ({}) should be "
                "greater than or equal to {}.".format(mult, nact_els, mult - 1)
            )

        if mult % 2 == 1:
            if nact_els % 2 != 0:
                raise ValueError(
                    "For a reference state with multiplicity {}, "
                    "the number of active electrons ({}) should be even.".format(mult, nact_els)
                )
        else:
            if nact_els % 2 != 1:
                raise ValueError(
                    "For a reference state with multiplicity {}, "
                    "the number of active electrons ({}) should be odd.".format(mult, nact_els)
                )

        ncore_orbs = (n_electrons - nact_els) // 2
        core = list(range(ncore_orbs))

    if nact_orbs is None:
        active = list(range(ncore_orbs, n_orbitals))
    else:
        if nact_orbs <= 0:
            raise ValueError(
                "The number of active orbitals ({}) " "has to be greater than 0.".format(nact_orbs)
            )

        if ncore_orbs + nact_orbs > n_orbitals:
            raise ValueError(
                "The number of core ({}) + active orbitals ({}) can not be "
                "greater than the total number of orbitals ({})".format(
                    ncore_orbs, nact_orbs, n_orbitals
                )
            )

        homo = (n_electrons + mult - 1) / 2
        if ncore_orbs + nact_orbs <= homo:
            raise ValueError(
                "For n_active_orbitals={}, there are no virtual orbitals "
                "in the active space.".format(nact_orbs)
            )

        active = list(range(ncore_orbs, ncore_orbs + nact_orbs))

    return core, active


def decompose_molecular_hamiltonian(hf_file, mapping="jordan_wigner", core=None, active=None):
    r"""Decomposes the electronic Hamiltonian into a linear combination of Pauli operators using
    OpenFermion tools.

    This function uses OpenFermion functions to build the second-quantized electronic Hamiltonian
    of the molecule and map it to the Pauli basis using the Jordan-Wigner or Bravyi-Kitaev
    transformation.

    Args:
        hf_file (str): Full path to the hdf5-formatted file with the
            Hartree-Fock electronic structure
        mapping (str): Specifies the transformation to map the fermionic Hamiltonian to the
            Pauli basis. Input values can be ``'jordan_wigner'`` or ``'bravyi_kitaev'``.
        core (list): Indices of core molecular orbitals, i.e., the orbitals that are
            not correlated in the many-body wave function
        active (list): Indices of active molecular orbitals, i.e., the orbitals used to
            build the correlated many-body wave function

    Returns:
        QubitOperator: an instance of OpenFermion's ``QubitOperator``

    **Example**

    >>> decompose_molecular_hamiltonian('./pyscf/sto-3g/h2', mapping='bravyi_kitaev')
    (-0.04207897696293986+0j) [] + (0.04475014401986122+0j) [X0 Z1 X2] +
    (0.04475014401986122+0j) [X0 Z1 X2 Z3] +(0.04475014401986122+0j) [Y0 Z1 Y2] +
    (0.04475014401986122+0j) [Y0 Z1 Y2 Z3] +(0.17771287459806262+0j) [Z0] +
    (0.17771287459806265+0j) [Z0 Z1] +(0.1676831945625423+0j) [Z0 Z1 Z2] +
    (0.1676831945625423+0j) [Z0 Z1 Z2 Z3] +(0.12293305054268105+0j) [Z0 Z2] +
    (0.12293305054268105+0j) [Z0 Z2 Z3] +(0.1705973832722409+0j) [Z1] +
    (-0.2427428049645989+0j) [Z1 Z2 Z3] +(0.1762764080276107+0j) [Z1 Z3] +
    (-0.2427428049645989+0j) [Z2]
    """

    # loading HF data from the hdf5 file
    molecule = MolecularData(filename=hf_file.strip())

    # getting the terms entering the second-quantized Hamiltonian
    terms_molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=core, active_indices=active
    )

    # generating the fermionic Hamiltonian
    fermionic_hamiltonian = get_fermion_operator(terms_molecular_hamiltonian)

    mapping = mapping.strip().lower()

    if mapping not in ("jordan_wigner", "bravyi_kitaev"):
        raise TypeError(
            "The '{}' transformation is not available. \n "
            "Please set 'mapping' to 'jordan_wigner' or 'bravyi_kitaev'.".format(mapping)
        )

    # fermionic-to-qubit transformation of the Hamiltonian
    if mapping == "bravyi_kitaev":
        return bravyi_kitaev(fermionic_hamiltonian)

    return jordan_wigner(fermionic_hamiltonian)


def _qubit_operator_to_terms(qubit_operator):
    r"""Converts OpenFermion ``QubitOperator`` to a 2-tuple of coefficients and
    PennyLane Pauli observables.

    Args:
        qubit_operator (QubitOperator): fermionic-to-qubit transformed operator in terms of
            Pauli matrices

    Returns:
        tuple[array[float], Iterable[pennylane.operation.Observable]]: coefficients and their
        corresponding PennyLane observables in the Pauli basis
    """
    if not qubit_operator.terms:  # added since can't unpack empty zip to (coeffs, ops) below
        return np.array([0.0]), [qml.operation.Tensor(qml.Identity(0))]

    xyz2pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

    coeffs, ops = zip(
        *[
            (
                coef,
                qml.operation.Tensor(*[xyz2pauli[q[1]](wires=q[0]) for q in term])
                if term
                else qml.operation.Tensor(qml.Identity(0))
                # example term: ((0,'X'), (2,'Z'), (3,'Y'))
            )
            for term, coef in qubit_operator.terms.items()
        ]
    )

    return np.real(np.array(coeffs)), list(ops)


def _terms_to_qubit_operator(coeffs, ops):
    r"""Converts a 2-tuple of complex coefficients and PennyLane operations to
    OpenFermion ``QubitOperator``.

    This function is the inverse of ``_qubit_operator_to_terms``.

    Args:
        coeffs (array[complex]):
            coefficients for each observable, same length as ops
        ops (Iterable[pennylane.operation.Observable]): List of PennyLane observables as
            Tensor products of Pauli observables

    Returns:
        QubitOperator: an instance of OpenFermion's ``QubitOperator``.
    """
    q_op = QubitOperator()
    for coeff, op in zip(coeffs, ops):

        # wire ids
        wires = op.wires

        # Pauli axis names, note s[-1] expects only 'Pauli{X,Y,Z}'
        pauli_names = [s[-1] for s in op.name]

        extra_obsvbs = set(op.name) - {"PauliX", "PauliY", "PauliZ", "Identity"}
        if extra_obsvbs != set():
            raise ValueError(
                "Expected only PennyLane observables PauliX/Y/Z or Identity, "
                + "but also got {}.".format(extra_obsvbs)
            )

        if op.name == ["Identity"] and wires == [0]:
            term_str = ""
        else:
            term_str = " ".join(
                ["{}{}".format(pauli, wire) for pauli, wire in zip(pauli_names, wires)]
            )

        # This is how one makes QubitOperator in OpenFermion
        q_op += coeff * QubitOperator(term_str)

    return q_op


def _qubit_operators_equivalent(openfermion_qubit_operator, pennylane_qubit_operator):
    r"""Checks equivalence between OpenFermion :class:`~.QubitOperator` and Pennylane  VQE
    ``Hamiltonian`` (Tensor product of Pauli matrices).

    Equality is based on OpenFermion :class:`~.QubitOperator`'s equality.

    Args:
        openfermion_qubit_operator (QubitOperator): OpenFermion qubit operator represented as
            a Pauli summation
        pennylane_qubit_operator (pennylane.Hamiltonian): PennyLane
            Hamiltonian object

    Returns:
        (bool): True if equivalent
    """
    coeffs, ops = pennylane_qubit_operator.terms
    return openfermion_qubit_operator == _terms_to_qubit_operator(coeffs, ops)


def convert_observable(qubit_observable):
    r"""Converts an OpenFermion :class:`~.QubitOperator` operator to a Pennylane VQE observable

    **Example usage**

    >>> h_of = decompose_hamiltonian('h2', './pyscf/sto-3g/')
    >>> h_pl = convert_observable(h_of)
    >>> h_pl.coeffs
    [-0.04207898+0.j  0.17771287+0.j  0.17771287+0.j -0.2427428 +0.j -0.2427428 +0.j  0.17059738+0.j
    0.04475014+0.j  0.04475014+0.j  0.04475014+0.j  0.04475014+0.j  0.12293305+0.j  0.16768319+0.j
    0.16768319+0.j  0.12293305+0.j  0.17627641+0.j]

    Args:
        qubit_observable (QubitOperator): Observable represented as an OpenFermion ``QubitOperator``

    Returns:
        (pennylane.Hamiltonian): Pennylane VQE observable. PennyLane :class:`~.Hamiltonian`
        represents any operator expressed as linear combinations of observables, e.g.,
        :math:`\sum_{k=0}^{N-1} c_k O_k`.
    """

    return Hamiltonian(*_qubit_operator_to_terms(qubit_observable))


def generate_hamiltonian(
    mol_name,
    mol_geo_file,
    mol_charge,
    multiplicity,
    basis_set,
    qc_package="pyscf",
    n_active_electrons=None,
    n_active_orbitals=None,
    mapping="jordan_wigner",
    outpath=".",
):  # pylint:disable=too-many-arguments
    r"""Generates the qubit Hamiltonian based on geometry and mean field electronic structure.

    An active space can be defined, otherwise the Hamiltonian is expanded in the full basis of
    Hartree-Fock (HF) molecular orbitals.

    **Example usage:**

    >>> H, n_qubits = generate_hamiltonian('h2', 'h2.xyz', 0, 1, 'sto-3g')
    >>> print(n_qubits)
    4
    >>> print(H)
    (-0.04207897647782188) [I0]
    + (0.17771287465139934) [Z0]
    + (0.1777128746513993) [Z1]
    + (-0.24274280513140484) [Z2]
    + (-0.24274280513140484) [Z3]
    + (0.17059738328801055) [Z0 Z1]
    + (0.04475014401535161) [Y0 X1 X2 Y3]
    + (-0.04475014401535161) [Y0 Y1 X2 X3]
    + (-0.04475014401535161) [X0 X1 Y2 Y3]
    + (0.04475014401535161) [X0 Y1 Y2 X3]
    + (0.12293305056183801) [Z0 Z2]
    + (0.1676831945771896) [Z0 Z3]
    + (0.1676831945771896) [Z1 Z2]
    + (0.12293305056183801) [Z1 Z3]
    + (0.176276408043196) [Z2 Z3]

    Args:
        mol_name (str): name of the molecule
        mol_geo_file (str): name of the file containing the geometry of the molecule
        mol_charge (int): net charge of the molecule
        multiplicity (int): multiplicity of the Hartree-Fock reference state
        basis_set (str): atomic Gaussian-type orbitals basis set. Basis set availability per
                element can be found `here
                <www.psicode.org/psi4manual/master/basissets_byelement.html>`_
        qc_package (str): quantum chemistry package (pyscf or psi4) used to solve the
                mean field electronic structure problem
        n_active_electrons (int): number of active electrons. If not specified, all electrons
                are considered to be active
        n_active_orbitals (int): number of active orbitals. If not specified, all orbitals
                are considered to be active
        mapping (str): the transformation (``'jordan_wigner'`` or ``'bravyi_kitaev'``) used to
                map the second-quantized electronic Hamiltonian to the qubit Hamiltonian
        outpath (str): path to the directory containing output files
    Returns:
        tuple[pennylane.Hamiltonian, int]: the fermionic-to-qubit transformed
        Hamiltonian and the number of qubits

     """

    geometry = read_structure(mol_geo_file, outpath)

    hf_data = meanfield_data(
        mol_name, geometry, mol_charge, multiplicity, basis_set, qc_package, outpath
    )

    docc_indices, active_indices = active_space(
        mol_name, hf_data, n_active_electrons, n_active_orbitals
    )

    h_of, nr_qubits = (
        decompose_hamiltonian(mol_name, hf_data, mapping, docc_indices, active_indices),
        2 * len(active_indices),
    )

    return convert_observable(h_of), nr_qubits


def sd_excitations(n_electrons, n_orbitals, delta_sz=0):
    r"""Generates single and double excitations from a Hartree-Fock (HF) reference state.

    The singly- and doubly-excited configurations are generated by acting with the operators
    :math:`\hat T_1` and :math:`\hat T_2` on the HF state:

    .. math:
        && \vert \Phi_\mathrm{S} \rangle = \hat{T}_1 \vert \mathrm{HF} \rangle = \sum_{r \in
        \mathrm{occ} \\ p \in \mathrm{virt}} \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF}
        \rangle \\
        && \vert \Phi_\mathrm{D} \rangle = \hat{T}_2 \vert \mathrm{HF} \rangle = \sum_{r>s \in
        \mathrm{occ} \\ p>q \in \mathrm{virt}} \hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s \vert \mathrm{HF} \rangle

	where the indices :math:`r, s` and :math:`p, q` run over the occupied (occ) and unoccupied,
	referred to as virtual (virt), molecular orbitals and :math:`\hat c` and
	:math:`\hat c^\dagger` are the electron annihilation and creation operators, respectively.

    **Example**

    >>> ph, pphh = sd_configs(2, 4, 0)
    >>> print(ph)
    [[0, 2], [1, 3]]
    >>> print(pphh)
    [[0, 1, 2, 3]]

    Args:
        n_electrons (int): number of active electrons
        n_orbitals (int): number of active orbitals
        delta_sz (int): optional argument to specify the spin-projection selection rule.
            For single excitations ``sz[p] - sz[r] = delta_sz``.
            For double excitations ``sz[p] + sz[p] - sz[r] - sz[s] = delta_sz``.
            ``sz`` is the single-particle state spin quantum number and ``delta_sz``, in the
            case of singles and doubles, can take the values :math:`0`, :math:`\pm 1`
            and :math:`\pm 2`.

    Returns:
        tuple(list, list): lists with the indices of the molecular orbitals
        involved in the single and double excitations
    """

    if not n_electrons > 0:
        raise ValueError(
            "The number of active electrons has to be greater than 0 \n"
            "Got n_electrons = {}".format(n_electrons)
        )

    if n_orbitals <= n_electrons:
        raise ValueError(
            "The number of active orbitals ({}) "
            "has to be greater than the number of active electrons ({}).".format(
                n_orbitals, n_electrons
            )
        )

    if delta_sz not in (0, 1, -1, 2, -2):
        raise ValueError(
            "Expected values for 'delta_sz' are 0, +/- 1 and +/- 2 but got ({}).".format(delta_sz)
        )

    # define the single-particle state spin quantum number 'sz'
    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(n_orbitals)])

    # nested list with the indices 'p, r' for each 1particle-1hole (ph) configuration
    ph = [
        [r, p]
        for r in range(n_electrons)
        for p in range(n_electrons, n_orbitals)
        if sz[p] - sz[r] == delta_sz
    ]

    # nested list with the indices 's, r, q, p' for each 2particle-2hole (pphh) configuration
    pphh = [
        [s, r, q, p]
        for s in range(n_electrons - 1)
        for r in range(s + 1, n_electrons)
        for q in range(n_electrons, n_orbitals - 1)
        for p in range(q + 1, n_orbitals)
        if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz
    ]

    return ph, pphh


def hf_state(n_electrons, m_spin_orbitals):
    r"""Generates the occupation-number vector representing the Hartree-Fock (HF)
    state of :math:`N` electrons in a basis of :math:`M` spin orbitals.

    The many-particle wave function in the HF approximation is a `Slater determinant
    <https://en.wikipedia.org/wiki/Slater_determinant>`_. In Fock space, a Slater determinant
    is represented by the occupation-number vector:

    .. math:
        \vert {\bf n} \rangle = \vert n_1, n_2, \dots, n_M \rangle,
        n_i = \left\lbrace \begin{array}{ll} 1 & i \leq N \\ 0 & i > N \end{array} \right.

    **Example**

    >>> init_state = hf_state(2, 6)
    >>> print(init_state)
    [1 1 0 0 0 0]

    Args:
        n_electrons (int): number of active electrons
        m_spin_orbitals (int): number of active **spin-orbitals**

    Returns:
        array: NumPy array containing the vector :math:`\vert {\bf n} \rangle`
    """

    if n_electrons <= 0:
        raise ValueError(
            "The number of active electrons has to be larger than zero; got 'n_electrons' = {}".format(
                n_electrons
            )
        )

    if n_electrons > m_spin_orbitals:
        raise ValueError(
            "The number of active orbitals cannot be smaller than the number of active electrons;"
            " got 'm_spin_orbitals'={} < 'n_electrons'={}".format(m_spin_orbitals, n_electrons)
        )

    hf_state_on = np.where(np.arange(m_spin_orbitals) < n_electrons, 1, 0)

    return np.array(hf_state_on)


def excitations_to_wires(ph_confs, pphh_confs, wires=None):

    r"""Map the indices representing the particle-hole configurations
    generated by the Coupled-Cluster excitation operator to the wires that
    the Unitary Coupled-Cluster Singles and Doubles (UCCSD) template will act on.

    **Example**

    >>> ph_confs = [[0, 2], [1, 3]]
    >>> pphh_confs = [[0, 1, 2, 3]]
    >>> ph, pphh = excitations_to_wires(ph_confs, pphh_confs)
    >>> print(ph)
    [[0, 1, 2], [1, 2, 3]]
    >>> print(pphh)
    [[[0, 1], [2, 3]]]

    >>> wires=['a0', 'b1', 'c2', 'd3']
    >>> ph, pphh = excitations_to_wires(ph_confs, pphh_confs, wires=wires)
    >>> print(ph)
    [['a0', 'b1', 'c2'], ['b1', 'c2', 'd3']]
    >>> print(pphh)
    [[['a0', 'b1'], ['c2', 'd3']]]

    Args:
        ph_confs (list[list[int]]): list of indices of the two qubits representing
            the 1particle-1hole (ph) configuration
            :math:`\vert ph \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF}\rangle`.
        pphh_confs (list[list[int]]): list of indices of the four qubits representing
            the 2particle-2hole (pphh) configuration
            :math:`\vert pphh \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger
            \hat{c}_r \hat{c}_s \vert \mathrm{HF}\rangle`. The indices :math:`r, s`
            and :math:`p, q` run over the occupied and virtual Hartree-Fock (HF)
            orbitals, respectively.
        wires (Iterable[Any]): Wires of the quantum device. If None, will use consecutive wires.

    Returns:
        tuple(list[list[Any]], list[list[list[Any]]]): lists with the sequence of wires
        the Unitary Coupled-Cluster Singles and Doubles (UCCSD) template will act on.
    """

    if (not ph_confs) and (not pphh_confs):
        raise ValueError(
            "'ph_confs' and 'pphh_confs' lists can not be both empty;\
            got ph_confs = {}, pphh_confs = {}".format(
                ph_confs, pphh_confs
            )
        )

    expected_shape = (2,)
    for ph_confs_ in ph_confs:
        if np.array(ph_confs_).shape != expected_shape:
            raise ValueError(
                "expected entries of 'ph_confs' to be of shape (2,); got {}".format(
                    np.array(ph_confs_).shape
                )
            )

    expected_shape = (4,)
    for pphh_confs_ in pphh_confs:
        if np.array(pphh_confs_).shape != expected_shape:
            raise ValueError(
                "expected entries of 'pphh_confs' to be of shape (4,); got {}".format(
                    np.array(pphh_confs_).shape
                )
            )

    max_idx = 0
    if ph_confs:
        max_idx = np.max(ph_confs)
    if pphh_confs:
        max_idx = max(np.max(pphh_confs), max_idx)

    if wires is None:
        wires = range(max_idx + 1)
    elif len(wires) != max_idx + 1:
        raise ValueError("Expected number of wires is {}; got {}".format(max_idx + 1, len(wires)))

    ph = []
    for r, p in ph_confs:
        ph_wires = [wires[i] for i in range(r, p + 1)]
        ph.append(ph_wires)

    pphh = []
    for s, r, q, p in pphh_confs:
        pphh1_wires = [wires[i] for i in range(s, r + 1)]
        pphh2_wires = [wires[i] for i in range(q, p + 1)]
        pphh.append([pphh1_wires, pphh2_wires])

    return ph, pphh


__all__ = [
    "read_structure",
    "meanfield",
    "active_space",
    "decompose_molecular_hamiltonian",
    "_qubit_operator_to_terms",
    "_terms_to_qubit_operator",
    "_qubit_operators_equivalent",
    "convert_observable",
    "generate_hamiltonian",
    "sd_excitations",
    "hf_state",
    "excitations_to_wires",
]
