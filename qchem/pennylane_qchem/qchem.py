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
"""The PennyLane quantum chemistry package. Supports OpenFermion, PySCF,
and Psi4 for quantum chemistry calculations using PennyLane."""
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


def meanfield_data(
    mol_name, geometry, charge, multiplicity, basis, qc_package="pyscf", outpath="."
):  # pylint: disable=too-many-arguments
    r"""Launches the meanfield (Hartree-Fock) electronic structure calculation.

    Also builds the path to the directory containing the input data file for quantum simulations.
    The path to the hdf5-formatted file is ``os.path.join(outpath, qc_package, basis)``.

    **Example usage:**

    >>> geometry = read_structure('h2_ref.xyz')
    >>> meanfield_data('h2', geometry, 0, 1, 'sto-3g', qc_package='pyscf')
    ./pyscf/sto-3g

    Args:
        mol_name (str): name of the molecule
        geometry (list): list containing the symbol and Cartesian coordinates for each atom
        charge (int): net charge of the molecule
        multiplicity (int): spin multiplicity based on the number of unpaired electrons
            in the Hartree-Fock state
        basis (str): atomic basis set. Basis set availability per element can be found
            `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx
            -basiselement>`_
        qc_package (str): quantum chemistry package used to solve Hartree-Fock equations.
            Either ``'pyscf'`` or ``'psi4'`` can be used
        outpath (str): path to ouput directory

    Returns:
        str: path to the directory containing the file with the Hartree-Fock electronic structure
    """

    qc_package = qc_package.strip().lower()

    if qc_package not in ("psi4", "pyscf"):
        qc_package_error_message = (
            "Integration with quantum chemistry package '{}' is not available. \n Please set"
            " 'qc_package' to 'pyscf' or 'psi4'.".format(qc_package)
        )
        raise TypeError(qc_package_error_message)

    qcp_dir = os.path.join(outpath.strip(), qc_package)
    path_to_hf_data = os.path.join(qcp_dir, basis.strip())

    if not os.path.isdir(qcp_dir):
        os.mkdir(qcp_dir)
        os.mkdir(path_to_hf_data)
    elif not os.path.isdir(path_to_hf_data):
        os.mkdir(path_to_hf_data)

    molecule = MolecularData(
        geometry,
        basis,
        multiplicity,
        charge,
        filename=os.path.join(path_to_hf_data, mol_name.strip()),
    )

    if qc_package == "psi4":
        run_psi4(molecule, run_scf=1, verbose=0, tolerate_error=1)

    if qc_package == "pyscf":
        run_pyscf(molecule, run_scf=1, verbose=0)

    return path_to_hf_data


def active_space(mol_name, hf_data, n_active_electrons=None, n_active_orbitals=None):
    r"""Builds the active space by partitioning the set of Hartree-Fock molecular orbitals.

    **Example usage:**

    >>> d_occ_orbitals, active_orbitals = active_space('lih', './pyscf/sto-3g',
    n_active_electrons=2, n_active_orbitals=2)
    >>> d_occ_indices  # doubly-occupied molecular orbitals
    [0]
    >>> active_indices # active molecular orbitals
    [1, 2]
    >>> 2*len(active_indices) # number of qubits required for simulation
    4

    .. note::
        The number of active *spin*-orbitals ``2*n_active_orbitals`` determines the number of
        qubits to perform quantum simulations of the electronic structure of the molecule.

    Args:
        mol_name (str): name of the molecule
        hf_data (str): path to the directory containing the file with the Hartree-Fock electronic
            structure
        n_active_electrons (int): Optional argument to specify the number of active electrons.
            If not specified, all electrons are treated as active
        n_active_orbitals (int): Optional argument to specify the number of active orbitals.
            If not specified, all orbitals considered active

    Returns:
        tuple: lists of indices for doubly-occupied and active orbitals
    """
    # pylint: disable=too-many-branches
    molecule = MolecularData(filename=os.path.join(hf_data.strip(), mol_name.strip()))

    if n_active_electrons is None:
        n_docc_orbitals = 0
        docc_indices = []
    else:
        if n_active_electrons <= 0:
            raise ValueError(
                "The number of active electrons ({}) "
                "has to be greater than 0.".format(n_active_electrons)
            )

        if n_active_electrons > molecule.n_electrons:
            raise ValueError(
                "The number of active electrons ({}) "
                "can not be greater than the total "
                "number of electrons ({}).".format(n_active_electrons, molecule.n_electrons)
            )

        if n_active_electrons < molecule.multiplicity - 1:
            raise ValueError(
                "For a reference state with multiplicity {}, "
                "the number of active electrons ({}) should be "
                "greater than or equal to {}.".format(
                    molecule.multiplicity, n_active_electrons, molecule.multiplicity - 1
                )
            )

        if molecule.multiplicity % 2 == 1:
            if n_active_electrons % 2 != 0:
                raise ValueError(
                    "For a reference state with multiplicity {}, "
                    "the number of active electrons ({}) should be even.".format(
                        molecule.multiplicity, n_active_electrons
                    )
                )
        else:
            if n_active_electrons % 2 != 1:
                raise ValueError(
                    "For a reference state with multiplicity {}, "
                    "the number of active electrons ({}) should be odd.".format(
                        molecule.multiplicity, n_active_electrons
                    )
                )

        n_docc_orbitals = (molecule.n_electrons - n_active_electrons) // 2
        docc_indices = list(range(n_docc_orbitals))

    if n_active_orbitals is None:
        active_indices = list(range(n_docc_orbitals, molecule.n_orbitals))
    else:
        if n_active_orbitals <= 0:
            raise ValueError(
                "The number of active orbitals ({}) "
                "has to be greater than 0.".format(n_active_orbitals)
            )

        if n_docc_orbitals + n_active_orbitals > molecule.n_orbitals:
            raise ValueError(
                "The number of doubly occupied orbitals ({}) + "
                "the number of active orbitals ({}) can not be "
                "greater than the number of molecular orbitals ({})".format(
                    n_docc_orbitals, n_active_orbitals, molecule.n_orbitals
                )
            )

        homo = (molecule.n_electrons + molecule.multiplicity - 1) / 2
        if n_docc_orbitals + n_active_orbitals <= homo:
            raise ValueError(
                "For n_active_orbitals={}, there are no virtual orbitals "
                "in the active space.".format(n_active_orbitals)
            )

        active_indices = list(range(n_docc_orbitals, n_docc_orbitals + n_active_orbitals))

    return docc_indices, active_indices


def decompose_hamiltonian(
    mol_name, hf_data, mapping="jordan_wigner", docc_mo_indices=None, active_mo_indices=None
):
    r"""Decomposes the electronic Hamiltonian into a linear combination of Pauli operators using
    OpenFermion tools.

    **Example usage:**

    >>> decompose_hamiltonian('h2', './pyscf/sto-3g/', mapping='bravyi_kitaev')
    (-0.04207897696293986+0j) [] + (0.04475014401986122+0j) [X0 Z1 X2] +
    (0.04475014401986122+0j) [X0 Z1 X2 Z3] +(0.04475014401986122+0j) [Y0 Z1 Y2] +
    (0.04475014401986122+0j) [Y0 Z1 Y2 Z3] +(0.17771287459806262+0j) [Z0] +
    (0.17771287459806265+0j) [Z0 Z1] +(0.1676831945625423+0j) [Z0 Z1 Z2] +
    (0.1676831945625423+0j) [Z0 Z1 Z2 Z3] +(0.12293305054268105+0j) [Z0 Z2] +
    (0.12293305054268105+0j) [Z0 Z2 Z3] +(0.1705973832722409+0j) [Z1] +
    (-0.2427428049645989+0j) [Z1 Z2 Z3] +(0.1762764080276107+0j) [Z1 Z3] +
    (-0.2427428049645989+0j) [Z2]

    Args:
        mol_name (str): name of the molecule
        hf_data (str): path to the directory containing the file with the Hartree-Fock
            electronic structure
        mapping (str): optional argument to specify the fermion-to-qubit mapping
            Input values can be ``'jordan_wigner'`` or ``'bravyi_kitaev'``
        docc_mo_indices (list): indices of doubly-occupied molecular orbitals, i.e.,
            the orbitals that are not correlated in the many-body wave function
        active_mo_indices (list): indices of active molecular orbitals, i.e., the orbitals used to
            build the correlated many-body wave function

    Returns:
        transformed_operator: instance of the QubitOperator class representing the electronic
        Hamiltonian
    """

    # loading HF data from a hdf5 file
    molecule = MolecularData(filename=os.path.join(hf_data.strip(), mol_name.strip()))

    # getting the terms entering the second-quantized Hamiltonian
    terms_molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=docc_mo_indices, active_indices=active_mo_indices
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


__all__ = [
    "read_structure",
    "meanfield_data",
    "active_space",
    "decompose_hamiltonian",
    "_qubit_operator_to_terms",
    "_terms_to_qubit_operator",
    "_qubit_operators_equivalent",
    "convert_observable",
    "generate_hamiltonian",
    "sd_excitations",
    "hf_state",
]
