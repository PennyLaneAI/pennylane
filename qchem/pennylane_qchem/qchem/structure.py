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
from pennylane.wires import Wires


def _proc_wires(wires, n_wires=None):
    r"""
    Checks and processes custom user wire mapping into a consistent, direction-free, Wires format.
    Used for converting between OpenFermion qubit numbering and Pennylane wire labels.

    Since OpenFermion's quibit numbering is always consecutive int, simple iterable types such as
    list, tuple, or Wires can be used to specify the qubit<->wire mapping with indices acting as
    qubits. Dict can also be used as a mapping, but does not provide any advantage over lists other
    than the ability to do partial mapping/permutation in the qubit->wire direction.

    It is recommended pass Wires/list/tuple `wires` since it's direction-free, i.e. the same `wires`
    argument can be used to convert both ways between OpenFermion and Pennylane. Only use dict for
    partial or unordered mapping.

    **Example usage:**

    >>> # consec int wires if no wires mapping provided, ie. identity map: 0<->0, 1<->1, 2<->2
    >>> _proc_wires(None, 3)
    <Wires = [0, 1, 2]>

    >>> # List as mapping, qubit indices with wire label values: 0<->w0, 1<->w1, 2<->w2
    >>> _proc_wires(['w0','w1','w2'])
    <Wires = ['w0', 'w1', 'w2']>

    >>> # Wires as mapping, qubit indices with wire label values: 0<->w0, 1<->w1, 2<->w2
    >>> _proc_wires(Wires(['w0', 'w1', 'w2']))
    <Wires = ['w0', 'w1', 'w2']>

    >>> # Dict as partial mapping, int qubits keys to wire label values: 0->w0, 1 unchanged, 2->w2
    >>> _proc_wires({0:'w0',2:'w2'})
    <Wires = ['w0', 1, 'w2']>

    >>> # Dict as mapping, wires label keys to consec int qubit values: w2->2, w0->0, w1->1
    >>> _proc_wires({'w2':2, 'w0':0, 'w1':1})
    <Wires = ['w0', 'w1', 'w2']>


    Args:
        wires (Wires, list, tuple, dict): User wire labels or mapping for Pennylane ansatz.
            For types Wires, list, or tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) or
            consecutive-int-valued dict (for wire-to-qubit conversion) is accepted.
            If None, will be set to consecutive int based on ``n_wires``.
        n_wires (int): Number of wires used if known. If None, will infer from ``wires``; if
            ``wires`` is not available, will be set to 1. Defaults to None.

    Returns:
        Wires: Cleaned wire mapping with indices corresponding to qubits and values
            corresponding to wire labels.
    """

    # infer from wires, or assume 1 if wires is not of accepted types.
    if n_wires is None:
        n_wires = len(wires) if isinstance(wires, (Wires, list, tuple, dict)) else 1

    # defaults to no mapping.
    if wires is None:
        return Wires(range(n_wires))

    if isinstance(wires, (Wires, list, tuple)):
        # does not care about the tail if more wires are provided than n_wires.
        wires = Wires(wires[:n_wires])

    elif isinstance(wires, dict):

        if all([isinstance(w, int) for w in wires.keys()]):
            # Assuming keys are taken from consecutive int wires. Allows for partial mapping.
            n_wires = max(wires) + 1
            labels = list(range(n_wires))  # used for completing potential partial mapping.
            for k, v in wires.items():
                if k < n_wires:
                    labels[k] = v
            wires = Wires(labels)
        elif set(range(n_wires)).issubset(set(wires.values())):
            # Assuming values are consecutive int wires (up to n_wires, ignores the rest).
            # Does NOT allow for partial mapping.
            wires = {v: k for k, v in wires.items()}  # flip for easy indexing
            wires = Wires([wires[i] for i in range(n_wires)])
        else:
            raise ValueError("Expected only int-keyed or consecutive int-valued dict for `wires`")

    else:
        raise ValueError(
            "Expected type Wires, list, tuple, or dict for `wires`, got {}".format(type(wires))
        )

    if len(wires) != n_wires:
        # check length consistency when all checking and cleaning are done.
        raise ValueError(
            "Length of `wires` ({}) does not match `n_wires` ({})".format(len(wires), n_wires)
        )

    return wires


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


def _qubit_operator_to_terms(qubit_operator, wires=None):
    r"""Converts OpenFermion ``QubitOperator`` to a 2-tuple of coefficients and
    PennyLane Pauli observables.

    **Example usage:**

    >>> q_op = 0.1*QubitOperator('X0') + 0.2*QubitOperator('Y0 Z2')
    >>> q_op
    0.1 [X0] +
    0.2 [Y0 Z2]
    >>> _qubit_operator_to_terms(q_op, wires=['w0','w1','w2','extra_wire'])
    (array([0.1, 0.2]), [Tensor(PauliX(wires=['w0'])), Tensor(PauliY(wires=['w0']), PauliZ(wires=['w2']))])

    Args:
        qubit_operator (QubitOperator): Fermionic-to-qubit transformed operator in terms of
            Pauli matrices
        wires (Wires, list, tuple, dict): Custom wire mapping for connecting to Pennylane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identiy map. Defaults to None.

    Returns:
        tuple[array[float], Iterable[pennylane.operation.Observable]]: coefficients and their
        corresponding PennyLane observables in the Pauli basis
    """
    n_wires = (
        1 + max([max([i for i, _ in t]) if t else 1 for t in qubit_operator.terms])
        if qubit_operator.terms
        else 1
    )
    wires = _proc_wires(wires, n_wires=n_wires)

    if not qubit_operator.terms:  # added since can't unpack empty zip to (coeffs, ops) below
        return np.array([0.0]), [qml.operation.Tensor(qml.Identity(wires[0]))]

    xyz2pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

    coeffs, ops = zip(
        *[
            (
                coef,
                qml.operation.Tensor(*[xyz2pauli[q[1]](wires=wires[q[0]]) for q in term])
                if term
                else qml.operation.Tensor(qml.Identity(wires[0]))
                # example term: ((0,'X'), (2,'Z'), (3,'Y'))
            )
            for term, coef in qubit_operator.terms.items()
        ]
    )

    return np.real(np.array(coeffs)), list(ops)


def _terms_to_qubit_operator(coeffs, ops, wires=None):
    r"""Converts a 2-tuple of complex coefficients and PennyLane operations to
    OpenFermion ``QubitOperator``.

    This function is the inverse of ``_qubit_operator_to_terms``.

    **Example usage:**

    >>> coeffs = np.array([0.1, 0.2])
    >>> ops = [
    ...     qml.operation.Tensor(qml.PauliX(wires=['w0'])),
    ...     qml.operation.Tensor(qml.PauliY(wires=['w0']), qml.PauliZ(wires=['w2']))
    ... ]
    >>> _terms_to_qubit_operator(coeffs, ops, wires=Wires(['w0', 'w1', 'w2']))
    0.1 [X0] +
    0.2 [Y0 Z2]

    Args:
        coeffs (array[complex]):
            coefficients for each observable, same length as ops
        ops (Iterable[pennylane.operation.Observable]): List of PennyLane observables as
            Tensor products of Pauli observables
        wires (Wires, list, tuple, dict): Custom wire mapping for translating from Pennylane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only consecutive-int-valued dict (for wire-to-qubit conversion) is
            accepted. If None, will use identiy map. Defaults to None.

    Returns:
        QubitOperator: an instance of OpenFermion's ``QubitOperator``.
    """
    all_wires = Wires.all_wires([op.wires for op in ops], sort=True)
    # n_all_wires = len(all_wires)
    if wires is not None:
        qubit_indexed_wires = _proc_wires(wires,)
        if not set(all_wires).issubset(set(qubit_indexed_wires)):
            raise ValueError("Supplied `wires` does not cover all wires defined in `ops`.")
    else:
        qubit_indexed_wires = all_wires

    q_op = QubitOperator()
    for coeff, op in zip(coeffs, ops):

        # Pauli axis names, note s[-1] expects only 'Pauli{X,Y,Z}'
        pauli_names = [s[-1] for s in op.name]

        extra_obsvbs = set(op.name) - {"PauliX", "PauliY", "PauliZ", "Identity"}
        if extra_obsvbs != set():
            raise ValueError(
                "Expected only PennyLane observables PauliX/Y/Z or Identity, "
                + "but also got {}.".format(extra_obsvbs)
            )

        # if op.name == ["Identity"] and wires == [0]:
        if op.name == ["Identity"] and len(op.wires) == 1:
            term_str = ""
        else:
            term_str = " ".join(
                [
                    "{}{}".format(pauli, qubit_indexed_wires.index(wire))
                    for pauli, wire in zip(pauli_names, op.wires)
                ]
            )

        # This is how one makes QubitOperator in OpenFermion
        q_op += coeff * QubitOperator(term_str)

    return q_op


def _qubit_operators_equivalent(openfermion_qubit_operator, pennylane_qubit_operator, wires=None):
    r"""Checks equivalence between OpenFermion :class:`~.QubitOperator` and Pennylane  VQE
    ``Hamiltonian`` (Tensor product of Pauli matrices).

    Equality is based on OpenFermion :class:`~.QubitOperator`'s equality.

    Args:
        openfermion_qubit_operator (QubitOperator): OpenFermion qubit operator represented as
            a Pauli summation
        pennylane_qubit_operator (pennylane.Hamiltonian): PennyLane
            Hamiltonian object
        wires (Wires, list, tuple, dict): Custom wire mapping for connecting to Pennylane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identiy map. Defaults to None.

    Returns:
        (bool): True if equivalent
    """
    coeffs, ops = pennylane_qubit_operator.terms
    return openfermion_qubit_operator == _terms_to_qubit_operator(coeffs, ops, wires=wires)


def convert_observable(qubit_observable, wires=None):
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
        wires (Wires, list, tuple, dict): Custom wire mapping for connecting to Pennylane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identiy map. Defaults to None.

    Returns:
        (pennylane.Hamiltonian): Pennylane VQE observable. PennyLane :class:`~.Hamiltonian`
        represents any operator expressed as linear combinations of observables, e.g.,
        :math:`\sum_{k=0}^{N-1} c_k O_k`.
    """

    return Hamiltonian(*_qubit_operator_to_terms(qubit_observable, wires=wires))


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
    wires=None,
):  # pylint:disable=too-many-arguments
    r"""Generates the qubit Hamiltonian based on geometry and mean field electronic structure.

    An active space can be defined, otherwise the Hamiltonian is expanded in the full basis of
    Hartree-Fock (HF) molecular orbitals.

    **Example usage:**

    >>> H, n_qubits = generate_hamiltonian('h2', 'h2.xyz', 0, 1, 'sto-3g', wires=['w0','w1','w2','w3'])
    >>> print(n_qubits)
    4
    >>> print(H)
    (-0.04207897647782188) [Iw0]
    + (0.17771287465139934) [Zw0]
    + (0.1777128746513993) [Zw1]
    + (-0.24274280513140484) [Zw2]
    + (-0.24274280513140484) [Zw3]
    + (0.17059738328801055) [Zw0 Zw1]
    + (0.04475014401535161) [Yw0 Xw1 Xw2 Yw3]
    + (-0.04475014401535161) [Yw0 Yw1 Xw2 Xw3]
    + (-0.04475014401535161) [Xw0 Xw1 Yw2 Yw3]
    + (0.04475014401535161) [Xw0 Yw1 Yw2 Xw3]
    + (0.12293305056183801) [Zw0 Zw2]
    + (0.1676831945771896) [Zw0 Zw3]
    + (0.1676831945771896) [Zw1 Zw2]
    + (0.12293305056183801) [Zw1 Zw3]
    + (0.176276408043196) [Zw2 Zw3]

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
        wires (Wires, list, tuple, dict): Custom wire mapping for connecting to Pennylane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted for
            partial mapping.
            If None, will use identiy map. Defaults to None.
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

    return convert_observable(h_of, wires=wires), nr_qubits


def excitations(electrons, orbitals, delta_sz=0):
    r"""Generates single and double excitations from a Hartree-Fock reference state.

    Single and double excitations can be generated by acting with the operators
    :math:`\hat T_1` and :math:`\hat T_2` on the Hartree-Fock reference state:

    .. math::

        && \hat{T}_1 = \sum_{r \in \mathrm{occ} \\ p \in \mathrm{unocc}}
        \hat{c}_p^\dagger \hat{c}_r \\
        && \hat{T}_2 = \sum_{r>s \in \mathrm{occ} \\ p>q \in
        \mathrm{unocc}} \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s.


    In the equations above the indices :math:`r, s` and :math:`p, q` run over the
    occupied (occ) and unoccupied (unocc) *spin-orbitals* and :math:`\hat c` and
    :math:`\hat c^\dagger` are the electron annihilation and creation operators,
    respectively.

    |

    .. figure:: ../../_static/qchem/sd_excitations.png
        :align: center
        :width: 80%

    |

    Args:
        electrons (int): Number of electrons. If an active space is defined, this
            is the number of active electrons.
        orbitals (int): Number of spin orbitals. If an active space is defined,
            this is the number of active spin-orbitals.
        delta_sz (int): Specifies the selection rules ``sz[p] - sz[r] = delta_sz`` and
            ``sz[p] + sz[p] - sz[r] - sz[s] = delta_sz`` for the spin-projection ``sz`` of
            the orbitals involved in the single and double excitations, respectively.
            ``delta_sz`` can take the values :math:`0`, :math:`\pm 1` and :math:`\pm 2`.

    Returns:
        tuple(list, list): lists with the indices of the spin orbitals involved in the
        single and double excitations

    **Example**

    >>> electrons = 2
    >>> orbitals = 4
    >>> singles, doubles = excitations(electrons, orbitals)
    >>> print(singles)
    [[0, 2], [1, 3]]
    >>> print(doubles)
    [[0, 1, 2, 3]]
    """

    if not electrons > 0:
        raise ValueError(
            "The number of active electrons has to be greater than 0 \n"
            "Got n_electrons = {}".format(electrons)
        )

    if orbitals <= electrons:
        raise ValueError(
            "The number of active spin-orbitals ({}) "
            "has to be greater than the number of active electrons ({}).".format(
                orbitals, electrons
            )
        )

    if delta_sz not in (0, 1, -1, 2, -2):
        raise ValueError(
            "Expected values for 'delta_sz' are 0, +/- 1 and +/- 2 but got ({}).".format(delta_sz)
        )

    # define the spin projection 'sz' of the single-particle states
    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    singles = [
        [r, p]
        for r in range(electrons)
        for p in range(electrons, orbitals)
        if sz[p] - sz[r] == delta_sz
    ]

    doubles = [
        [s, r, q, p]
        for s in range(electrons - 1)
        for r in range(s + 1, electrons)
        for q in range(electrons, orbitals - 1)
        for p in range(q + 1, orbitals)
        if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz
    ]

    return singles, doubles


def hf_state(electrons, orbitals):
    r"""Generates the occupation-number vector representing the Hartree-Fock state.

    The many-particle wave function in the Hartree-Fock (HF) approximation is a `Slater determinant
    <https://en.wikipedia.org/wiki/Slater_determinant>`_. In Fock space, a Slater determinant
    for :math:`N` electrons is represented by the occupation-number vector:

    .. math::

        \vert {\bf n} \rangle = \vert n_1, n_2, \dots, n_\mathrm{orbs} \rangle,
        n_i = \left\lbrace \begin{array}{ll} 1 & i \leq N \\ 0 & i > N \end{array} \right.,

    where :math:`n_i` indicates the occupation of the :math:`i`-th orbital.

    Args:
        electrons (int): Number of electrons. If an active space is defined, this
            is the number of active electrons.
        orbitals (int): Number of spin orbitals. If an active space is defined,
            this is the number of active spin-orbitals.

    Returns:
        array: NumPy array containing the vector :math:`\vert {\bf n} \rangle`

    **Example**

    >>> state = hf_state(2, 6)
    >>> print(state)
    [1 1 0 0 0 0]
    """

    if electrons <= 0:
        raise ValueError(
            "The number of active electrons has to be larger than zero; got 'electrons' = {}".format(
                electrons
            )
        )

    if electrons > orbitals:
        raise ValueError(
            "The number of active orbitals cannot be smaller than the number of active electrons;"
            " got 'orbitals'={} < 'electrons'={}".format(orbitals, electrons)
        )

    state = np.where(np.arange(orbitals) < electrons, 1, 0)

    return np.array(state)


def excitations_to_wires(singles, doubles, wires=None):
    r"""Map the indices representing the single and double excitations
    generated with the function :func:`~.excitations` to the wires that
    the Unitary Coupled-Cluster (UCCSD) template will act on.

    Args:
        singles (list[list[int]]): list with the indices ``r``, ``p`` of the two qubits
            representing the single excitation
            :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF}\rangle`
        doubles (list[list[int]]): list with the indices ``s``, ``r``, ``q``, ``p`` of the four
            qubits representing the double excitation
            :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger
            \hat{c}_r \hat{c}_s \vert \mathrm{HF}\rangle`
        wires (Iterable[Any]): Wires of the quantum device. If None, will use consecutive wires.

    The indices :math:`r, s` and :math:`p, q` in these lists correspond, respectively, to the
    occupied and virtual orbitals involved in the generated single and double excitations.

    Returns:
        tuple[list[list[Any]], list[list[list[Any]]]]: lists with the sequence of wires,
        resulting from the single and double excitations, that the Unitary Coupled-Cluster
        (UCCSD) template will act on.

    **Example**

    >>> singles = [[0, 2], [1, 3]]
    >>> doubles = [[0, 1, 2, 3]]
    >>> singles_wires, doubles_wires = excitations_to_wires(singles, doubles)
    >>> print(single_wires)
    [[0, 1, 2], [1, 2, 3]]
    >>> print(doubles_wires)
    [[[0, 1], [2, 3]]]

    >>> wires=['a0', 'b1', 'c2', 'd3']
    >>> singles_wires, doubles_wires = excitations_to_wires(singles, doubles, wires=wires)
    >>> print(singles_wires)
    [['a0', 'b1', 'c2'], ['b1', 'c2', 'd3']]
    >>> print(doubles_wires)
    [[['a0', 'b1'], ['c2', 'd3']]]
    """

    if (not singles) and (not doubles):
        raise ValueError(
            "'singles' and 'doubles' lists can not be both empty;\
            got singles = {}, doubles = {}".format(
                singles, doubles
            )
        )

    expected_shape = (2,)
    for single_ in singles:
        if np.array(single_).shape != expected_shape:
            raise ValueError(
                "Expected entries of 'singles' to be of shape (2,); got {}".format(
                    np.array(single_).shape
                )
            )

    expected_shape = (4,)
    for double_ in doubles:
        if np.array(double_).shape != expected_shape:
            raise ValueError(
                "Expected entries of 'doubles' to be of shape (4,); got {}".format(
                    np.array(double_).shape
                )
            )

    max_idx = 0
    if singles:
        max_idx = np.max(singles)
    if doubles:
        max_idx = max(np.max(doubles), max_idx)

    if wires is None:
        wires = range(max_idx + 1)
    elif len(wires) != max_idx + 1:
        raise ValueError("Expected number of wires is {}; got {}".format(max_idx + 1, len(wires)))

    singles_wires = []
    for r, p in singles:
        s_wires = [wires[i] for i in range(r, p + 1)]
        singles_wires.append(s_wires)

    doubles_wires = []
    for s, r, q, p in doubles:
        d1_wires = [wires[i] for i in range(s, r + 1)]
        d2_wires = [wires[i] for i in range(q, p + 1)]
        doubles_wires.append([d1_wires, d2_wires])

    return singles_wires, doubles_wires


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
    "excitations",
    "hf_state",
    "excitations_to_wires",
]
