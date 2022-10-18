# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the functions needed for tapering qubits using symmetries.
"""
# pylint: disable=unnecessary-lambda

import functools
import itertools
import autograd.numpy as anp
import scipy
import numpy
import pennylane as qml
from pennylane import numpy as np
from pennylane.qchem.observable_hf import simplify, jordan_wigner
from pennylane.wires import Wires


def _binary_matrix(terms, num_qubits, wire_map=None):
    r"""Get a binary matrix representation of the Hamiltonian where each row corresponds to a
    Pauli term, which is represented by a concatenation of Z and X vectors.

    Args:
        terms (Iterable[Observable]): operators defining the Hamiltonian
        num_qubits (int): number of wires required to define the Hamiltonian
        wire_map (dict): dictionary containing all wire labels used in the Pauli words as keys, and
            unique integer labels as their values

    Returns:
        array[int]: binary matrix representation of the Hamiltonian of shape
        :math:`len(terms) \times 2*num_qubits`

    **Example**

    >>> wire_map = {'a':0, 'b':1, 'c':2, 'd':3}
    >>> terms = [qml.PauliZ(wires=['a']) @ qml.PauliX(wires=['b']),
    ...          qml.PauliZ(wires=['a']) @ qml.PauliY(wires=['c']),
    ...          qml.PauliX(wires=['a']) @ qml.PauliY(wires=['d'])]
    >>> _binary_matrix(terms, 4, wire_map=wire_map)
    array([[1, 0, 0, 0, 0, 1, 0, 0],
           [1, 0, 1, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 1, 0, 0, 1]])
    """
    if wire_map is None:
        all_wires = qml.wires.Wires.all_wires([term.wires for term in terms], sort=True)
        wire_map = {i: c for c, i in enumerate(all_wires)}

    binary_matrix = np.zeros((len(terms), 2 * num_qubits), dtype=int)
    for idx, term in enumerate(terms):
        ops, wires = term.name, term.wires
        if len(wires) == 1:
            ops = [ops]
        for op, wire in zip(ops, wires):
            if op in ["PauliX", "PauliY"]:
                binary_matrix[idx][wire_map[wire] + num_qubits] = 1
            if op in ["PauliZ", "PauliY"]:
                binary_matrix[idx][wire_map[wire]] = 1

    return binary_matrix


def _reduced_row_echelon(binary_matrix):
    r"""Returns the reduced row echelon form (RREF) of a matrix in a binary finite field :math:`\mathbb{Z}_2`.

    Args:
        binary_matrix (array[int]): binary matrix representation of the Hamiltonian
    Returns:
        array[int]: reduced row-echelon form of the given `binary_matrix`

    **Example**

    >>> binary_matrix = np.array([[1, 0, 0, 0, 0, 1, 0, 0],
    ...                           [1, 0, 1, 0, 0, 0, 1, 0],
    ...                           [0, 0, 0, 1, 1, 0, 0, 1]])
    >>> _reduced_row_echelon(binary_matrix)
    array([[1, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 1, 1, 0],
           [0, 0, 0, 1, 1, 0, 0, 1]])
    """
    rref_mat = binary_matrix.copy()
    shape = rref_mat.shape
    icol = 0

    for irow in range(shape[0]):

        while icol < shape[1] and not rref_mat[irow][icol]:

            # get the nonzero indices in the remainder of column icol
            non_zero_idx = rref_mat[irow:, icol].nonzero()[0]

            if len(non_zero_idx) == 0:  # if remainder of column icol is all zero
                icol += 1
            else:
                # find value and index of largest element in remainder of column icol
                krow = irow + non_zero_idx[0]

                # swap rows krow and irow
                rref_mat[irow, icol:], rref_mat[krow, icol:] = (
                    rref_mat[krow, icol:].copy(),
                    rref_mat[irow, icol:].copy(),
                )
        if icol < shape[1] and rref_mat[irow][icol]:

            # store remainder right hand side columns of the pivot row irow
            rpvt_cols = rref_mat[irow, icol:].copy()

            # get the column icol and set its irow element to 0 to avoid XORing pivot row with itself
            currcol = rref_mat[:, icol].copy()
            currcol[irow] = 0

            # XOR the right hand side of the pivot row irow with all of the other rows
            rref_mat[:, icol:] ^= np.outer(currcol, rpvt_cols)
            icol += 1

    return rref_mat.astype(int)


def _kernel(binary_matrix):
    r"""Computes the kernel of a binary matrix on the binary finite field :math:`\mathbb{Z}_2`.

    Args:
        binary_matrix (array[int]): binary matrix representation of the Hamiltonian
    Returns:
        array[int]: nullspace of the `binary_matrix` where each row corresponds to a
        basis vector in the nullspace

    **Example**

    >>> binary_matrix = np.array([[1, 0, 0, 0, 0, 1, 0, 0],
    ...                          [0, 0, 1, 1, 1, 1, 1, 1],
    ...                          [0, 0, 0, 1, 1, 0, 0, 1]])
    >>> _kernel(binary_matrix)
    array([[0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0, 0],
           [1, 0, 1, 0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0, 0, 1, 0],
           [0, 0, 1, 1, 0, 0, 0, 1]])
    """
    # Get the columns with and without pivots
    pivots = (binary_matrix.T != 0).argmax(axis=0)
    nonpivots = np.setdiff1d(range(len(binary_matrix[0])), pivots)

    # Initialize the nullspace
    null_vector = np.zeros((binary_matrix.shape[1], len(nonpivots)), dtype=int)
    null_vector[nonpivots, np.arange(len(nonpivots))] = 1

    # Fill up the nullspace vectors from the binary matrix
    null_vector_indices = np.ix_(pivots, np.arange(len(nonpivots)))
    binary_vector_indices = np.ix_(np.arange(len(pivots)), nonpivots)
    null_vector[null_vector_indices] = -binary_matrix[binary_vector_indices] % 2

    nullspace = null_vector.T
    return nullspace


def symmetry_generators(h):
    r"""Compute the generators :math:`\{\tau_1, \ldots, \tau_k\}` for a Hamiltonian over the binary
    field :math:`\mathbb{Z}_2`.

    These correspond to the generator set of the :math:`\mathbb{Z}_2`-symmetries present
    in the Hamiltonian as given in `arXiv:1910.14644 <https://arxiv.org/abs/1910.14644>`_.

    Args:
        h (Hamiltonian): Hamiltonian for which symmetries are to be generated to perform tapering

    Returns:
        list[Hamiltonian]: list of generators of symmetries, taus, for the Hamiltonian

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    >>> t = symmetry_generators(H)
    >>> t
    [<Hamiltonian: terms=1, wires=[0, 1]>,
     <Hamiltonian: terms=1, wires=[0, 2]>,
     <Hamiltonian: terms=1, wires=[0, 3]>]
    >>> print(t[0])
    (1.0) [Z0 Z1]
    """
    num_qubits = len(h.wires)

    # Generate binary matrix for qubit_op
    binary_matrix = _binary_matrix(h.ops, num_qubits)

    # Get reduced row echelon form of binary matrix
    rref_binary_matrix = _reduced_row_echelon(binary_matrix)
    rref_binary_matrix_red = rref_binary_matrix[
        ~np.all(rref_binary_matrix == 0, axis=1)
    ]  # remove all-zero rows

    # Get kernel (i.e., nullspace) for trimmed binary matrix using gaussian elimination
    nullspace = _kernel(rref_binary_matrix_red)

    generators = []
    pauli_map = {"00": qml.Identity, "10": qml.PauliX, "11": qml.PauliY, "01": qml.PauliZ}

    for null_vector in nullspace:
        tau = qml.Identity(0)
        for idx, op in enumerate(zip(null_vector[:num_qubits], null_vector[num_qubits:])):
            x, z = op
            tau @= pauli_map[f"{x}{z}"](idx)

        ham = simplify(qml.Hamiltonian([1.0], [tau]))
        generators.append(ham)

    return generators


def paulix_ops(generators, num_qubits):
    r"""Generate the single qubit Pauli-X operators :math:`\sigma^{x}_{i}` for each symmetry :math:`\tau_j`,
    such that it anti-commutes with :math:`\tau_j` and commutes with all others symmetries :math:`\tau_{k\neq j}`.
    These are required to obtain the Clifford operators :math:`U` for the Hamiltonian :math:`H`.

    Args:
        generators (list[Hamiltonian]): list of generators of symmetries, taus, for the Hamiltonian
        num_qubits (int): number of wires required to define the Hamiltonian
    Return:
        list[Observable]: list of single-qubit Pauli-X operators which will be used to build the
        Clifford operators :math:`U`.

    **Example**

    >>> generators = [qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
    ...               qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
    ...               qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)])]
    >>> paulix_ops(generators, 4)
    [PauliX(wires=[1]), PauliX(wires=[2]), PauliX(wires=[3])]
    """
    ops_generator = [g.ops[0] if isinstance(g.ops, list) else g.ops for g in generators]
    bmat = _binary_matrix(ops_generator, num_qubits)

    paulixops = []
    for row in range(bmat.shape[0]):
        bmatrow = bmat[row]
        bmatrest = np.delete(bmat, row, axis=0)
        # reversing the order to prioritize removing higher index wires first
        for col in range(bmat.shape[1] // 2)[::-1]:
            # Anti-commutes with the (row) and commutes with all other symmetries.
            if bmatrow[col] and np.array_equal(
                bmatrest[:, col], np.zeros(bmat.shape[0] - 1, dtype=int)
            ):
                paulixops.append(qml.PauliX(col))
                break

    return paulixops


def _observable_mult(obs_a, obs_b):
    r"""Multiply two PennyLane observables together.

    Each observable should be a linear combination of Pauli words, e.g.,
    :math:`\sum_{k=0}^{N} c_k P_k`, and represented as a PennyLane Hamiltonian.

    Args:
        obs_a (Hamiltonian): first observable
        obs_b (Hamiltonian): second observable

    Returns:
        qml.Hamiltonian: Observable expressed as a PennyLane Hamiltonian

    **Example**

    >>> c = np.array([0.5, 0.5])
    >>> obs_a = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)])
    >>> obs_b = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(1)])
    >>> print(_observable_mult(obs_a, obs_b))
      (-0.25j) [Z1]
    + (-0.25j) [Y0]
    + ( 0.25j) [Y1]
    + ((0.25+0j)) [Y0 X1]
    """
    o = []
    c = []
    for i in range(len(obs_a.terms()[0])):
        for j in range(len(obs_b.terms()[0])):
            op, phase = qml.grouping.pauli_mult_with_phase(obs_a.terms()[1][i], obs_b.terms()[1][j])
            o.append(op)
            c.append(phase * obs_a.terms()[0][i] * obs_b.terms()[0][j])
    return simplify(qml.Hamiltonian(qml.math.stack(c), o))


def clifford(generators, paulixops):
    r"""Compute a Clifford operator from a set of generators and Pauli-X operators.

    This function computes :math:`U = U_0U_1...U_k` for a set of :math:`k` generators and
    :math:`k` Pauli-X operators.

    Args:
        generators (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulixops (list[Operation]): list of single-qubit Pauli-X operators

    Returns:
        Hamiltonian: Clifford operator expressed as a PennyLane Hamiltonian

    **Example**

    >>> t1 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZZII')])
    >>> t2 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZIZI')])
    >>> t3 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZIIZ')])
    >>> generators = [t1, t2, t3]
    >>> paulixops = [qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)]
    >>> u = clifford(generators, paulixops)
    >>> print(u)
      (0.3535533905932737) [Z1 Z2 X3]
    + (0.3535533905932737) [X1 X2 X3]
    + (0.3535533905932737) [Z1 X2 Z3]
    + (0.3535533905932737) [X1 Z2 Z3]
    + (0.3535533905932737) [Z0 X1 X2 Z3]
    + (0.3535533905932737) [Z0 Z1 Z2 Z3]
    + (0.3535533905932737) [Z0 X1 Z2 X3]
    + (0.3535533905932737) [Z0 Z1 X2 X3]
    """
    cliff = []
    for i, t in enumerate(generators):
        cliff.append(1 / 2**0.5 * (paulixops[i] + t))

    u = functools.reduce(lambda i, j: _observable_mult(i, j), cliff)

    return u


def taper(h, generators, paulixops, paulix_sector):
    r"""Transform a Hamiltonian with a Clifford operator and then taper qubits.

    The Hamiltonian is transformed as :math:`H' = U^{\dagger} H U` where :math:`U` is a Clifford
    operator. The transformed Hamiltonian acts trivially on some qubits which are then replaced
    with the eigenvalues of their corresponding Pauli-X operator. The list of these
    eigenvalues is defined as the Pauli sector.

    Args:
        h (Hamiltonian): PennyLane Hamiltonian
        generators (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulixops (list[Operation]): list of single-qubit Pauli-X operators
        paulix_sector (llist[int]): eigenvalues of the Pauli-X operators

    Returns:
        Hamiltonian: the tapered Hamiltonian

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]])
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
    >>> generators = qml.qchem.symmetry_generators(H)
    >>> paulixops = paulix_ops(generators, 4)
    >>> paulix_sector = [1, -1, -1]
    >>> H_tapered = taper(H, generators, paulixops, paulix_sector)
    >>> print(H_tapered)
      ((-0.321034397355757+0j)) [I0]
    + ((0.1809270275619003+0j)) [X0]
    + ((0.7959678503869626+0j)) [Z0]
    """
    u = clifford(generators, paulixops)
    h = _observable_mult(_observable_mult(u, h), u)

    val = np.ones(len(h.terms()[0])) * complex(1.0)

    wireset = u.wires + h.wires
    wiremap = dict(zip(wireset, range(len(wireset) + 1)))
    paulix_wires = [x.wires[0] for x in paulixops]

    for idx, w in enumerate(paulix_wires):
        for i in range(len(h.terms()[0])):
            s = qml.grouping.pauli_word_to_string(h.terms()[1][i], wire_map=wiremap)
            if s[w] == "X":
                val[i] *= paulix_sector[idx]

    o = []
    wires_tap = [i for i in h.wires if i not in paulix_wires]
    wiremap_tap = dict(zip(wires_tap, range(len(wires_tap) + 1)))

    for i in range(len(h.terms()[0])):
        s = qml.grouping.pauli_word_to_string(h.terms()[1][i], wire_map=wiremap)

        wires = [x for x in h.wires if x not in paulix_wires]
        o.append(
            qml.grouping.string_to_pauli_word(
                "".join([s[wiremap[i]] for i in wires]), wire_map=wiremap_tap
            )
        )

    c = anp.multiply(val, h.terms()[0])
    c = qml.math.stack(c)

    tapered_ham = simplify(qml.Hamiltonian(c, o))
    # If simplified Hamiltonian is missing wires, then add wires manually for consistency
    if wires_tap != list(tapered_ham.wires):
        identity_op = functools.reduce(
            lambda i, j: i @ j,
            [
                qml.Identity(wire)
                for wire in Wires.unique_wires([tapered_ham.wires, Wires(wires_tap)])
            ],
        )
        tapered_ham = qml.Hamiltonian(
            np.array([*tapered_ham.coeffs, 0.0]), [*tapered_ham.ops, identity_op]
        )
    return tapered_ham


def optimal_sector(qubit_op, generators, active_electrons):
    r"""Get the optimal sector which contains the ground state.

    To obtain the optimal sector, we need to choose the right eigenvalues for the symmetry generators :math:`\bm{\tau}`.
    We can do so by using the following relation between the Pauli-Z qubit operator and the occupation number under a
    Jordan-Wigner transform.

    .. math::

        \sigma_{i}^{z} = I - 2a_{i}^{\dagger}a_{i}

    According to this relation, the occupied and unoccupied fermionic modes correspond to the -1 and +1 eigenvalues of
    the Pauli-Z operator, respectively. Since all of the generators :math:`\bm{\tau}` consist only of :math:`I` and
    Pauli-Z operators, the correct eigenvalue for each :math:`\tau` operator can be simply obtained by applying it on
    the reference Hartree-Fock (HF) state, and looking at the overlap between the wires on which the Pauli-Z operators
    act and the wires that correspond to occupied orbitals in the HF state.

    Args:
        qubit_op (Hamiltonian): Hamiltonian for which symmetries are being generated
        generators (list[Hamiltonian]): list of symmetry generators for the Hamiltonian
        active_electrons (int): The number of active electrons in the system

    Returns:
        list[int]: eigenvalues corresponding to the optimal sector which contains the ground state

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]])
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
    >>> generators = qml.qchem.symmetry_generators(H)
    >>> qml.qchem.optimal_sector(H, generators, 2)
        [1, -1, -1]
    """

    if active_electrons < 1:
        raise ValueError(
            f"The number of active electrons must be greater than zero;"
            f"got 'electrons'={active_electrons}"
        )

    num_orbitals = len(qubit_op.wires)

    if active_electrons > num_orbitals:
        raise ValueError(
            f"Number of active orbitals cannot be smaller than number of active electrons;"
            f" got 'orbitals'={num_orbitals} < 'electrons'={active_electrons}."
        )

    hf_str = np.where(np.arange(num_orbitals) < active_electrons, 1, 0)

    perm = []
    for tau in generators:
        symmstr = np.array([1 if wire in tau.ops[0].wires else 0 for wire in qubit_op.wires])
        coeff = -1 if numpy.logical_xor.reduce(numpy.logical_and(symmstr, hf_str)) else 1
        perm.append(coeff)

    return perm


def taper_hf(generators, paulixops, paulix_sector, num_electrons, num_wires):
    r"""Transform a Hartree-Fock state with a Clifford operator and then taper qubits.

    The fermionic operators defining the molecule's Hartree-Fock (HF) state are first mapped onto a qubit operator
    using the Jordan-Wigner encoding. This operator is then transformed using the Clifford operators :math:`U`
    obtained from the :math:`\mathbb{Z}_2` symmetries of the molecular Hamiltonian resulting in a qubit operator
    that acts non-trivially only on a subset of qubits. A new, tapered HF state is built on this reduced subset
    of qubits by placing the qubits which are acted on by a Pauli-X or Pauli-Y operators in state :math:`|1\rangle`
    and leaving the rest in state :math:`|0\rangle`.

    Args:
        generators (list[Hamiltonian]): list of generators of symmetries, taus, for the Hamiltonian
        paulixops (list[Operation]):  list of single-qubit Pauli-X operators
        paulix_sector (list[int]): list of eigenvalues of Pauli-X operators
        num_electrons (int): number of active electrons in the system
        num_wires (int): number of wires in the system for generating the Hartree-Fock bitstring

    Returns:
        array(int): tapered Hartree-Fock state

    **Example**

    >>> symbols = ['He', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4588684632]])
    >>> mol = qml.qchem.Molecule(symbols, geometry, charge=1)
    >>> H, n_qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
    >>> n_elec = mol.n_electrons
    >>> generators = qml.qchem.symmetry_generators(H)
    >>> paulixops = qml.qchem.paulix_ops(generators, 4)
    >>> paulix_sector = qml.qchem.optimal_sector(H, generators, n_elec)
    >>> taper_hf(generators, paulixops, paulix_sector, n_elec, n_qubits)
    tensor([1, 1], requires_grad=True)
    """
    # build the untapered Hartree Fock state
    hf = np.where(np.arange(num_wires) < num_electrons, 1, 0)

    # convert the HF state to a corresponding HF observable under the JW transform
    fermop_terms = []
    for idx, bit in enumerate(hf):
        if bit:
            op_coeffs, op_terms = jordan_wigner([idx])
            op_term = qml.Hamiltonian(np.array(op_coeffs), op_terms)
        else:
            op_term = qml.Hamiltonian([1.0], [qml.Identity(idx)])
        fermop_terms.append(op_term)

    ferm_op = functools.reduce(lambda i, j: _observable_mult(i, j), fermop_terms)

    # taper the HF observable using the symmetries obtained from the molecular hamiltonian
    fermop_taper = taper(ferm_op, generators, paulixops, paulix_sector)
    fermop_mat = _binary_matrix(fermop_taper.ops, len(fermop_taper.wires))

    # build a wireset to match wires with that of the tapered Hamiltonian
    gen_wires = Wires.all_wires([generator.wires for generator in generators])
    xop_wires = Wires.all_wires([paulix_op.wires for paulix_op in paulixops])
    wireset = Wires.unique_wires([gen_wires, xop_wires])

    # iterate over the terms in tapered HF observable and build the tapered HF state
    tapered_hartree_fock = []
    for col in fermop_mat.T[fermop_mat.shape[1] // 2 :]:
        if 1 in col:
            tapered_hartree_fock.append(1)
        else:
            tapered_hartree_fock.append(0)
    while len(tapered_hartree_fock) < len(wireset):
        tapered_hartree_fock.append(0)

    return np.array(tapered_hartree_fock).astype(int)


def _build_callables(operation, op_wires=None, op_gen=None):
    r"""Instantiates objects for whichever of the ``operation`` or ``op_gen`` args are callables. For the former,
    it is done using an arbitrary choice of variational arguments to be 1.0 and the specified wires. Whereas for the latter, it
    is done only with the specified wires.

    Args:
        operation (Operation or Callable): qubit operation to be tapered, or a function that applies that operation
        op_wires (Sequence[Any]): wires for the operation in case any of the provided `operation` or `op_gen` are callables
        op_gen (Hamiltonian or Callable): generator of the operation, or a function that returns it in case it cannot be computed internally.

    Returns:
        Tuple(Operation, Hamiltonian)

    Raises:
        ValueError: optional argument `op_wires` is not provided when the provided ``operation`` or ``gen_op`` is a callable
        TypeError: optional argument `op_gen` is a callable but does not have 'wires' as its only keyword argument

    **Example**

    >>> gen_fn = lambda wires: qml.Hamiltonian(
    ...        [0.25, -0.25],
    ...        [qml.PauliX(wires=wires[0]) @ qml.PauliY(wires=wires[1]),
    ...         qml.PauliY(wires=wires[0]) @ qml.PauliX(wires=wires[1])])
    >>> _build_callables(qml.SingleExcitation, op_wires=[0, 2], op_gen=gen_fn)
    (SingleExcitation(1.0, wires=[0, 2]),
    <Hamiltonian: terms=2, wires=[0, 2]>)
    """

    if callable(operation) or callable(op_gen):
        if op_wires is None:
            raise ValueError(
                f"Wires for the operation must be provided with 'op_wires' args if either of 'operation' or 'op_gen' is a callable, got {op_wires}."
            )

    if callable(operation):
        operation = operation(*([1.0] * operation.num_params), wires=op_wires)

    if callable(op_gen):
        try:
            op_gen = op_gen(wires=op_wires)
        except TypeError as exc:
            raise TypeError(
                "Generator function provided with 'op_gen' should have 'wires' as its only required keyword argument."
            ) from exc

    return operation, op_gen


def _build_generator(operation, wire_order, op_gen=None):
    r"""Computes the generator `G` for the general unitary operation :math:`U(\theta)=e^{iG\theta}`, where :math:`\theta` could either be a variational parameter,
    or a constant with some arbitrary fixed value.

    Args:
        operation (Operation): qubit operation to be tapered
        wire_order (Sequence[Any]): order of the wires in the quantum circuit
        op_gen (Hamiltonian): generator of the operation in case it cannot be computed internally.

    Returns:
        Hamiltonian: the generator of the operation

    Raises:
        NotImplementedError: generator of the operation cannot be constructed internally
        ValueError: optional argument `op_gen` is either not a :class:`~.pennylane.Hamiltonian` or a valid generator of the operation

    **Example**

    >>> _build_generator(qml.SingleExcitation, [0, 1], op_wires=[0, 2])
      (-0.25) [Y0 X1]
    + (0.25) [X0 Y1]
    """

    if op_gen is None:
        if operation.num_params < 1:  # Non-parameterized gates
            gen_mat = 1j * scipy.linalg.logm(qml.matrix(operation, wire_order=wire_order))
            op_gen = qml.Hamiltonian(
                *qml.utils.decompose_hamiltonian(gen_mat, wire_order=wire_order, hide_identity=True)
            )
            qml.simplify(op_gen)
            if op_gen.ops[0].label() == qml.Identity(wires=[wire_order[0]]).label():
                op_gen -= qml.Hamiltonian([op_gen.coeffs[0]], [qml.Identity(wires=wire_order[0])])
        else:  # Single-parameter gates
            try:
                op_gen = qml.generator(operation, "hamiltonian")

            except ValueError as exc:
                raise NotImplementedError(
                    f"Generator for {operation} is not implemented, please provide it with 'op_gen' args."
                ) from exc
    else:  # check that user-provided generator is correct
        if not isinstance(op_gen, qml.Hamiltonian):
            raise ValueError(
                f"Generator for the operation needs to be a qml.Hamiltonian, but got {type(op_gen)}."
            )
        coeffs = 1.0

        if operation.parameters and isinstance(operation.parameters[0], (float, complex)):
            coeffs = functools.reduce(
                lambda i, j: i * j, operation.parameters
            )  # coeffs from operation

        mat1 = scipy.linalg.expm(1j * qml.matrix(op_gen, wire_order=wire_order) * coeffs)
        mat2 = qml.matrix(operation, wire_order=wire_order)
        phase = np.divide(mat1, mat2, out=np.zeros_like(mat1, dtype=complex), where=mat1 != 0)[
            np.nonzero(np.round(mat1, 10))
        ]
        if not np.allclose(phase / phase[0], np.ones(len(phase))):  # check if the phase is global
            raise ValueError(
                f"Given op_gen: {op_gen} doesn't seem to be the correct generator for the {operation}."
            )

    return op_gen


# pylint: disable=too-many-branches, too-many-arguments, inconsistent-return-statements, no-member
def taper_operation(
    operation, generators, paulixops, paulix_sector, wire_order, op_wires=None, op_gen=None
):
    r"""Transform a gate operation with a Clifford operator and then taper qubits.

    The qubit operator for the generator of the gate operation is computed either internally or can be provided
    manually via `op_gen` argument. If this operator commutes with all the :math:`\mathbb{Z}_2` symmetries of
    the molecular Hamiltonian, then this operator is transformed using the Clifford operators :math:`U` and
    tapered; otherwise it is discarded. Finally, the tapered generator is exponentiated using :class:`~.pennylane.Exp`
    for building the tapered unitary.

    Args:
        operation (Operation or Callable): qubit operation to be tapered, or a function that applies that operation
        generators (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulixops (list[Operation]):  list of single-qubit Pauli-X operators
        paulix_sector (list[int]): eigenvalues of the Pauli-X operators
        wire_order (Sequence[Any]): order of the wires in the quantum circuit
        op_wires (Sequence[Any]): wires for the operation in case any of the provided `operation` or `op_gen` are callables
        op_gen (Hamiltonian or Callable): generator of the operation, or a function that returns it in case it cannot be computed internally.

    Returns:
        list(Operation): list of operations of type :class:`~.pennylane.Exp` implementing tapered unitary operation

    Raises:
        ValueError: optional argument `op_wires` is not provided when the provided operation is a callable
        TypeError: optional argument `op_gen` is a callable but does not have 'wires' as its only keyword argument
        NotImplementedError: generator of the operation cannot be constructed internally
        ValueError: optional argument `op_gen` is either not a :class:`~.pennylane.Hamiltonian` or a valid generator of the operation

    **Example**

    >>> symbols, geometry = ['He', 'H'], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4589]])
    >>> mol = qchem.Molecule(symbols, geometry, charge=1)
    >>> H, n_qubits = qchem.molecular_hamiltonian(symbols, geometry)
    >>> generators = qchem.symmetry_generators(H)
    >>> paulixops = qchem.paulix_ops(generators, n_qubits)
    >>> paulix_sector = qchem.optimal_sector(H, generators, mol.n_electrons)
    >>> tap_op = qchem.taper_operation(qml.SingleExcitation, generators, paulixops,
    ...                paulix_sector, wire_order=H.wires, op_wires=[0, 2])
    >>> tap_op(3.14159)
    [Exp(1.570795j, 'PauliY', wires=[0])]

    The obtained tapered operation function can then be used within a :class:`~.pennylane.QNode`:

    >>> dev = qml.device('default.qubit', wires=[0, 1])
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     tap_op(params[0])
    ...     return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))
    >>> drawer = qml.draw(circuit, show_all_wires=True)
    >>> print(drawer(params=[3.14159]))
        0: ─Exp(1.570795j PauliY)─┤ ╭<Z@Z>
        1: ───────────────────────┤ ╰<Z@Z>

    .. details::

        **Usage Details**

        ``qml.taper_operation`` can also be used with the quantum operations, in which case one does not need to specify `op_wires` args:

        >>> qchem.taper_operation(qml.SingleExcitation(3.14159, wires=[0, 2]), generators,
                                    paulixops, paulix_sector, wire_order=H.wires)
        [Exp(1.570795j, 'PauliY', wires=[0])]

        Moreover, it can also be used within a :class:`~.pennylane.QNode` directly:

        >>> dev = qml.device('default.qubit', wires=[0, 1])
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qchem.taper_operation(qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3]),
        ...                             generators, paulixops, paulix_sector, H.wires)
        ...     return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))
        >>> drawer = qml.draw(circuit, show_all_wires=True)
        >>> print(drawer(params=[3.14159]))
            0: -╭Exp(0-0.7854j PauliX(0)@PauliY(1))─╭Exp(0-0.7854j PauliY(0)@PauliX(1))─┤ ╭<Z@Z>
            1: ─╰Exp(0-0.7854j PauliX(0)@PauliY(1))─╰Exp(0-0.7854j PauliY(0)@PauliX(1))─┤ ╰<Z@Z>

        For more involved gates operations such as the ones constructed from matrices, users would need to provide their generators manually
        via `op_gen` argument. The generator can be passed as a :class:`~.pennylane.Hamiltonian`:

        >>> op_fun = qml.QubitUnitary(
        ...            np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
        ...                      [0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j],
        ...                      [0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j],
        ...                      [0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j]]), wires=[0, 2])
        >>> op_gen = qml.Hamiltonian([-0.5 * np.pi],
        ...                      [qml.PauliX(wires=[0]) @ qml.PauliX(wires=[2])])
        >>> qchem.taper_operation(op_fun, generators, paulixops, paulix_sector,
        ...                       wire_order=H.wires, op_gen=op_gen)
        [Exp(1.570796j, 'PauliX', wires=[0])]

        Alternatively, generator can also be specified as a function which returns :class:`~.pennylane.Hamiltonian` and uses `wires` as
        its only required keyword argument:

        >>> op_gen = lambda wires: qml.Hamiltonian(
        ...        [0.25, -0.25],
        ...        [qml.PauliX(wires=wires[0]) @ qml.PauliY(wires=wires[1]),
        ...         qml.PauliY(wires=wires[0]) @ qml.PauliX(wires=wires[1])])
        >>> qchem.taper_operation(SingleExcitation, generators, paulixops, paulix_sector,
        ...         wire_order=H.wires, op_wires=[0, 2], op_gen=op_gen)(3.14159)
        [Exp(1.570795j, 'PauliY', wires=[0])]

        **Theory**

        Consider :math:`G` to be the generator of a unitrary :math:`V(\theta)`, i.e.,

        .. math::

            V(\theta) = e^{i G \theta}.

        Then, for :math:`V` to have a non-trivial and compatible tapering with the generators of symmetry
        :math:`\tau`, we should have :math:`[V, \tau_i] = 0` for all :math:`\theta` and :math:`\tau_i`.
        This would hold only when its generator itself commutes with each :math:`\tau_i`,

        .. math::

            [V, \tau_i] = 0 \iff [G, \tau_i]\quad \forall \theta, \tau_i.

        By ensuring this, we can taper the generator :math:`G` using the Clifford operators :math:`U`,
        and exponentiate the transformed generator :math:`G^{\prime}` to obtain a tapered unitary
        :math:`V^{\prime}`,

        .. math::

            V^{\prime} \equiv e^{i U^{\dagger} G U \theta} = e^{i G^{\prime} \theta}.
    """
    # maintain a flag to track functional form of the operation
    callable_op = callable(operation)
    # get dummy objects in case functional form of operation or op_gen is being used
    operation, op_gen = _build_callables(operation, op_wires=op_wires, op_gen=op_gen)

    # build generator for the operation either internally or using the provided op_gen
    op_gen = _build_generator(operation, wire_order, op_gen=op_gen)
    # check compatibility between the generator and the symmeteries
    if np.all(
        [
            [
                qml.is_commuting(op1, op2)
                for op1, op2 in itertools.product(generator.ops, op_gen.ops)
            ]
            for generator in generators
        ]
    ) and not np.all(np.isclose(op_gen.coeffs, np.zeros_like(op_gen.coeffs), rtol=1e-8)):
        gen_tapered = qml.taper(op_gen, generators, paulixops, paulix_sector)
    else:
        gen_tapered = qml.Hamiltonian([], [])
    qml.simplify(gen_tapered)

    def _tapered_op(params):
        r"""Applies the tapered operation for the specified parameter value whenever
        queing context is active, otherwise returns it as a list."""
        if qml.QueuingManager.recording():
            qml.QueuingManager.update_info(operation, owner=gen_tapered)
            for coeff, op in zip(*gen_tapered.terms()):
                qml.exp(op, 1j * params * coeff)
        else:
            ops_tapered = []
            for coeff, op in zip(*gen_tapered.terms()):
                ops_tapered.append(qml.exp(op, 1j * params * coeff))
            return ops_tapered

    # if operation was a callable, return the functional form that accepts new parameters
    if callable_op:
        return _tapered_op

    params = 1.0
    if operation.parameters and isinstance(operation.parameters[0], (float, complex)):
        params = functools.reduce(lambda i, j: i * j, operation.parameters)

    return _tapered_op(params=params)
