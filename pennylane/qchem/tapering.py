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

import numpy as np
import scipy

import pennylane as qml
from pennylane.operation import active_new_opmath, convert_to_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires

# Global Variables
PAULI_SENTENCE_MEMORY_SPLITTING_SIZE = 15000


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
        h (Operator): Hamiltonian for which symmetries are to be generated to perform tapering

    Returns:
        list[Operator]: list of generators of symmetries, :math:`\tau`'s, for the Hamiltonian

    **Example**

    >>> symbols = ["H", "H"]
    >>> coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    >>> t = symmetry_generators(H)
    >>> t
    [Z(0) @ Z(1), Z(0) @ Z(2), Z(0) @ Z(3)]
    """
    num_qubits = len(h.wires)

    # Generate binary matrix for qubit_op
    ps = pauli_sentence(h)
    binary_matrix = _binary_matrix_from_pws(list(ps), num_qubits)

    # Get reduced row echelon form of binary matrix
    rref_binary_matrix = _reduced_row_echelon(binary_matrix)
    rref_binary_matrix_red = rref_binary_matrix[
        ~np.all(rref_binary_matrix == 0, axis=1)
    ]  # remove all-zero rows

    # Get kernel (i.e., nullspace) for trimmed binary matrix using gaussian elimination
    nullspace = _kernel(rref_binary_matrix_red)

    generators = []
    pauli_map = {"00": "I", "10": "X", "11": "Y", "01": "Z"}

    for null_vector in nullspace:
        tau = {}
        for idx, op in enumerate(zip(null_vector[:num_qubits], null_vector[num_qubits:])):
            x, z = op
            tau[idx] = pauli_map[f"{x}{z}"]

        ham = qml.pauli.PauliSentence({qml.pauli.PauliWord(tau): 1.0})
        ham = ham.operation(h.wires) if active_new_opmath() else ham.hamiltonian(h.wires)
        generators.append(ham)

    return generators


def paulix_ops(generators, num_qubits):  # pylint: disable=protected-access
    r"""Generate the single qubit Pauli-X operators :math:`\sigma^{x}_{i}` for each symmetry :math:`\tau_j`,
    such that it anti-commutes with :math:`\tau_j` and commutes with all others symmetries :math:`\tau_{k\neq j}`.
    These are required to obtain the Clifford operators :math:`U` for the Hamiltonian :math:`H`.

    Args:
        generators (list[Operator]): list of generators of symmetries, :math:`\tau`'s,
            for the Hamiltonian
        num_qubits (int): number of wires required to define the Hamiltonian

    Return:
        list[Observable]: list of single-qubit Pauli-X operators which will be used to build the
        Clifford operators :math:`U`.

    **Example**

    >>> generators = [qml.Hamiltonian([1.0], [qml.Z(0) @ qml.Z(1)]),
    ...               qml.Hamiltonian([1.0], [qml.Z(0) @ qml.Z(2)]),
    ...               qml.Hamiltonian([1.0], [qml.Z(0) @ qml.Z(3)])]
    >>> paulix_ops(generators, 4)
    [X(1), X(2), X(3)]
    """
    ops_generator = functools.reduce(
        lambda a, b: list(a) + list(b), [pauli_sentence(g) for g in generators]
    )
    bmat = _binary_matrix_from_pws(ops_generator, num_qubits)

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
                paulixops.append(qml.X(col))
                break

    return paulixops


def clifford(generators, paulixops):
    r"""Compute a Clifford operator from a set of generators and Pauli-X operators.

    This function computes :math:`U = U_0U_1...U_k` for a set of :math:`k` generators and
    :math:`k` Pauli-X operators.

    Args:
        generators (list[Operator]): generators expressed as PennyLane Hamiltonians
        paulixops (list[Operation]): list of single-qubit Pauli-X operators

    Returns:
        (Operator): Clifford operator expressed as a PennyLane operator

    **Example**

    >>> t1 = qml.Hamiltonian([1.0], [qml.pauli.string_to_pauli_word('ZZII')])
    >>> t2 = qml.Hamiltonian([1.0], [qml.pauli.string_to_pauli_word('ZIZI')])
    >>> t3 = qml.Hamiltonian([1.0], [qml.pauli.string_to_pauli_word('ZIIZ')])
    >>> generators = [t1, t2, t3]
    >>> paulixops = [qml.X(1), qml.X(2), qml.X(3)]
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
        cliff.append(pauli_sentence(1 / 2**0.5 * (paulixops[i] + t)))

    u = functools.reduce(lambda p, q: p @ q, cliff)

    return u.operation() if active_new_opmath() else u.hamiltonian()


def _split_pauli_sentence(pl_sentence, max_size=15000):
    r"""Splits PauliSentences into smaller chunks of the size determined by the `max_size`.

    Args:
        pl_sentence (PauliSentence): PennyLane PauliSentence to be split
        max_size (int): Maximum size of each chunk

    Returns:
        Iterable consisting of smaller `PauliSentence` objects.
    """
    it, length = iter(pl_sentence), len(pl_sentence)
    for _ in range(0, length, max_size):
        yield qml.pauli.PauliSentence({k: pl_sentence[k] for k in itertools.islice(it, max_size)})


def _taper_pauli_sentence(ps_h, generators, paulixops, paulix_sector):
    r"""Transform a PauliSentence with a Clifford operator and then taper qubits.

    Args:
        ps_h (~.PauliSentence): The Hamiltonian to be tapered
        generators (list[Operator]): generators expressed as PennyLane Hamiltonians
        paulixops (list[~.PauliX]): list of single-qubit Pauli-X operators
        paulix_sector (list[int]): eigenvalues of the Pauli-X operators.

    Returns:
        (Operator): the tapered Hamiltonian
    """

    u = clifford(generators, paulixops)
    ps_u = pauli_sentence(u)  # cast to pauli sentence

    ts_ps = qml.pauli.PauliSentence()
    for ps in _split_pauli_sentence(ps_h, max_size=PAULI_SENTENCE_MEMORY_SPLITTING_SIZE):
        ts_ps += ps_u @ ps @ ps_u  # helps restrict the peak memory usage for u @ h @ u

    wireset = ps_u.wires + ps_h.wires
    wiremap = dict(zip(list(wireset.toset()), range(len(wireset) + 1)))
    paulix_wires = [x.wires[0] for x in paulixops]

    o = []
    val = np.ones(len(ts_ps))

    wires_tap = [i for i in ts_ps.wires if i not in paulix_wires]
    wiremap_tap = dict(zip(wires_tap, range(len(wires_tap) + 1)))

    for i, pw_coeff in enumerate(ts_ps.items()):
        pw, _ = pw_coeff

        for idx, w in enumerate(paulix_wires):
            if pw[w] == "X":
                val[i] *= paulix_sector[idx]

        o.append(
            qml.pauli.string_to_pauli_word(
                "".join([pw[wiremap[i]] for i in wires_tap]),
                wire_map=wiremap_tap,
            )
        )

    c = qml.math.stack(qml.math.multiply(val * complex(1.0), list(ts_ps.values())))

    tapered_ham = (
        qml.simplify(qml.dot(c, o)) if active_new_opmath() else simplify(qml.Hamiltonian(c, o))
    )
    # If simplified Hamiltonian is missing wires, then add wires manually for consistency
    if set(wires_tap) != tapered_ham.wires.toset():
        identity_op = functools.reduce(
            lambda i, j: i @ j,
            [
                qml.Identity(wire)
                for wire in Wires.unique_wires([tapered_ham.wires, Wires(wires_tap)])
            ],
        )

        if active_new_opmath():
            return tapered_ham + (0.0 * identity_op)

        tapered_ham = qml.Hamiltonian(
            np.array([*tapered_ham.coeffs, 0.0]), [*tapered_ham.ops, identity_op]
        )
    return tapered_ham


def taper(h, generators, paulixops, paulix_sector):
    r"""Transform a Hamiltonian with a Clifford operator and then taper qubits.

    The Hamiltonian is transformed as :math:`H' = U^{\dagger} H U` where :math:`U` is a Clifford
    operator. The transformed Hamiltonian acts trivially on some qubits which are then replaced
    with the eigenvalues of their corresponding Pauli-X operator. The list of these
    eigenvalues is defined as the Pauli sector.

    Args:
        h (Operator): Hamiltonian as a PennyLane operator
        generators (list[Operator]): generators expressed as PennyLane Hamiltonians
        paulixops (list[Operation]): list of single-qubit Pauli-X operators
        paulix_sector (list[int]): eigenvalues of the Pauli-X operators

    Returns:
        (Operator): the tapered Hamiltonian

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]])
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
    >>> generators = qml.qchem.symmetry_generators(H)
    >>> paulixops = paulix_ops(generators, 4)
    >>> paulix_sector = [1, -1, -1]
    >>> H_tapered = taper(H, generators, paulixops, paulix_sector)
    >>> H_tapered
    (
        (-0.3210343973331179-2.0816681711721685e-17j) * I(0)
      + (0.7959678504583807+0j) * Z(0)
      + (0.18092702760702645+0j) * X(0)
    )
    """

    ps_h = pauli_sentence(h)
    return _taper_pauli_sentence(ps_h, generators, paulixops, paulix_sector)


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
        qubit_op (Operator): Hamiltonian for which symmetries are being generated
        generators (list[Operator]): list of symmetry generators for the Hamiltonian
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
        symmstr = np.array([1 if wire in tau.wires else 0 for wire in qubit_op.wires.toset()])
        coeff = -1 if np.logical_xor.reduce(np.logical_and(symmstr, hf_str)) else 1
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
        generators (list[Operator]): list of generators of symmetries, taus, for the Hamiltonian
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
    >>> H, n_qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=1)
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
    ferm_ps = PauliSentence({PauliWord({0: "I"}): 1.0})
    for idx, bit in enumerate(hf):
        if bit:
            ps = qml.jordan_wigner(qml.FermiC(idx), ps=True)
        else:
            ps = PauliSentence({PauliWord({idx: "I"}): 1.0})
        ferm_ps @= ps

    # taper the HF observable using the symmetries obtained from the molecular hamiltonian
    fermop_taper = _taper_pauli_sentence(ferm_ps, generators, paulixops, paulix_sector)
    fermop_ps = pauli_sentence(fermop_taper)
    fermop_mat = _binary_matrix_from_pws(list(fermop_ps), len(fermop_taper.wires))

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
    ...        [qml.X(wires[0]) @ qml.Y(wires[1]),
    ...         qml.Y(wires[0]) @ qml.X(wires[1])])
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
        op_gen (Hamiltonian or PauliSentence): generator of the operation in case it cannot be computed internally.
    Returns:
        Hamiltonian: the generator of the operation
    Raises:
        NotImplementedError: generator of the operation cannot be constructed internally
        ValueError: optional argument `op_gen` either is not a valid generator of the operation or is not a
            :class:`~.pennylane.Hamiltonian`, :class:`~.PauliSentence`, or an arithmetic operator
    **Example**
    >>> _build_generator(qml.SingleExcitation, [0, 1], op_wires=[0, 2])
      (-0.25) [Y0 X1]
    + (0.25) [X0 Y1]
    """
    if op_gen is None:
        if operation.num_params < 1:  # Non-parameterized gates
            gen_mat = 1j * scipy.linalg.logm(qml.matrix(operation, wire_order=wire_order))
            op_gen = qml.pauli_decompose(
                gen_mat, wire_order=wire_order, hide_identity=True, pauli=True
            )
            op_gen.simplify()
            op_gen.pop(PauliWord({}), 0.0)
        else:  # Single-parameter gates
            try:
                # TODO: simplify when qml.generator has a proper support for "arithmetic".
                op_gen = (
                    operation.generator()
                    if active_new_opmath()
                    else qml.generator(operation, "arithmetic")
                ).pauli_rep

            except (ValueError, qml.operation.GeneratorUndefinedError) as exc:
                raise NotImplementedError(
                    f"Generator for {operation} is not implemented, please provide it with 'op_gen' args."
                ) from exc
    else:  # check that user-provided generator is correct
        if not isinstance(
            op_gen, (qml.ops.Hamiltonian, qml.ops.LinearCombination, PauliSentence)
        ) and not isinstance(getattr(op_gen, "pauli_rep", None), PauliSentence):
            raise ValueError(
                f"Generator for the operation needs to be a valid operator, but got {type(op_gen)}."
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
        op_gen = convert_to_opmath(op_gen).pauli_rep

    return op_gen


# pylint: disable=too-many-branches, too-many-arguments, inconsistent-return-statements, no-member
def taper_operation(
    operation, generators, paulixops, paulix_sector, wire_order, op_wires=None, op_gen=None
):
    r"""Transform a gate operation with a Clifford operator and then taper qubits.

    The qubit operator for the generator of the gate operation is computed either internally or can be provided
    manually via the ``op_gen`` argument. If this operator commutes with all the :math:`\mathbb{Z}_2` symmetries of
    the molecular Hamiltonian, then this operator is transformed using the Clifford operators :math:`U` and
    tapered; otherwise it is discarded. Finally, the tapered generator is exponentiated using :class:`~.Exp`
    for building the tapered unitary.

    Args:
        operation (Operation or Callable): qubit operation to be tapered, or a function that applies that operation
        generators (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulixops (list[Operation]):  list of single-qubit Pauli-X operators
        paulix_sector (list[int]): eigenvalues of the Pauli-X operators
        wire_order (Sequence[Any]): order of the wires in the quantum circuit
        op_wires (Sequence[Any]): wires for the operation in case any of the provided ``operation`` or ``op_gen`` are callables
        op_gen (Hamiltonian or PauliSentence or Callable): generator of the operation, or a function that returns it in case it cannot be computed internally.

    Returns:
        list[Operation]: list of operations of type :class:`~.pennylane.Exp` implementing tapered unitary operation

    Raises:
        ValueError: optional argument ``op_wires`` is not provided when the provided operation is a callable
        TypeError: optional argument ``op_gen`` is a callable but does not have ``wires`` as its only keyword argument
        NotImplementedError: generator of the operation cannot be constructed internally
        ValueError: optional argument ``op_gen`` is either not a :class:`~.pennylane.Hamiltonian` or a valid generator of the operation

    **Example**

    >>> symbols, geometry = ['He', 'H'], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4589]])
    >>> mol = qchem.Molecule(symbols, geometry, charge=1)
    >>> H, n_qubits = qchem.molecular_hamiltonian(symbols, geometry, charge=1)
    >>> generators = qchem.symmetry_generators(H)
    >>> paulixops = qchem.paulix_ops(generators, n_qubits)
    >>> paulix_sector = qchem.optimal_sector(H, generators, mol.n_electrons)
    >>> tap_op = qchem.taper_operation(qml.SingleExcitation, generators, paulixops,
    ...                                paulix_sector, wire_order=H.wires, op_wires=[0, 2])
    >>> tap_op(3.14159)
    [Exp(1.5707949999999993j PauliY), Exp(0j Identity)]

    The obtained tapered operation function can then be used within a :class:`~.pennylane.QNode`:

    >>> dev = qml.device('default.qubit', wires=[0, 1])
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     tap_op(params[0])
    ...     return qml.expval(qml.Z(0)@qml.Z(1))
    >>> drawer = qml.draw(circuit, show_all_wires=True)
    >>> print(drawer(params=[3.14159]))
    0: ──Exp(0.00+1.57j Y)─┤ ╭<Z@Z>
    1: ────────────────────┤ ╰<Z@Z>

    .. details::
        :title: Usage Details
        :href: usage-taper-operation

        ``qml.taper_operation`` can also be used with the quantum operations, in which case one does not need to specify ``op_wires`` args:

        >>> qchem.taper_operation(qml.SingleExcitation(3.14159, wires=[0, 2]), generators,
        ...                       paulixops, paulix_sector, wire_order=H.wires)
        [Exp(1.570795j PauliY)]

        Moreover, it can also be used within a :class:`~.pennylane.QNode` directly:

        >>> dev = qml.device('default.qubit', wires=[0, 1])
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qchem.taper_operation(qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3]),
        ...                           generators, paulixops, paulix_sector, H.wires)
        ...     return qml.expval(qml.Z(0)@qml.Z(1))
        >>> drawer = qml.draw(circuit, show_all_wires=True)
        >>> print(drawer(params=[3.14159]))
        0: ─╭Exp(-0.00-0.79j X@Y)─╭Exp(-0.00-0.79j Y@X)─┤ ╭<Z@Z>
        1: ─╰Exp(-0.00-0.79j X@Y)─╰Exp(-0.00-0.79j Y@X)─┤ ╰<Z@Z>

        For more involved gates operations such as the ones constructed from matrices, users would need to provide their generators manually
        via the ``op_gen`` argument. The generator can be passed as a :class:`~.pennylane.Hamiltonian`, :class:`~.PauliSentence` or any
        arithmetic operator:

        >>> op_fun = qml.QubitUnitary(np.array([[0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
        ...                                     [0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j],
        ...                                     [0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j],
        ...                                     [0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j]]), wires=[0, 2])
        >>> op_gen = qml.Hamiltonian([-0.5 * np.pi],
        ...                          [qml.X(0) @ qml.X(2)])
        >>> qchem.taper_operation(op_fun, generators, paulixops, paulix_sector,
        ...                       wire_order=H.wires, op_gen=op_gen)
        [Exp(1.5707963267948957j PauliX)]

        Alternatively, generators can also be specified as a function which returns :class:`~.pennylane.Hamiltonian`
        or an arithmetic operator, and uses ``wires`` as its only required keyword argument:

        >>> op_gen = lambda wires: qml.Hamiltonian(
        ...     [0.25, -0.25],
        ...     [qml.X(wires[0]) @ qml.Y(wires[1]),
        ...      qml.Y(wires[0]) @ qml.X(wires[1])])
        >>> qchem.taper_operation(qml.SingleExcitation, generators, paulixops, paulix_sector,
        ...                       wire_order=H.wires, op_wires=[0, 2], op_gen=op_gen)(3.14159)
        [Exp(1.570795j PauliY)]

    .. details::
        :title: Theory
        :href: theory-taper-operation

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

    # Performing commutation check for pauli sentences
    # TODO: replace when qml.is_commuting supports Pauli words and sentences
    def _is_commuting(ps1, ps2):
        commutator = ps1.commutator(ps2)
        commutator.simplify()
        return commutator == PauliSentence({})

    # Obtain the tapered generator for the operation
    with qml.QueuingManager.stop_recording():
        # Get pauli rep for symmetery generators
        ps_gen = list(map(lambda x: convert_to_opmath(x).pauli_rep, generators))

        gen_tapered = PauliSentence({})
        if all(_is_commuting(sym, op_gen) for sym in ps_gen) and not qml.math.allclose(
            list(op_gen.values()), 0.0, rtol=1e-8
        ):
            gen_tapered = qml.taper(op_gen, generators, paulixops, paulix_sector)
            gen_tapered = pauli_sentence(gen_tapered)
        gen_tapered.simplify()

    def _tapered_op(params):
        r"""Applies the tapered operation for the specified parameter value whenever
        queing context is active, otherwise returns it as a list."""
        if qml.QueuingManager.recording():
            qml.QueuingManager.remove(operation)
            for op, coeff in gen_tapered.items():
                qml.exp(op.operation(), 1j * params * coeff)
        else:
            ops_tapered = []
            for op, coeff in gen_tapered.items():
                ops_tapered.append(qml.exp(op.operation(), 1j * params * coeff))
            return ops_tapered

    # if operation was a callable, return the functional form that accepts new parameters
    if callable_op:
        return _tapered_op

    params = 1.0
    if operation.parameters and isinstance(operation.parameters[0], (float, complex)):
        params = functools.reduce(lambda i, j: i * j, operation.parameters)

    return _tapered_op(params=params)
