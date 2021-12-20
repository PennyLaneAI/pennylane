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

import autograd.numpy as anp
import pennylane as qml
from pennylane import numpy as np


def _binary_matrix(terms, num_qubits):
    r"""Get a binary matrix representation of the Hamiltonian where each row corresponds to a
    Pauli term, which is represented by a concatenation of Z and X vectors.

    Args:
        terms (Iterable[Observable]): operators defining the Hamiltonian
        num_qubits (int): number of wires required to define the Hamiltonian
    Returns:
        array[int]: binary matrix representation of the Hamiltonian of shape
        :math:`len(terms) \times 2*num_qubits`

    **Example**

    >>> terms = [qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[1]),
    ...          qml.PauliZ(wires=[0]) @ qml.PauliY(wires=[2]),
    ...          qml.PauliX(wires=[0]) @ qml.PauliY(wires=[3])]
    >>> _binary_matrix(terms, 4)
    array([[1, 0, 0, 0, 0, 1, 0, 0],
           [1, 0, 1, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 1, 0, 0, 1]])
    """
    binary_matrix = np.zeros((len(terms), 2 * num_qubits), dtype=int)
    for idx, term in enumerate(terms):
        ops, wires = term.name, term.wires
        if len(wires) == 1:
            ops = [ops]
        for op, wire in zip(ops, wires):
            if op in ["PauliX", "PauliY"]:
                binary_matrix[idx][wire + num_qubits] = 1
            if op in ["PauliZ", "PauliY"]:
                binary_matrix[idx][wire] = 1

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


def get_generators(nullspace, num_qubits):
    r"""Compute the generators :math:`\{\tau_1, \ldots, \tau_k\}` from the nullspace of
    the binary matrix form of a Hamiltonian over the binary field :math:`\mathbb{Z}_2`.
    These correspond to the generator set of the :math:`\mathbb{Z}_2`-symmetries present
    in the Hamiltonian as given in `arXiv:1910.14644 <https://arxiv.org/abs/1910.14644>`_.

    Args:
        nullspace (array[int]): kernel of the binary matrix corresponding to the Hamiltonian
        num_qubits (int): number of wires required to define the Hamiltonian

    Returns:
        list[Hamiltonian]: list of generators of symmetries, taus, for the Hamiltonian

    **Example**

    >>> kernel = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 1, 1, 1, 0, 0, 0],
    ...                    [1, 0, 1, 0, 0, 1, 0, 0],
    ...                    [0, 0, 1, 0, 0, 0, 1, 0],
    ...                    [0, 0, 1, 1, 0, 0, 0, 1]])
    >>> get_generators(kernel, 4)
    [(1.0) [X1], (1.0) [Z0 X2 X3], (1.0) [X0 Z1 X2], (1.0) [Y2], (1.0) [X2 Y3]]
    """
    generators = []
    pauli_map = {"00": qml.Identity, "10": qml.PauliX, "11": qml.PauliY, "01": qml.PauliZ}

    for null_vector in nullspace:
        tau = qml.Identity(0)
        for idx, op in enumerate(zip(null_vector[:num_qubits], null_vector[num_qubits:])):
            x, z = op
            tau @= pauli_map[f"{x}{z}"](idx)

        ham = qml.Hamiltonian([1.0], [tau], simplify=True)
        generators.append(ham)

    return generators


def generate_paulis(generators, num_qubits):
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
    >>> generate_paulis(generators, 4)
    [PauliX(wires=[1]), PauliX(wires=[2]), PauliX(wires=[3])]
    """
    ops_generator = [g.ops[0] if isinstance(g.ops, list) else g.ops for g in generators]
    bmat = _binary_matrix(ops_generator, num_qubits)

    paulix_ops = []
    for row in range(bmat.shape[0]):
        bmatrow = bmat[row]
        bmatrest = np.delete(bmat, row, axis=0)
        for col in range(bmat.shape[1] // 2):
            # Anti-commutes with the (row) and commutes with all other symmetries.
            if bmatrow[col] and np.array_equal(
                bmatrest[:, col], np.zeros(bmat.shape[0] - 1, dtype=int)
            ):
                paulix_ops.append(qml.PauliX(col))
                break

    return paulix_ops


def generate_symmetries(qubit_op, num_qubits):
    r"""Compute the generator set of the symmetries :math:`\mathbf{\tau}` and the corresponding single-qubit
    set of the Pauli-X operators :math:`\mathbf{\sigma^x}` that are used to build the Clifford operators
    :math:`U`, according to the following relation:

    .. math::

        U_i = \frac{1}{\sqrt{2}}(\tau_i+\sigma^{x}_{q}).

    Here, :math:`\sigma^{x}_{q}` is the Pauli-X operator acting on qubit :math:`q`. These :math:`U_i` can be
    used to transform the Hamiltonian :math:`H` in such a way that it acts trivially or at most with one
    Pauli-gate on a subset of qubits, which allows us to taper off those qubits from the simulation
    using :func:`~.transform_hamiltonian`.

    Args:
        qubit_op (Hamiltonian): Hamiltonian for which symmetries are to be generated to perform tapering
        num_qubits (int): number of wires required to define the Hamiltonian

    Returns:
        tuple (list[Hamiltonian], list[Observable]):

            * list[Hamiltonian]: list of generators of symmetries, :math:`\mathbf{\tau}`,
              for the Hamiltonian.
            * list[Operation]: list of single-qubit Pauli-X operators which will be used
              to build the Clifford operators :math:`U`.

    **Example**

    >>> symbols = ['H', 'H']
    >>> geometry = np.array([[0., 0., -0.66140414], [0., 0., 0.66140414]])
    >>> mol = qml.hf.Molecule(symbols, geometry)
    >>> H, qubits = qml.hf.generate_hamiltonian(mol)(geometry), 4
    >>> generators, paulix_ops = qml.hf.tapering.generate_symmetries(H, qubits)
    >>> generators, paulix_ops
    ([(1.0) [Z0 Z1], (1.0) [Z0 Z2], (1.0) [Z0 Z3]],
    [PauliX(wires=[1]), PauliX(wires=[2]), PauliX(wires=[3])])
    """
    # Generate binary matrix for qubit_op
    binary_matrix = _binary_matrix(qubit_op.ops, num_qubits)

    # Get reduced row echelon form of binary matrix
    rref_binary_matrix = _reduced_row_echelon(binary_matrix)
    rref_binary_matrix_red = rref_binary_matrix[
        ~np.all(rref_binary_matrix == 0, axis=1)
    ]  # remove all-zero rows

    # Get kernel (i.e., nullspace) for trimmed binary matrix using gaussian elimination
    nullspace = _kernel(rref_binary_matrix_red)

    # Get generators tau from the calculated nullspace
    generators = get_generators(nullspace, num_qubits)

    # Get unitaries from the calculated nullspace
    paulix_ops = generate_paulis(generators, num_qubits)

    return generators, paulix_ops


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
    for i in range(len(obs_a.terms[0])):
        for j in range(len(obs_b.terms[0])):
            op, phase = qml.grouping.pauli_mult_with_phase(obs_a.terms[1][i], obs_b.terms[1][j])
            o.append(op)
            c.append(phase * obs_a.terms[0][i] * obs_b.terms[0][j])

    return _simplify(qml.Hamiltonian(qml.math.stack(c), o))


def _simplify(h, cutoff=1.0e-12):
    r"""Add together identical terms in the Hamiltonian.

    The Hamiltonian terms with identical Pauli words are added together and eliminated if the
    overall coefficient is zero.

    Args:
        h (Hamiltonian): PennyLane Hamiltonian
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        Hamiltonian: Simplified PennyLane Hamiltonian

    **Example**

    >>> c = np.array([0.5, 0.5])
    >>> h = qml.Hamiltonian(c, [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)])
    >>> print(_simplify(h))
    (1.0) [X0 Y1]
    """
    s = []
    wiremap = dict(zip(h.wires, range(len(h.wires) + 1)))
    for term in h.terms[1]:
        term = qml.operation.Tensor(term).prune()
        s.append(qml.grouping.pauli_word_to_string(term, wire_map=wiremap))

    o = list(set(s))
    c = [0.0] * len(o)
    for i, item in enumerate(s):
        c[o.index(item)] += h.terms[0][i]
    c = qml.math.stack(c)

    coeffs = []
    ops = []
    nonzero_ind = np.argwhere(abs(c) > cutoff).flatten()
    for i in nonzero_ind:
        coeffs.append(c[i])
        ops.append(qml.grouping.string_to_pauli_word(o[i], wire_map=wiremap))
    try:
        coeffs = qml.math.stack(coeffs)
    except ValueError:
        pass

    return qml.Hamiltonian(coeffs, ops)


def clifford(generators, paulix_ops):
    r"""Compute a Clifford operator from a set of generators and Pauli-X operators.

    This function computes :math:`U = U_0U_1...U_k` for a set of :math:`k` generators and
    :math:`k` Pauli-X operators.

    Args:
        generators (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulix_ops (list[Operation]): list of single-qubit Pauli-X operators

    Returns:
        Hamiltonian: Clifford operator expressed as a PennyLane Hamiltonian

    **Example**

    >>> t1 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZZII')])
    >>> t2 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZIZI')])
    >>> t3 = qml.Hamiltonian([1.0], [qml.grouping.string_to_pauli_word('ZIIZ')])
    >>> generators = [t1, t2, t3]
    >>> paulix_ops = [qml.PauliX(1), qml.PauliX(2), qml.PauliX(3)]
    >>> u = clifford(generators, paulix_ops)
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
        cliff.append(1 / 2 ** 0.5 * (paulix_ops[i] + t))

    u = functools.reduce(lambda i, j: _observable_mult(i, j), cliff)

    return u


def transform_hamiltonian(h, generators, paulix_ops, paulix_sector):
    r"""Transform a Hamiltonian with a Clifford operator and taper qubits.

    The Hamiltonian is transformed as :math:`H' = U^{\dagger} H U` where :math:`U` is a Clifford
    operator. The transformed Hamiltonian acts trivially on some qubits which are then replaced
    with the eigenvalues of their corresponding Pauli-X operator. The list of these
    eigenvalues is defined as the Pauli sector.

    Args:
        h (Hamiltonian): PennyLane Hamiltonian
        generators (list[Hamiltonian]): generators expressed as PennyLane Hamiltonians
        paulix_ops (list[Operation]): list of single-qubit Pauli-X operators
        paulix_sector (llist[int]): eigenvalues of the Pauli-X operators

    Returns:
        Hamiltonian: the tapered Hamiltonian

    **Example**

    >>> symbols = ["H", "H"]
    >>> geometry = np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]])
    >>> mol = qml.hf.Molecule(symbols, geometry)
    >>> H = qml.hf.generate_hamiltonian(mol)(geometry)
    >>> generators, paulix_ops = qml.hf.generate_symmetries(H, len(H.wires))
    >>> paulix_sector = [1, -1, -1]
    >>> H_tapered = transform_hamiltonian(H, generators, paulix_ops, paulix_sector)
    >>> print(H_tapered)
      ((-0.321034397355757+0j)) [I0]
    + ((0.1809270275619003+0j)) [X0]
    + ((0.7959678503869626+0j)) [Z0]
    """
    u = clifford(generators, paulix_ops)
    h = _observable_mult(_observable_mult(u, h), u)

    val = np.ones(len(h.terms[0])) * complex(1.0)

    wiremap = dict(zip(h.wires, range(len(h.wires) + 1)))
    paulix_wires = [x.wires[0] for x in paulix_ops]
    for idx, w in enumerate(paulix_wires):
        for i in range(len(h.terms[0])):
            s = qml.grouping.pauli_word_to_string(h.terms[1][i], wire_map=wiremap)
            if s[w] == "X":
                val[i] *= paulix_sector[idx]

    o = []
    wires_tap = [h.wires[i] for i in range(len(h.wires)) if i not in paulix_wires]
    wiremap_tap = dict(zip(wires_tap, range(len(wires_tap) + 1)))
    for i in range(len(h.terms[0])):
        s = qml.grouping.pauli_word_to_string(h.terms[1][i], wire_map=wiremap)
        wires = [x for x in h.wires if x not in paulix_wires]
        o.append(
            qml.grouping.string_to_pauli_word("".join([s[i] for i in wires]), wire_map=wiremap_tap)
        )

    c = anp.multiply(val, h.terms[0])
    c = qml.math.stack(c)

    return _simplify(qml.Hamiltonian(c, o))
