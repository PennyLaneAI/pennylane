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
import numpy as np
import pennylane as qml


def _binary_matrix(terms, num_qubits):
    r"""Get a binary matrix representation of the Hamiltonian where each row corresponds to a
    Pauli term, which is represented by a concatenation of Z and X vectors.

    Args:
        terms (Iterable[pennylane.Observable]): operators defining the Hamiltonian
        num_qubits (int): number of wires required to define the Hamiltonian
    Returns:
        array[int]: binary matrix representation of the Hamiltonian of shape
        :math:`len(terms) \times 2*num_qubits`

    **Example**

    .. code-block:: python

        >>> terms = [qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[1]),
        ...          qml.PauliZ(wires=[0]) @ qml.PauliY(wires=[2]),
        ...          qml.PauliX(wires=[0]) @ qml.PauliY(wires=[3])]
        >>> _binary_matrix(terms, 4)
         array([[1, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 0, 0, 1]]))

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

    .. code-block:: python

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

    .. code-block:: python

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
        list[pennylane.Hamiltonian]: list of generators of symmetries, taus, for the Hamiltonian

    **Example**

    .. code-block:: python

        >>> kernel = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
        ...                    [0, 0, 1, 1, 1, 0, 0, 0],
        ...                    [1, 0, 1, 0, 0, 1, 0, 0],
        ...                    [0, 0, 1, 0, 0, 0, 1, 0],
        ...                    [0, 0, 1, 1, 0, 0, 0, 1]])
        >>> generate_taus(kernel, 4)
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
        generators (list[pennylane.Hamiltonian]): list of generators of symmetries, taus, for the Hamiltonian
        num_qubits (int): number of wires required to define the Hamiltonian
    Return:
        list[pennylane.Observable]: list of single-qubit Pauli-X operators which will be used to build the
        Clifford operators :math:`U`.

    **Example**

    .. code-block:: python

        >>> generators = [qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
        ...               qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
        ...               qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)])]
        >>> generate_paulis(generators, qubits)
         [PauliX(wires=[1]), PauliX(wires=[2]), PauliX(wires=[3])]

    """

    ops_generator = [g.ops[0] if isinstance(g.ops, list) else g.ops for g in generators]
    bmat = _binary_matrix(ops_generator, num_qubits)

    pauli_x_ops = []
    for row in range(bmat.shape[0]):
        bmatrow = bmat[row]
        bmatrest = np.delete(bmat, row, axis=0)
        for col in range(bmat.shape[1] // 2):
            # Anti-commutes with the (row) and commutes with all other symmetries.
            if bmatrow[col] and np.array_equal(
                bmatrest[:, col], np.zeros(bmat.shape[0] - 1, dtype=int)
            ):
                pauli_x_ops.append(qml.PauliX(col))
                break

    return pauli_x_ops


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
        qubit_op (pennylane.Hamiltonian): Hamiltonian for which symmetries are to be generated to perform tapering
        num_qubits (int): number of wires required to define the Hamiltonian

    Returns:
        tuple (list[pennylane.Hamiltonian], list[pennylane.operation.Observable]):

            * list[pennylane.Hamiltonian]: list of generators of symmetries, :math:`\mathbf{\tau}`,
              for the Hamiltonian.
            * list[pennylane.operation.Operation]: list of single-qubit Pauli-X operators which will be used
              to build the Clifford operators :math:`U`.

    .. code-block:: python

        >>> symbols = ['H', 'H']
        >>> coordinates = np.array([0., 0., -0.66140414, 0., 0., 0.66140414])
        >>> mol = qml.hf.Molecule(symbols, coordinates)
        >>> H, qubits = qml.hf.generate_hamiltonian(mol)(), 4
        >>> generators, pauli_x = generate_symmetries(H, qubits)
        >>> generators, pauli_x
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
    pauli_x_ops = generate_paulis(generators, num_qubits)

    return generators, pauli_x_ops
