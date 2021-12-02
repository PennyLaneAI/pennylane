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
from copy import deepcopy
import numpy as np
import pennylane as qml


def _binary_matrix(terms, num_qubits):
    """Get a binary matrix representation of the hamiltonian where each row coressponds to a
    Pauli term, which is represented by concatenation of Z and X vectors.

    Args:
        terms (Iterable[Observable]): operators defining the Hamiltonian.
        num_qubits (int): number of wires required to define the Hamiltonian.
    Returns:
        E (ndarray): binary matrix representation of the Hamiltonian of shape
        :math:`len(terms) \times 2*num_qubits`.

    .. code-block::
        >>> terms = [PauliZ(wires=[0]) @ PauliX(wires=[1]), PauliZ(wires=[0]) @ PauliY(wires=[2]),
                        PauliXwires=[0]) @ PauliY(wires=[3])]
        >>> get_binary_matrix(terms, 4)
         array([[1, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 0, 0, 1]]))

    """

    E = np.zeros((len(terms), 2 * num_qubits), dtype=int)
    for idx, term in enumerate(terms):
        ops, wires = term.name, term.wires
        if len(term.wires) == 1:
            ops = [ops]
        for op, wire in zip(ops, wires):
            if op in ["PauliX", "PauliY"]:
                E[idx][wire + num_qubits] = 1
            if op in ["PauliZ", "PauliY"]:
                E[idx][wire] = 1

    return E


def _row_echelon_form(binary_matrix):
    """Computes the row echelon form (ref) of a binary matrix on the binary finite field Z_2"""

    shape = binary_matrix.shape

    for irow in range(shape[0]):
        pivot_index = 0
        if np.count_nonzero(binary_matrix[irow, :]):
            pivot_index = np.nonzero(binary_matrix[irow, :])[0][0]

        for jrow in range(shape[0]):
            if jrow != irow and binary_matrix[jrow, pivot_index]:
                binary_matrix[jrow, :] = (binary_matrix[jrow, :] + binary_matrix[irow, :]) % 2

    indices = [
        irow
        for irow in range(shape[0] - 1)
        if np.array_equal(binary_matrix[irow, :], np.zeros(shape[1]))
    ]

    temp_row_echelon_matrix = deepcopy(binary_matrix)
    for row in indices[::-1]:
        temp_row_echelon_matrix = np.delete(temp_row_echelon_matrix, row, axis=0)

    row_echelon_matrix = np.zeros(shape, dtype=int)
    row_echelon_matrix[: shape[0] - len(indices), :] = temp_row_echelon_matrix

    return row_echelon_matrix


def _kernel(binary_matrix):
    """Computes the kernel of a binary matrix on the binary finite field Z_2"""

    shape = binary_matrix.shape

    row_aug_mat = np.vstack((binary_matrix, np.identity(shape[1], dtype=int)))
    col_ech_mat = (_row_echelon_form(row_aug_mat.T)).T

    kernel = []
    for icol in range(shape[1]):
        b_col, c_col = col_ech_mat[: shape[0], icol], col_ech_mat[shape[0] :, icol]
        if np.array_equal(b_col, np.zeros(shape[0])) and not np.array_equal(
            c_col, np.zeros(shape[1])
        ):
            kernel.append(col_ech_mat[shape[0] :, icol])

    return kernel


def generate_taus(nullspace, num_qubits):
    """Generate generators tau from the nullspace

    Args:
        nullspace (list): kernel of the binary matrix corresponding to the Hamiltonian.
        num_qubits (int): number of wires required to define the Hamiltonian.

    Returns:
        generators (list): list of generators of symmetries, taus, for the Hamiltonian.

    """

    generators = []

    for null_vector in nullspace:
        tau = qml.Identity(0)
        for idx, op in enumerate(
            zip(null_vector[:num_qubits][::-1], null_vector[num_qubits:][::-1])
        ):
            x, z = op
            if x == 0 and z == 0:
                tau @= qml.Identity(idx)
            elif x == 1 and z == 0:
                tau @= qml.PauliX(idx)
            elif x == 1 and z == 1:
                tau @= qml.PauliY(idx)
            else:
                tau @= qml.PauliZ(idx)
        ham = qml.Hamiltonian([1.0], [tau], simplify=True)
        generators.append(ham)
    return generators


def generate_paulis(generators, num_qubits):
    """Generate the single qubit Pauli X operators :math:`sigma^{x}_{i}` that will be used for
    for generating cliffords for a Hamiltonian :math:`H`.

    Args:
        generators (list): list of generators of symmetries, taus, for the Hamiltonian.
        num_qubits (int): number of wires required to define the Hamiltonian.
    Return:
        sigma_x (list): the list of support of the single-qubit PauliX operators used to build
                        the Clifford operators

    .. code-block::
        >>> symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
        >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
        >>> generators = generate_symmetries(H, qubits)
        >>> generate_clifford(generators, qubits)
        [PauliX(wires=[1]), PauliX(wires=[2]), PauliX(wires=[3])]

    """
    ops_generator = [g.ops[0] if isinstance(g.ops, list) else g.ops for g in generators]
    bmat = _binary_matrix(ops_generator, num_qubits)

    sigma_x = []
    for row in range(bmat.shape[0]):
        bmatrow = bmat[row]
        bmatrest = np.delete(bmat, row, axis=0)
        for col in range(bmat.shape[1] // 2):
            # Anti-commutes with the (row) and commutes with all other symmetries.
            if bmatrow[col] and np.array_equal(
                bmatrest[:, col], np.zeros(bmat.shape[0] - 1, dtype=int)
            ):
                sigma_x.append(qml.PauliX(col))
                break

    return sigma_x


def generate_symmetries(qubit_op, num_qubits):
    """Get generators of symmetries, taus, for a given Hamiltonian.

    Args:
        qubit_op (Hamiltonian): Hamiltonian for which symmetries are to be generated to perform tapering.
        num_qubits (int): number of wires required to define the Hamiltonian.

    Returns:
        generators (list): list of generators of symmetries, taus, for the Hamiltonian.

    .. code-block::
        >>> symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
        >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
        >>> generators = generate_symmetries(H, qubits)
         [(1.0) [Z0 Z1], (1.0) [Z0 Z2], (1.0) [Z0 Z3]]
    """

    # Generate binary matrix for qubit_op
    E = _binary_matrix(qubit_op.ops, num_qubits)

    # Get reduced row echelon form of binary matrix E
    E_rref = _row_echelon_form(E)  # row_echelon(E)
    E_reduced = E_rref[~np.all(E_rref == 0, axis=1)]  # remove all-zero rows

    # Get kernel (i.e., nullspace) for trimmed binary matrix using gaussian elimination
    nullspace = _kernel(E_reduced)

    # Get generators tau from the calculated nullspace
    generators = generate_taus(nullspace, num_qubits)

    # Get unitaries from the calculated nullspace
    pauli_x = generate_paulis(generators, num_qubits)

    return generators, pauli_x
