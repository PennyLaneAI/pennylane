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
"""
Utility functions used in Pauli partitioning and measurement reduction schemes utilizing the
symplectic vector-space representation of Pauli words. For information on the symplectic binary
representation of Pauli words and applications, see:

* `arXiv:quant-ph/9705052 <https://arxiv.org/abs/quant-ph/9705052>`_
* `arXiv:1701.08213 <https://arxiv.org/abs/1701.08213>`_
* `arXiv:1907.09386 <https://arxiv.org/abs/1907.09386>`_
"""

from pennylane.wires import Wires
from pennylane.pauli.pauli_utils import pauli_to_binary

import numpy as np


def is_qwc(pauli_vec_1, pauli_vec_2):
    """Checks if two Pauli words in the binary vector representation are qubit-wise commutative.

    Args:
        pauli_vec_1 (Union[list, tuple, array]): first binary vector argument in qubit-wise
            commutator
        pauli_vec_2 (Union[list, tuple, array]): second binary vector argument in qubit-wise
            commutator

    Returns:
        bool: returns True if the input Pauli words are qubit-wise commutative, returns False
        otherwise

    Raises:
        ValueError: if the input vectors are of different dimension, if the vectors are not of even
            dimension, or if the vector components are not strictly binary

    **Example**

    >>> is_qwc([1,0,0,1,1,0],[1,0,1,0,1,0])
    False
    >>> is_qwc([1,0,1,1,1,0],[1,0,0,1,1,0])
    True
    """

    if isinstance(pauli_vec_1, (list, tuple)):
        pauli_vec_1 = np.asarray(pauli_vec_1)
    if isinstance(pauli_vec_2, (list, tuple)):
        pauli_vec_2 = np.asarray(pauli_vec_2)

    if len(pauli_vec_1) != len(pauli_vec_2):
        raise ValueError(
            "Vectors a and b must be the same dimension, instead got shapes {} and {}.".format(
                np.shape(pauli_vec_1), np.shape(pauli_vec_2)
            )
        )

    if len(pauli_vec_1) % 2 != 0:
        raise ValueError(
            "Symplectic vector-space must have even dimension, instead got vectors of shape {}.".format(
                np.shape(pauli_vec_1)
            )
        )

    if not (
        np.array_equal(pauli_vec_1, pauli_vec_1.astype(bool))
        and np.array_equal(pauli_vec_2, pauli_vec_2.astype(bool))
    ):
        raise ValueError(
            "Vectors a and b must have strictly binary components, instead got {} and {}".format(
                pauli_vec_1, pauli_vec_2
            )
        )

    n_qubits = int(len(pauli_vec_1) / 2)

    for i in range(n_qubits):

        first_vec_ith_qubit_paulix = pauli_vec_1[i]
        first_vec_ith_qubit_pauliz = pauli_vec_1[n_qubits + i]
        second_vec_ith_qubit_paulix = pauli_vec_2[i]
        second_vec_ith_qubit_pauliz = pauli_vec_2[n_qubits + i]

        first_vec_qubit_i_is_identity = (
            first_vec_ith_qubit_paulix == first_vec_ith_qubit_pauliz == 0
        )
        second_vec_qubit_i_is_identity = (
            second_vec_ith_qubit_paulix == second_vec_ith_qubit_pauliz == 0
        )
        both_vecs_qubit_i_have_same_x = first_vec_ith_qubit_paulix == second_vec_ith_qubit_paulix
        both_vecs_qubit_i_have_same_z = first_vec_ith_qubit_pauliz == second_vec_ith_qubit_pauliz

        if not (
            ((first_vec_qubit_i_is_identity) or (second_vec_qubit_i_is_identity))
            or ((both_vecs_qubit_i_have_same_x) and (both_vecs_qubit_i_have_same_z))
        ):
            return False

    return True


def observables_to_binary_matrix(observables, n_qubits=None, wire_map=None):
    """Converts a list of Pauli words to the binary vector representation and yields a row matrix
    of the binary vectors.

    The dimension of the binary vectors will be implied from the highest wire being acted on
    non-trivially by the Pauli words in observables.

    Args:
        observables (list[Union[Identity, PauliX, PauliY, PauliZ, Tensor]]): the list of Pauli
            words
        n_qubits (int): number of qubits to specify dimension of binary vector representation
        wire_map (dict): dictionary containing all wire labels used in the Pauli words as keys, and
            unique integer labels as their values


    Returns:
        array[array[int]]: a matrix whose rows are Pauli words in binary vector representation

    **Example**

    >>> observables_to_binary_matrix([PauliX(0) @ PauliY(2), PauliZ(0) @ PauliZ(1) @ PauliZ(2)])
    array([[1., 1., 0., 0., 1., 0.],
           [0., 0., 0., 1., 1., 1.]])
    """

    m_cols = len(observables)

    if wire_map is None:
        all_wires = Wires.all_wires([pauli_word.wires for pauli_word in observables])
        wire_map = {i: c for c, i in enumerate(all_wires.labels)}

    n_qubits_min = max(wire_map.values()) + 1
    if n_qubits is None:
        n_qubits = n_qubits_min
    elif n_qubits < n_qubits_min:
        raise ValueError(
            "n_qubits must support the highest mapped wire index {},"
            " instead got n_qubits={}.".format(n_qubits_min, n_qubits)
        )

    binary_mat = np.zeros((m_cols, 2 * n_qubits))

    for i in range(m_cols):
        binary_mat[i, :] = pauli_to_binary(observables[i], n_qubits=n_qubits, wire_map=wire_map)

    return binary_mat


def qwc_complement_adj_matrix(binary_observables):
    """Obtains the adjacency matrix for the complementary graph of the qubit-wise commutativity
    graph for a given set of observables in the binary representation.

    The qubit-wise commutativity graph for a set of Pauli words has a vertex for each Pauli word,
    and two nodes are connected if and only if the corresponding Pauli words are qubit-wise
    commuting.

    Args:
        binary_observables (array[array[int]]): a matrix whose rows are the Pauli words in the
            binary vector representation

    Returns:
        array[array[int]]: the adjacency matrix for the complement of the qubit-wise commutativity graph

    Raises:
        ValueError: if input binary observables contain components which are not strictly binary

    **Example**

    >>> binary_observables
    array([[1., 0., 1., 0., 0., 1.],
           [0., 1., 1., 1., 0., 1.],
           [0., 0., 0., 1., 0., 0.]])

    >>> qwc_complement_adj_matrix(binary_observables)
    array([[0., 1., 1.],
           [1., 0., 0.],
           [1., 0., 0.]])
    """

    if isinstance(binary_observables, (list, tuple)):
        binary_observables = np.asarray(binary_observables)

    if not np.array_equal(binary_observables, binary_observables.astype(bool)):
        raise ValueError("Expected a binary array, instead got {}".format(binary_observables))

    m_terms = np.shape(binary_observables)[0]
    adj = np.zeros((m_terms, m_terms))

    for i in range(m_terms):
        for j in range(i + 1, m_terms):
            adj[i, j] = int(not is_qwc(binary_observables[i], binary_observables[j]))
            adj[j, i] = adj[i, j]

    return adj
