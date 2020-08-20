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
Functions for finding the unitary transformations required to diagonalize Pauli partitions in the
measurement basis, and templates for the circuit implementations.
"""
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.wires import Wires
from pennylane.templates import template
from pennylane.grouping.utils import pauli_to_binary, is_qwc, get_n_qubits
import numpy as np


@template
def qwc_rotation(pauli_dict):
    """Performs circuit implementation of diagonalizing unitary for a Pauli string.

    **Usage example:**

    >>> pauli_dict = {0: qml.PauliX(0), 2: qml.PauliY(2), 3: qml.PauliZ(3)}
    >>> qwc_rotation(pauli_dict)
    [RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[2])]

    Args:
        pauli_dict (dict[int, Observable]): key is the wire (int), and value is a qml.PauliX,
            qml.PauliY, or qml.PauliZ instance.

    Raises:
        TypeError: if any values of pauli_dict are not instances of qml.PauliX, qml.PauliY, or
            qml.PauliZ.

    """
    pauli_ops = (qml.PauliX, qml.PauliY, qml.PauliZ)
    if not all(isinstance(value, pauli_ops) for value in pauli_dict.values()):
        idx = [[isinstance(value, pauli_ops) for value in pauli_dict.values()].index(False)]
        value_type = type(list(pauli_dict.values()))
        raise TypeError(
            "All values of input pauli_dict must be either PauliX, PauliY, or PauliZ instances,"
            " instead got {} instance.".format(value_type)
        )

    for wire in pauli_dict:
        if isinstance(pauli_dict[wire], qml.PauliX):
            qml.RY(-np.pi / 2, wires=wire)

        elif isinstance(pauli_dict[wire], qml.PauliY):
            qml.RX(np.pi / 2, wires=wire)


def diagonalize_qwc_grouping(qwc_grouping):
    """Diagonalizes a list of mutually qubit-wise commutative Pauli strings.

    **Usage example:**

    >>> qwc_group = [qml.PauliX(0) @ qml.PauliZ(1),
                     qml.PauliX(0) @ qml.PauliY(3),
                     qml.PauliZ(1) @ qml.PauliY(3)]
    >>> diagonalize_qwc_grouping(qwc_group)
    ([RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[3])],
     [Tensor(PauliZ(wires=[0]), PauliZ(wires=[1])),
     Tensor(PauliZ(wires=[0]), PauliZ(wires=[3])),
     Tensor(PauliZ(wires=[1]), PauliZ(wires=[3]))])

    Args:
        qwc_grouping (list[Observable]): a list of Pauli string observables that is mutually
            qubit-wise commutative.

    Returns:
        unitary (list[Operation]): an instance of the qwc_rotation template which diagonalizes the
            qubit-wise commuting grouping.
        diag_terms (list[Observable]): list of Pauli string observables diagonal in the
            computational basis.

    Raises:
        ValueError: if any 2 elements in the input QWC grouping are not qubit-wise commutative.

    """
    m_paulis = len(qwc_grouping)
    all_wires = Wires.all_wires([pauli_word.wires for pauli_word in qwc_grouping])
    wire_map = {i: c for c, i in enumerate(all_wires)}
    for i in range(m_paulis):
        for j in range(i + 1, m_paulis):
            if not is_qwc(
                pauli_to_binary(qwc_grouping[i], wire_map=wire_map),
                pauli_to_binary(qwc_grouping[j], wire_map=wire_map),
            ):
                raise ValueError(
                    "{} and {} are not qubit-wise commuting.".format(
                        qwc_grouping[i], qwc_grouping[j]
                    )
                )

    pauli_dict = {}
    diag_terms = []

    for term in qwc_grouping:
        if term.name != ["Identity"]:
            diag_term = Tensor(qml.Identity(wires=0))
            if not isinstance(term, Tensor):
                term = Tensor(term)
            for sigma in term.obs:
                pauli_dict[sigma.wires[0]] = sigma
                diag_term @= qml.PauliZ(wires=sigma.wires)
            diag_term = diag_term.prune()
            if not isinstance(diag_term, Tensor):
                diag_term = Tensor(diag_term)
            diag_terms.append(diag_term)
        else:
            diag_terms.append(term)

    unitary = qwc_rotation(pauli_dict)

    return unitary, diag_terms


def obtain_qwc_post_rotations_and_diagonalized_groupings(qwc_groupings):
    """Diagonalizes a list of qubit-wise commutative groupings of Pauli strings.

    **Usage example:**

    >>> qwc_group_1 = [qml.PauliX(0) @ qml.PauliZ(1),
                       qml.PauliX(0) @ qml.PauliY(3),
                       qml.PauliZ(1) @ qml.PauliY(3)]
    >>> qwc_group_2 = [qml.PauliY(0),
                       qml.PauliY(0) @ qml.PauliX(2),
                       qml.PauliX(1) @ qml.PauliZ(3)]
    >>> obtain_qwc_post_rotations_and_diagonalized_groupings([qwc_group_1, qwc_group_2])
    ([[RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[3])],
     [RX(1.5707963267948966, wires=[0]), RY(-1.5707963267948966, wires=[2]),
     RY(-1.5707963267948966, wires=[1])]],
     [[Tensor(PauliZ(wires=[0]), PauliZ(wires=[1])), Tensor(PauliZ(wires=[0]), PauliZ(wires=[3])),
       Tensor(PauliZ(wires=[1]), PauliZ(wires=[3]))], [Tensor(PauliZ(wires=[0])),
       Tensor(PauliZ(wires=[0]), PauliZ(wires=[2])), Tensor(PauliZ(wires=[1]),
       PauliZ(wires=[3]))]])

    Args:
        qwc_groupings (list[list[Observable]]): a list of mutualy qubit-wise commutative groupings
            of Pauli string observables.

    Returns:
        post_rotations (list[list[Operation]]): a list of instances of the qwc_rotation template
            which diagonalizes the qubit-wise commuting grouping, order corresponding to
            qwc_groupings.
        diag_groupings (list[list[Observable]]): a list of QWC groupings diagonalized in the
            computational basis, order corresponding to qwc_groupings.

    """

    post_rotations = []
    diag_groupings = []
    m_groupings = len(qwc_groupings)

    for i in range(m_groupings):

        diagonalizing_unitary, diag_grouping = diagonalize_qwc_grouping(qwc_groupings[i])
        post_rotations.append(diagonalizing_unitary)
        diag_groupings.append(diag_grouping)

    return post_rotations, diag_groupings
