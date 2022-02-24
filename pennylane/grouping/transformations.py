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
Functions for finding the unitary transformations required to diagonalize Pauli partitions in the
measurement basis, and templates for the circuit implementations.
"""
import numpy as np

import pennylane as qml
from pennylane.operation import Tensor
from pennylane.wires import Wires
from pennylane.grouping.utils import (
    pauli_to_binary,
    are_identical_pauli_words,
    is_qwc,
    is_pauli_word,
)


def qwc_rotation(pauli_operators):
    """Performs circuit implementation of diagonalizing unitary for a Pauli word.

    Args:
        pauli_operators (list[Union[PauliX, PauliY, PauliZ, Identity]]): Single-qubit Pauli
            operations. No Pauli operations in this list may be acting on the same wire.
    Raises:
        TypeError: if any elements of ``pauli_operators`` are not instances of
            :class:`~.PauliX`, :class:`~.PauliY`, :class:`~.PauliZ`, or :class:`~.Identity`

    **Example**

    >>> pauli_operators = [qml.PauliX('a'), qml.PauliY('b'), qml.PauliZ('c')]
    >>> qwc_rotation(pauli_operators)
    [RY(-1.5707963267948966, wires=['a']), RX(1.5707963267948966, wires=['b'])]
    """
    paulis_with_identity = (qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ)
    if not all(isinstance(element, paulis_with_identity) for element in pauli_operators):
        raise TypeError(
            f"All values of input pauli_operators must be either Identity, PauliX, PauliY, or PauliZ instances,"
            f" instead got pauli_operators = {pauli_operators}."
        )
    with qml.tape.OperationRecorder() as rec:

        for pauli in pauli_operators:
            if isinstance(pauli, qml.PauliX):
                qml.RY(-np.pi / 2, wires=pauli.wires)

            elif isinstance(pauli, qml.PauliY):
                qml.RX(np.pi / 2, wires=pauli.wires)

    return rec.queue


def diagonalize_pauli_word(pauli_word):
    """Transforms the Pauli word to diagonal form in the computational basis.

    Args:
        pauli_word (Observable): the Pauli word to diagonalize in computational basis

    Returns:
        Observable: the Pauli word diagonalized in the computational basis

    Raises:
        TypeError: if the input is not a Pauli word, i.e., a Pauli operator,
            :class:`~.Identity`, or :class:`~.Tensor` instances thereof

    **Example**

    >>> diagonalize_pauli_word(qml.PauliX('a') @ qml.PauliY('b') @ qml.PauliZ('c'))
    PauliZ(wires=['a']) @ PauliZ(wires=['b']) @ PauliZ(wires=['c'])
    """

    if not is_pauli_word(pauli_word):
        raise TypeError(f"Input must be a Pauli word, instead got: {pauli_word}.")

    paulis_with_identity = (qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity)
    diag_term = None

    if isinstance(pauli_word, Tensor):
        for sigma in pauli_word.obs:
            if sigma.name != "Identity":
                if diag_term is None:
                    diag_term = qml.PauliZ(wires=sigma.wires)
                else:
                    diag_term @= qml.PauliZ(wires=sigma.wires)

    elif isinstance(pauli_word, paulis_with_identity):
        sigma = pauli_word
        if sigma.name != "Identity":
            diag_term = qml.PauliZ(wires=sigma.wires)

    if diag_term is None:
        diag_term = qml.Identity(pauli_word.wires.tolist()[0])

    return diag_term


def diagonalize_qwc_pauli_words(qwc_grouping):
    """Diagonalizes a list of mutually qubit-wise commutative Pauli words.

    Args:
        qwc_grouping (list[Observable]): a list of observables containing mutually
            qubit-wise commutative Pauli words

    Returns:
        tuple:

            * list[Operation]: an instance of the qwc_rotation template which
              diagonalizes the qubit-wise commuting grouping
            * list[Observable]: list of Pauli string observables diagonal in
              the computational basis

    Raises:
        ValueError: if any 2 elements in the input QWC grouping are not qubit-wise commutative

    **Example**

    >>> qwc_group = [qml.PauliX(0) @ qml.PauliZ(1),
                     qml.PauliX(0) @ qml.PauliY(3),
                     qml.PauliZ(1) @ qml.PauliY(3)]
    >>> diagonalize_qwc_pauli_words(qwc_group)
    ([RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[3])],
     [PauliZ(wires=[0]) @ PauliZ(wires=[1]),
      PauliZ(wires=[0]) @ PauliZ(wires=[3]),
      PauliZ(wires=[1]) @ PauliZ(wires=[3])])
    """
    m_paulis = len(qwc_grouping)
    all_wires = Wires.all_wires([pauli_word.wires for pauli_word in qwc_grouping])
    wire_map = {label: ind for ind, label in enumerate(all_wires)}
    for i in range(m_paulis):
        for j in range(i + 1, m_paulis):
            if not is_qwc(
                pauli_to_binary(qwc_grouping[i], wire_map=wire_map),
                pauli_to_binary(qwc_grouping[j], wire_map=wire_map),
            ):
                raise ValueError(
                    f"{qwc_grouping[i]} and {qwc_grouping[j]} are not qubit-wise commuting."
                )

    pauli_operators = []
    diag_terms = []

    paulis_with_identity = (qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity)
    for term in qwc_grouping:
        diag_terms.append(diagonalize_pauli_word(term))
        if isinstance(term, Tensor):
            for sigma in term.obs:
                if sigma.name != "Identity":
                    if not any(
                        are_identical_pauli_words(sigma, existing_pauli)
                        for existing_pauli in pauli_operators
                    ):
                        pauli_operators.append(sigma)
        elif isinstance(term, paulis_with_identity):
            sigma = term
            if sigma.name != "Identity":
                if not any(
                    are_identical_pauli_words(sigma, existing_pauli)
                    for existing_pauli in pauli_operators
                ):
                    pauli_operators.append(sigma)

    unitary = qwc_rotation(pauli_operators)

    return unitary, diag_terms


def diagonalize_qwc_groupings(qwc_groupings):
    """Diagonalizes a list of qubit-wise commutative groupings of Pauli strings.

    Args:
        qwc_groupings (list[list[Observable]]): a list of mutually qubit-wise commutative groupings
            of Pauli string observables

    Returns:
        tuple:

            * list[list[Operation]]: a list of instances of the qwc_rotation
              template which diagonalizes the qubit-wise commuting grouping,
              order corresponding to qwc_groupings
            * list[list[Observable]]: a list of QWC groupings diagonalized in the
              computational basis, order corresponding to qwc_groupings

    **Example**

    >>> qwc_group_1 = [qml.PauliX(0) @ qml.PauliZ(1),
                       qml.PauliX(0) @ qml.PauliY(3),
                       qml.PauliZ(1) @ qml.PauliY(3)]
    >>> qwc_group_2 = [qml.PauliY(0),
                       qml.PauliY(0) @ qml.PauliX(2),
                       qml.PauliX(1) @ qml.PauliZ(3)]
    >>> post_rotations, diag_groupings = diagonalize_qwc_groupings([qwc_group_1, qwc_group_2])
    >>> post_rotations
    [[RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[3])],
     [RX(1.5707963267948966, wires=[0]),
      RY(-1.5707963267948966, wires=[2]),
      RY(-1.5707963267948966, wires=[1])]]
    >>> diag_groupings
    [[PauliZ(wires=[0]) @ PauliZ(wires=[1]),
     PauliZ(wires=[0]) @ PauliZ(wires=[3]),
     PauliZ(wires=[1]) @ PauliZ(wires=[3])],
    [PauliZ(wires=[0]),
     PauliZ(wires=[0]) @ PauliZ(wires=[2]),
     PauliZ(wires=[1]) @ PauliZ(wires=[3])]]
    """

    post_rotations = []
    diag_groupings = []
    m_groupings = len(qwc_groupings)

    for i in range(m_groupings):

        diagonalizing_unitary, diag_grouping = diagonalize_qwc_pauli_words(qwc_groupings[i])
        post_rotations.append(diagonalizing_unitary)
        diag_groupings.append(diag_grouping)

    return post_rotations, diag_groupings
