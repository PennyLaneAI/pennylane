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
Functions for generating and manipulating elements of the Pauli group.
"""

import numpy as np

from pennylane import Identity
from pennylane.grouping.utils import (
    binary_to_pauli,
    pauli_to_binary,
    are_identical_pauli_words
)

def pauli_group_generator(n_qubits):
    """Generator for iterating over the n-qubit Pauli group.

    Args:
        n_qubits (int): The number of qubits for which to create the group.

    Returns:
        (qml.Operation): The next Pauli word in the group.
    """
    element_idx = 0

    while element_idx < 4 ** n_qubits:
        binary_string = format(element_idx, f"#0{2*n_qubits+2}b")[2:]
        binary_vector = [float(b) for b in binary_string]
        yield binary_to_pauli(binary_vector)
        element_idx += 1

def pauli_group(n_qubits):
    """ The n-qubit Pauli group.

    Args:
        n_qubits (int): The number of qubits for which to create the group.

    Returns:
        (list[qml.Operation]): The full n-qubit Pauli group.
    """

    return list(pauli_group_generator(n_qubits))


def pauli_mult(pauli_1, pauli_2, n_qubits=None, wire_map=None):
    """ Multiply two Pauli words together.

    Two Pauli operations can be multiplied together by taking the additive
    OR of their binary symplectic representations.

    Args:
        pauli_1 (qml.Operation): A Pauli word.
        pauli_2 (qml.Operation): A Pauli word to multiply with the first one.

    Returns:
        (qml.Operation): The product of pauli_1 and pauli_2
        as a Pauli word (ignoring the global phase).
    """
    # Check if pauli_1 and pauli_2 are the same; if so, the result is the Identity
    if are_identical_pauli_words(pauli_1, pauli_2):
        return Identity(0)

    # TODO: cover the case where the wires are not numbered like this

    # Expand Paulis so they have the same number of qubits
    # Grab the max index of the wires among the two Paulis
    if not n_qubits:
        n_qubits = max(pauli_1.wires + pauli_2.wires) + 1

    if not wire_map:
        wire_map = {x : x for x in range(n_qubits)}

    # Compute binary symplectic representations
    pauli_1_binary = pauli_to_binary(pauli_1, n_qubits=n_qubits, wire_map=wire_map)
    pauli_2_binary = pauli_to_binary(pauli_2, n_qubits=n_qubits, wire_map=wire_map)

    bin_symp_1 = np.array([int(x) for x in pauli_1_binary])
    bin_symp_2 = np.array([int(x) for x in pauli_2_binary])

    # Shorthand for bitwise XOR of numpy arrays
    pauli_product = bin_symp_1 ^ bin_symp_2

    return binary_to_pauli(pauli_product)


def pauli_mult_with_phase(pauli_1, pauli_2, n_qubits=None, wire_map=None):
    """ Multiply two Pauli words together including the global phase.

    Two Pauli operations can be multiplied together by taking the additive
    OR of their binary symplectic representations. The phase is computed by
    looking at the number of times we have the products  XY, YZ, or ZX (adds a
    phase of +1j), or YX, ZY, XZ (adds a phase of -1j).

    Args:
        pauli_1 (qml.Operation): A Pauli word.
        pauli_2 (qml.Operation): A Pauli word to multiply with the first one.

    Returns:
        (qml.Operation, np.complex): The product of pauli_1 and pauli_2, and the global phase.
    """

    # Get the product; use our earlier function
    pauli_product = pauli_mult(pauli_1, pauli_2, n_qubits, wire_map)

    if not n_qubits:
        n_qubits = max(pauli_1.wires + pauli_2.wires) + 1

    pauli_1_names = [pauli_1.name] if isinstance(pauli_1.name, str) else pauli_1.name
    pauli_2_names = [pauli_2.name] if isinstance(pauli_2.name, str) else pauli_2.name

    pauli_1_placeholder = 0
    pauli_2_placeholder = 0

    phase = 1

    for wire_idx in range(n_qubits):
        if wire_idx in pauli_1.wires:
            pauli_1_op_name = pauli_1_names[pauli_1_placeholder]
            pauli_1_placeholder += 1
        else:
            pauli_1_op_name = 'Identity'

        if wire_idx in pauli_2.wires:
            pauli_2_op_name = pauli_2_names[pauli_2_placeholder]
            pauli_2_placeholder += 1
        else:
            pauli_2_op_name = 'Identity'

        if pauli_1_op_name is not 'Identity' and pauli_2_op_name is not 'Identity':
            if pauli_1_op_name > pauli_2_op_name:
                phase *= -1j
            else:
                phase *= 1j

    return pauli_product, phase


def pauli_matrix(pauli, n_qubits=None):
    return
