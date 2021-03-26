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
Functions for constructing the :math:`n`-qubit Pauli group, and performing the
group operation (multiplication).
"""

import numpy as np

from pennylane import Identity
from pennylane.groups.pauli_utils import binary_to_pauli, pauli_to_binary, are_identical_pauli_words


def _pauli_group_generator(n_qubits, wire_map=None):
    """Generator function for the Pauli group.

    This function is called by ``pauli_group`` in order to actually generate the
    group elements. They are split so that the outer function can handle input
    validation, while this generator is responsible for performing the actual
    operations.

    Args:
        n_qubits (int): The number of qubits for which to create the group.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels
            used in the Pauli word as keys, and unique integer labels as their values.
            If no wire map is provided, wires will be labeled by integers between 0 and ``n_qubits``.

    Returns:
        .Operation: The next Pauli word in the group.
    """

    element_idx = 0

    if not wire_map:
        wire_map = {wire_idx: wire_idx for wire_idx in range(n_qubits)}

    while element_idx < 4 ** n_qubits:
        binary_string = format(element_idx, f"#0{2*n_qubits+2}b")[2:]
        binary_vector = [float(b) for b in binary_string]
        yield binary_to_pauli(binary_vector, wire_map=wire_map)
        element_idx += 1


def pauli_group(n_qubits, wire_map=None):
    """Iterating over the :math:`n`-qubit Pauli group.

    This function allows for iteration over elements of the Pauli group with no
    storage involved.  The :math:`n`-qubit Pauli group has size :math:`4^n`,
    thus it may not be desirable to construct it in full and store.

    The order of iteration is based on the binary symplectic representation of
    the Pauli group as :math:`2n`-bit strings. Ordering is done by converting
    the integers :math:`0` to :math:`2^{2n}` to binary strings, and converting those
    strings to Pauli operators using the ``binary_to_pauli`` method.

    Args:
        n_qubits (int): The number of qubits for which to create the group.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels
            used in the Pauli word as keys, and unique integer labels as their values.
            If no wire map is provided, wires will be labeled by integers between 0 and ``n_qubits``.

    Returns:
        .Operation: The next Pauli word in the group.

    **Example**

    The ``pauli_group`` generator can be used to loop over the Pauli group as follows:

    .. code-block:: python

        from pennylane.groups.pauli import pauli_group

        n_qubits = 3

        for p in pauli_group(n_qubits):
            print(p)

    The Pauli group in full can be obtained in full like so:

    .. code-block:: python

       full_pg = list(pauli_group(n_qubits))

    The group can also be created using a custom wire map (if no map is
    specified, a default map of label :math:`i` to wire ``i`` will be created).

    .. code-block:: python

        n_qubits = 3
        wire_map = {'a' : 0, 'b' : 1, 'c' : 2}

        for p in pauli_group(n_qubits, wire_map=wire_map):
            print(p)

    """
    if not isinstance(n_qubits, int):
        raise TypeError("Must specify an integer number of qubits construct the Pauli group.")

    if n_qubits <= 0:
        raise ValueError("Number of qubits must be at least 1 to construct Pauli group.")

    return _pauli_group_generator(n_qubits, wire_map=wire_map)


def pauli_mult(pauli_1, pauli_2, wire_map=None):
    """Multiply two Pauli words together.

    Two Pauli operations can be multiplied together by taking the additive
    OR of their binary symplectic representations.

    Args:
        pauli_1 (.Operation): A Pauli word.
        pauli_2 (.Operation): A Pauli word to multiply with the first one.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in the Pauli
            word as keys, and unique integer labels as their values. If no wire map is
            provided, the map will be constructed from the set of wires acted on
            by the input Pauli words.

    Returns:
        .Operation: The product of pauli_1 and pauli_2 as a Pauli word
            (ignoring the global phase).

    **Example**

    This function enables multiplication of Pauli group elements at the level of
    Pauli words, rather than matrices. For example,

    >>> from pennylane.groups.pauli import pauli_mult
    >>> pauli_1 = qml.PauliX(0) @ qml.PauliZ(1)
    >>> pauli_2 = qml.PauliY(0) @ qml.PauliZ(1)
    >>> product = pauli_mult(pauli_1, pauli_2)
    >>> print(product)
    PauliZ(wires=[0])
    """

    # If no wire map is specified, generate one from the union of wires
    # in both Paulis.
    if wire_map is None:
        wire_labels = set(pauli_1.wires.labels + pauli_2.wires.labels)
        wire_map = {label: i for i, label in enumerate(wire_labels)}

    # Check if pauli_1 and pauli_2 are the same; if so, the result is the Identity
    if are_identical_pauli_words(pauli_1, pauli_2):
        first_wire = list(wire_map.keys())[0]
        return Identity(first_wire)

    # Compute binary symplectic representations
    pauli_1_binary = pauli_to_binary(pauli_1, wire_map=wire_map)
    pauli_2_binary = pauli_to_binary(pauli_2, wire_map=wire_map)

    bin_symp_1 = np.array([int(x) for x in pauli_1_binary])
    bin_symp_2 = np.array([int(x) for x in pauli_2_binary])

    # Shorthand for bitwise XOR of numpy arrays
    pauli_product = bin_symp_1 ^ bin_symp_2

    return binary_to_pauli(pauli_product, wire_map=wire_map)


def pauli_mult_with_phase(pauli_1, pauli_2, wire_map=None):
    r"""Multiply two Pauli words together including the global phase.

    Two Pauli operations can be multiplied together by taking the additive
    OR of their binary symplectic representations. The phase is computed by
    looking at the number of times we have the products  XY, YZ, or ZX (adds a
    phase of :math:`i`), or YX, ZY, XZ (adds a phase of :math:`-i`).

    Args:
        pauli_1 (.Operation): A Pauli word.
        pauli_2 (.Operation): A Pauli word to multiply with the first one.
        wire_map  (dict[Union[str, int], int]): dictionary containing all wire labels used in the Pauli
            word as keys, and unique integer labels as their values. If no wire map is
            provided, the map will be constructed from the set of wires acted on
            by the input Pauli words.

    Returns:
        tuple[.Operation, complex]: The product of ``pauli_1`` and ``pauli_2``, and the
            global phase.

    **Example**

    This function works the same as ``pauli_mult`` but also returns the global
    phase accumulated as a result of the Pauli product rules
    :math:`\sigma_i \sigma_j = i \sigma_k`.

    >>> from pennylane.groups.pauli import pauli_mult_with_phase
    >>> pauli_1 = qml.PauliX(0) @ qml.PauliZ(1)
    >>> pauli_2 = qml.PauliY(0) @ qml.PauliZ(1)
    >>> product, phase = pauli_mult_with_phase(pauli_1, pauli_2)
    >>> product
    PauliZ(wires=[0])
    >>> phase
    1j
    """

    # If no wire map is specified, generate one from the union of wires
    # in both Paulis.
    if wire_map is None:
        wire_labels = set(pauli_1.wires.labels + pauli_2.wires.labels)
        wire_map = {label: i for i, label in enumerate(wire_labels)}

    # Get the product; use our earlier function
    pauli_product = pauli_mult(pauli_1, pauli_2, wire_map)

    pauli_1_names = [pauli_1.name] if isinstance(pauli_1.name, str) else pauli_1.name
    pauli_2_names = [pauli_2.name] if isinstance(pauli_2.name, str) else pauli_2.name

    pauli_1_placeholder = 0
    pauli_2_placeholder = 0

    phase = 1

    for wire in wire_map.keys():
        if wire in pauli_1.wires:
            pauli_1_op_name = pauli_1_names[pauli_1_placeholder]
            pauli_1_placeholder += 1
        else:
            pauli_1_op_name = "Identity"

        if wire in pauli_2.wires:
            pauli_2_op_name = pauli_2_names[pauli_2_placeholder]
            pauli_2_placeholder += 1
        else:
            pauli_2_op_name = "Identity"

        # If we have identities anywhere we don't pick up a phase
        if pauli_1_op_name == "Identity" or pauli_2_op_name == "Identity":
            continue

        # Likewise, no additional phase if the Paulis are the same
        if pauli_1_op_name == pauli_2_op_name:
            continue

        # Use Pauli commutation rules to determine the phase
        pauli_ordering = (pauli_1_op_name, pauli_2_op_name)

        if pauli_ordering in [("PauliX", "PauliY"), ("PauliY", "PauliZ"), ("PauliZ", "PauliX")]:
            phase *= 1j
        else:
            phase *= -1j

    return pauli_product, phase
