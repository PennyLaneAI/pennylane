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
Functions for constructing the :math:`n`-qubit Pauli group, and performing the
group operation (multiplication).
"""
import itertools
from functools import lru_cache
from typing import List

import numpy as np

from pennylane import Identity
from pennylane.grouping.utils import (
    _wire_map_from_pauli_pair,
    are_identical_pauli_words,
    binary_to_pauli,
    pauli_to_binary,
    pauli_word_to_string,
)


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
            If no wire map is provided, wires will be labeled by consecutive integers between :math:`0` and ``n_qubits``.

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
    """Generate the :math:`n`-qubit Pauli group.

    This function enables the construction of the :math:`n`-qubit Pauli group with no
    storage involved.  The :math:`n`-qubit Pauli group has size :math:`4^n`,
    thus it may not be desirable to construct it in full and store.

    The order of iteration is based on the binary symplectic representation of
    the Pauli group as :math:`2n`-bit strings. Ordering is done by converting
    the integers :math:`0` to :math:`2^{2n}` to binary strings, and converting those
    strings to Pauli operators using the :func:`~.binary_to_pauli` method.

    Args:
        n_qubits (int): The number of qubits for which to create the group.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels
            used in the Pauli word as keys, and unique integer labels as their values.
            If no wire map is provided, wires will be labeled by integers between 0 and ``n_qubits``.

    Returns:
        .Operation: The next Pauli word in the group.

    **Example**

    The ``pauli_group`` generator can be used to loop over the Pauli group as follows.
    (Note: in the example below, we display only the first 5 elements for brevity.)

    >>> from pennylane.grouping import pauli_group
    >>> n_qubits = 3
    >>> for p in pauli_group(n_qubits):
    ...     print(p)
    ...
    Identity(wires=[0])
    PauliZ(wires=[2])
    PauliZ(wires=[1])
    PauliZ(wires=[1]) @ PauliZ(wires=[2])
    PauliZ(wires=[0])

    The full Pauli group can then be obtained like so:

    >>> full_pg = list(pauli_group(n_qubits))

    The group can also be created using a custom wire map; if no map is
    specified, a default map of label :math:`i` to wire ``i`` as in the example
    above will be created. (Note: in the example below, we display only the first
    5 elements for brevity.)

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> for p in pauli_group(n_qubits, wire_map=wire_map):
    ...     print(p)
    ...
    Identity(wires=['a'])
    PauliZ(wires=['c'])
    PauliZ(wires=['b'])
    PauliZ(wires=['b']) @ PauliZ(wires=['c'])
    PauliZ(wires=['a'])

    """
    # Cover the case where n_qubits may be passed as a float
    if isinstance(n_qubits, float):
        if n_qubits.is_integer():
            n_qubits = int(n_qubits)

    # If not an int, or a float representing a int, raise an error
    if not isinstance(n_qubits, int):
        raise TypeError("Must specify an integer number of qubits to construct the Pauli group.")

    if n_qubits <= 0:
        raise ValueError("Number of qubits must be at least 1 to construct Pauli group.")

    return _pauli_group_generator(n_qubits, wire_map=wire_map)


def pauli_mult(pauli_1, pauli_2, wire_map=None):
    """Multiply two Pauli words together and return the product as a Pauli word.

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

    >>> from pennylane.grouping import pauli_mult
    >>> pauli_1 = qml.PauliX(0) @ qml.PauliZ(1)
    >>> pauli_2 = qml.PauliY(0) @ qml.PauliZ(1)
    >>> product = pauli_mult(pauli_1, pauli_2)
    >>> print(product)
    PauliZ(wires=[0])
    """

    if wire_map is None:
        wire_map = _wire_map_from_pauli_pair(pauli_1, pauli_2)

    # Check if pauli_1 and pauli_2 are the same; if so, the result is the Identity
    if are_identical_pauli_words(pauli_1, pauli_2):
        first_wire = list(pauli_1.wires)[0]
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
    r"""Multiply two Pauli words together, and return both their product as a Pauli word
    and the global phase.

    Two Pauli operations can be multiplied together by taking the additive
    OR of their binary symplectic representations. The phase is computed by
    looking at the number of times we have the products  :math:`XY, YZ`, or :math:`ZX` (adds a
    phase of :math:`i`), or :math:`YX, ZY, XZ` (adds a phase of :math:`-i`).

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

    This function works the same as :func:`~.pauli_mult` but also returns the global
    phase accumulated as a result of the ordering of Paulis in the product (e.g., :math:`XY = iZ`,
    and :math:`YX = -iZ`).

    >>> from pennylane.grouping import pauli_mult_with_phase
    >>> pauli_1 = qml.PauliX(0) @ qml.PauliZ(1)
    >>> pauli_2 = qml.PauliY(0) @ qml.PauliZ(1)
    >>> product, phase = pauli_mult_with_phase(pauli_1, pauli_2)
    >>> product
    PauliZ(wires=[0])
    >>> phase
    1j
    """

    if wire_map is None:
        wire_map = _wire_map_from_pauli_pair(pauli_1, pauli_2)

    # Get the product; use our earlier function
    pauli_product = pauli_mult(pauli_1, pauli_2, wire_map)

    # Get the names of the operations; in cases where only one single-qubit Pauli
    # is present, the operation name is stored as a string rather than a list, so convert it
    pauli_1_string = pauli_word_to_string(pauli_1, wire_map=wire_map)
    pauli_2_string = pauli_word_to_string(pauli_2, wire_map=wire_map)

    pos_phases = (("X", "Y"), ("Y", "Z"), ("Z", "X"))

    phase = 1

    for qubit_1_char, qubit_2_char in zip(pauli_1_string, pauli_2_string):
        # If we have identities anywhere we don't pick up a phase
        if qubit_1_char == "I" or qubit_2_char == "I":
            continue

        # Likewise, no additional phase if the Paulis are the same
        if qubit_1_char == qubit_2_char:
            continue

        # Use Pauli commutation rules to determine the phase
        if (qubit_1_char, qubit_2_char) in pos_phases:
            phase *= 1j
        else:
            phase *= -1j

    return pauli_product, phase


@lru_cache()
def partition_pauli_group(n_qubits: int) -> List[List[str]]:
    """Partitions the :math:`n`-qubit Pauli group into qubit-wise commuting terms.

    The :math:`n`-qubit Pauli group is composed of :math:`4^{n}` terms that can be partitioned into
    :math:`3^{n}` qubit-wise commuting groups.

    Args:
        n_qubits (int): number of qubits

    Returns:
        List[List[str]]: A collection of qubit-wise commuting groups containing Pauli words as
        strings

    **Example**

    >>> qml.grouping.partition_pauli_group(3)
    [['III', 'IIZ', 'IZI', 'IZZ', 'ZII', 'ZIZ', 'ZZI', 'ZZZ'],
     ['IIX', 'IZX', 'ZIX', 'ZZX'],
     ['IIY', 'IZY', 'ZIY', 'ZZY'],
     ['IXI', 'IXZ', 'ZXI', 'ZXZ'],
     ['IXX', 'ZXX'],
     ['IXY', 'ZXY'],
     ['IYI', 'IYZ', 'ZYI', 'ZYZ'],
     ['IYX', 'ZYX'],
     ['IYY', 'ZYY'],
     ['XII', 'XIZ', 'XZI', 'XZZ'],
     ['XIX', 'XZX'],
     ['XIY', 'XZY'],
     ['XXI', 'XXZ'],
     ['XXX'],
     ['XXY'],
     ['XYI', 'XYZ'],
     ['XYX'],
     ['XYY'],
     ['YII', 'YIZ', 'YZI', 'YZZ'],
     ['YIX', 'YZX'],
     ['YIY', 'YZY'],
     ['YXI', 'YXZ'],
     ['YXX'],
     ['YXY'],
     ['YYI', 'YYZ'],
     ['YYX'],
     ['YYY']]
    """
    # Cover the case where n_qubits may be passed as a float
    if isinstance(n_qubits, float):
        if n_qubits.is_integer():
            n_qubits = int(n_qubits)

    # If not an int, or a float representing a int, raise an error
    if not isinstance(n_qubits, int):
        raise TypeError("Must specify an integer number of qubits.")

    if n_qubits < 0:
        raise ValueError("Number of qubits must be at least 0.")

    if n_qubits == 0:
        return [[""]]

    strings = set()  # tracks all the strings that have already been grouped
    groups = []

    # We know that I and Z always commute on a given qubit. The following generates all product
    # sequences of len(n_qubits) over "FXYZ", with F indicating a free slot that can be swapped for
    # the product over I and Z, and all other terms fixed to the given X/Y/Z. For example, if
    # ``n_qubits = 3`` our first value for ``string`` will be ``('F', 'F', 'F')``. We then expand
    # the product of I and Z over the three free slots, giving
    # ``['III', 'IIZ', 'IZI', 'IZZ', 'ZII', 'ZIZ', 'ZZI', 'ZZZ']``, which is our first group. The
    # next element of ``string`` will be ``('F', 'F', 'X')`` which we use to generate our second
    # group ``['IIX', 'IZX', 'ZIX', 'ZZX']``.
    for string in itertools.product("FXYZ", repeat=n_qubits):
        if string not in strings:
            num_free_slots = string.count("F")

            group = []
            commuting = itertools.product("IZ", repeat=num_free_slots)

            for commuting_string in commuting:
                commuting_string = list(commuting_string)
                new_string = tuple(commuting_string.pop(0) if s == "F" else s for s in string)

                if new_string not in strings:  # only add if string has not already been grouped
                    group.append("".join(new_string))
                    strings |= {new_string}

            if len(group) > 0:
                groups.append(group)

    return groups
