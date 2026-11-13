# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert between Pauli operators and strings."""

from pennylane.ops import Identity, PauliX, PauliY, PauliZ
from pennylane.ops.op_math import prod

from .operator_to_pauli import is_pauli_word

_PAULI_MAP = {"X": PauliX, "Y": PauliY, "Z": PauliZ}
_PAULI_STRING_SET = frozenset(("I", "X", "Y", "Z"))


def pauli_word_to_string(pauli_word, wire_map=None):
    """Convert a Pauli word to a string.

    A Pauli word can be either:

    * A single pauli operator (see :class:`~.PauliX` for an example).

    * A :class:`.Prod` instance containing Pauli operators.

    * A :class:`.SProd` instance containing a Pauli operator.

    * A :class:`.Sum` instance with only one term.

    Given a Pauli in observable form, convert it into string of
    characters from ``['I', 'X', 'Y', 'Z']``. This representation is required for
    functions such as :class:`.PauliRot`.

    .. warning::

        This method ignores any potential coefficient multiplying the Pauli word:

        >>> qp.pauli.pauli_word_to_string(3 * qp.X(0) @ qp.Y(1))
        'XY'

    .. warning::

        This method assumes all Pauli operators are acting on different wires, ignoring
        any extra operators:

        >>> qp.pauli.pauli_word_to_string(qp.X(0) @ qp.Y(0) @ qp.Y(0))
        'X'

    Args:
        pauli_word (Operator): an observable, either a single-qubit observable
            representing a Pauli group element, or a tensor product of single-qubit observables.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        str: The string representation of the observable in terms of ``'I'``, ``'X'``, ``'Y'``,
        and/or ``'Z'``.

    Raises:
        TypeError: if the input observable is not a proper Pauli word.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> pauli_word = qp.X('a') @ qp.Y('c')
    >>> pauli_word_to_string(pauli_word, wire_map=wire_map)
    'XIY'
    """

    if not is_pauli_word(pauli_word):
        raise TypeError(f"Expected Pauli word observables, instead got {pauli_word}")

    pr = next(iter(pauli_word.pauli_rep.keys()))

    # If there is no wire map, we must infer from the structure of Paulis
    if wire_map is None:
        wire_map = {pauli_word.wires.labels[i]: i for i in range(len(pauli_word.wires))}

    n_qubits = len(wire_map)

    # Set default value of all characters to identity
    pauli_string = ["I"] * n_qubits

    for wire, op_label in pr.items():
        pauli_string[wire_map[wire]] = op_label

    return "".join(pauli_string)


def string_to_pauli_word(pauli_string, wire_map=None):
    """Convert a string in terms of ``'I'``, ``'X'``, ``'Y'``, and ``'Z'`` into a Pauli word
    for the given wire map.

    Args:
        pauli_string (str): A string of characters consisting of ``'I'``, ``'X'``, ``'Y'``, and ``'Z'``
            indicating a Pauli word.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        .Operator: The Pauli word representing of ``pauli_string`` on the wires
        enumerated in the wire map.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> string_to_pauli_word('XIY', wire_map=wire_map)
    X('a') @ Y('c')
    """

    if not isinstance(pauli_string, str):
        raise TypeError(f"Input to string_to_pauli_word must be string, obtained {pauli_string}")

    # If no wire map is provided, construct one using integers based on the length of the string
    if wire_map is None:
        wire_map = {x: x for x in range(len(pauli_string))}

    elif len(pauli_string) != len(wire_map):
        raise ValueError(
            "Wire map and pauli_string must have the same length to convert "
            "from string to Pauli word."
        )

    # Special case: all-identity Pauli
    if pauli_string == "I" * len(wire_map):
        first_wire = list(wire_map)[0]
        return Identity(first_wire)

    if any(char not in _PAULI_STRING_SET for char in pauli_string):
        raise ValueError(
            "Invalid characters encountered in string_to_pauli_word "
            f"string {pauli_string}. Permitted characters are 'I', 'X', 'Y', and 'Z'"
        )

    pauli_word = tuple(
        _PAULI_MAP[pauli_string[wire_idx]](wire_name)
        for wire_name, wire_idx in wire_map.items()
        if pauli_string[wire_idx] != "I"
    )

    return prod(*pauli_word)
