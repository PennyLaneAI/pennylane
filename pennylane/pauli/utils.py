# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Utility functions used in Pauli arithmetic, partitioning, and measurement reduction schemes utilizing the
symplectic vector-space representation of Pauli words. For information on the symplectic binary
representation of Pauli words and applications, see:

* `arXiv:quant-ph/9705052 <https://arxiv.org/abs/quant-ph/9705052>`_
* `arXiv:1701.08213 <https://arxiv.org/abs/1701.08213>`_
* `arXiv:1907.09386 <https://arxiv.org/abs/1907.09386>`_
"""
from functools import lru_cache, singledispatch
from itertools import product
from typing import List, Union

import numpy as np

import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import (
    Hamiltonian,
    LinearCombination,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    Prod,
    SProd,
)
from pennylane.wires import Wires

# To make this quicker later on
ID_MAT = np.eye(2)


def _wire_map_from_pauli_pair(pauli_word_1, pauli_word_2):
    """Generate a wire map from the union of wires of two Paulis.

    Args:
        pauli_word_1 (.Operation): A Pauli word.
        pauli_word_2 (.Operation): A second Pauli word.

    Returns:
        dict[Union[str, int], int]): dictionary containing all wire labels used
        in the Pauli word as keys, and unique integer labels as their values.
    """
    wire_labels = Wires.all_wires([pauli_word_1.wires, pauli_word_2.wires]).labels
    return {label: i for i, label in enumerate(wire_labels)}


def is_pauli_word(observable):
    """
    Checks if an observable instance consists only of Pauli and Identity Operators.

    A Pauli word can be either:

    * A single pauli operator (see :class:`~.PauliX` for an example).

    * A :class:`.Tensor` instance containing Pauli operators.

    * A :class:`.Prod` instance containing Pauli operators.

    * A :class:`.SProd` instance containing a valid Pauli word.

    * A :class:`.Hamiltonian` instance with only one term.

    .. Warning::

        This function will only confirm that all operators are Pauli or Identity operators,
        and not whether the Observable is mathematically a Pauli word.
        If an Observable consists of multiple Pauli operators targeting the same wire, the
        function will return ``True`` regardless of any complex coefficients.


    Args:
        observable (~.Operator): the operator to be examined

    Returns:
        bool: true if the input observable is a Pauli word, false otherwise.

    **Example**

    >>> is_pauli_word(qml.Identity(0))
    True
    >>> is_pauli_word(qml.X(0) @ qml.Z(2))
    True
    >>> is_pauli_word(qml.Z(0) @ qml.Hadamard(1))
    False
    >>> is_pauli_word(4 * qml.X(0) @ qml.Z(0))
    True
    """
    return _is_pauli_word(observable) or (len(observable.pauli_rep or []) == 1)


@singledispatch
def _is_pauli_word(observable):  # pylint:disable=unused-argument
    """
    Private implementation of is_pauli_word, to prevent all of the
    registered functions from appearing in the Sphinx docs.
    """
    return False


@_is_pauli_word.register(PauliX)
@_is_pauli_word.register(PauliY)
@_is_pauli_word.register(PauliZ)
@_is_pauli_word.register(Identity)
def _is_pw_pauli(
    observable: Union[PauliX, PauliY, PauliZ, Identity]
):  # pylint:disable=unused-argument
    return True


@_is_pauli_word.register
def _is_pw_tensor(observable: Tensor):
    pauli_word_names = ["Identity", "PauliX", "PauliY", "PauliZ"]
    return set(observable.name).issubset(pauli_word_names)


@_is_pauli_word.register(Hamiltonian)
@_is_pauli_word.register(LinearCombination)
def _is_pw_ham(observable: Union[Hamiltonian, LinearCombination]):
    return False if len(observable.ops) != 1 else is_pauli_word(observable.ops[0])


@_is_pauli_word.register
def _is_pw_prod(observable: Prod):
    return all(is_pauli_word(op) for op in observable)


@_is_pauli_word.register
def _is_pw_sprod(observable: SProd):
    return is_pauli_word(observable.base)


def are_identical_pauli_words(pauli_1, pauli_2):
    # pylint: disable=isinstance-second-argument-not-valid-type
    """Performs a check if two Pauli words have the same ``wires`` and ``name`` attributes.

    This is a convenience function that checks if two given :class:`~.Tensor` or :class:`~.Prod`
    instances specify the same Pauli word.

    Args:
        pauli_1 (Union[Identity, PauliX, PauliY, PauliZ, Tensor, Prod, SProd]): the first Pauli word
        pauli_2 (Union[Identity, PauliX, PauliY, PauliZ, Tensor, Prod, SProd]): the second Pauli word

    Returns:
        bool: whether ``pauli_1`` and ``pauli_2`` have the same wires and name attributes

    Raises:
        TypeError: if ``pauli_1`` or ``pauli_2`` are not :class:`~.Identity`, :class:`~.PauliX`,
            :class:`~.PauliY`, :class:`~.PauliZ`, :class:`~.Tensor`, :class:`~.SProd`, or
            :class:`~.Prod` instances

    **Example**

    >>> are_identical_pauli_words(qml.Z(0) @ qml.Z(1), qml.Z(0) @ qml.Z(1))
    True
    >>> are_identical_pauli_words(qml.Z(0) @ qml.Z(1), qml.Z(0) @ qml.X(3))
    False
    """
    if pauli_1.name == "Hamiltonian" or pauli_2.name == "Hamiltonian":
        return False
    if not (is_pauli_word(pauli_1) and is_pauli_word(pauli_2)):
        raise TypeError(f"Expected Pauli word observables, instead got {pauli_1} and {pauli_2}.")

    if pauli_1.pauli_rep is not None and pauli_2.pauli_rep is not None:
        return next(iter(pauli_1.pauli_rep)) == next(iter(pauli_2.pauli_rep))

    return False


def pauli_to_binary(pauli_word, n_qubits=None, wire_map=None, check_is_pauli_word=True):
    # pylint: disable=isinstance-second-argument-not-valid-type
    """Converts a Pauli word to the binary vector representation.

    This functions follows convention that the first half of binary vector components specify
    PauliX placements while the last half specify PauliZ placements.

    Args:
        pauli_word (Union[Identity, PauliX, PauliY, PauliZ, Tensor, Prod, SProd]): the Pauli word to be
            converted to binary vector representation
        n_qubits (int): number of qubits to specify dimension of binary vector representation
        wire_map (dict): dictionary containing all wire labels used in the Pauli word as keys, and
            unique integer labels as their values
        check_is_pauli_word (bool): If True (default) then a check is run to verify that pauli_word
            is infact a Pauli word

    Returns:
        array: the ``2*n_qubits`` dimensional binary vector representation of the input Pauli word

    Raises:
        TypeError: if the input ``pauli_word`` is not an instance of Identity, PauliX, PauliY,
            PauliZ or tensor products thereof
        ValueError: if ``n_qubits`` is less than the number of wires acted on by the Pauli word

    **Example**

    If ``n_qubits`` and ``wire_map`` are both unspecified, the dimensionality of the binary vector
    will be ``2 * len(pauli_word.wires)``. Regardless of wire labels, the vector components encoding
    Pauli operations will be read from left-to-right in the tensor product when ``wire_map`` is
    unspecified, e.g.,

    >>> pauli_to_binary(qml.X('a') @ qml.Y('b') @ qml.Z('c'))
    array([1., 1., 0., 0., 1., 1.])
    >>> pauli_to_binary(qml.X('c') @ qml.Y('a') @ qml.Z('b'))
    array([1., 1., 0., 0., 1., 1.])

    The above cases have the same binary representation since they are equivalent up to a
    relabelling of the wires. To keep binary vector component enumeration consistent with wire
    labelling across multiple Pauli words, or define any arbitrary enumeration, one can use
    keyword argument ``wire_map`` to set this enumeration.

    >>> wire_map = {'a': 0, 'b': 1, 'c': 2}
    >>> pauli_to_binary(qml.X('a') @ qml.Y('b') @ qml.Z('c'), wire_map=wire_map)
    array([1., 1., 0., 0., 1., 1.])
    >>> pauli_to_binary(qml.X('c') @ qml.Y('a') @ qml.Z('b'), wire_map=wire_map)
    array([1., 0., 1., 1., 1., 0.])

    Now the two Pauli words are distinct in the binary vector representation, as the vector
    components are consistently mapped from the wire labels, rather than enumerated
    left-to-right.

    If ``n_qubits`` is unspecified, the dimensionality of the vector representation will be inferred
    from the size of support of the Pauli word,

    >>> pauli_to_binary(qml.X(0) @ qml.X(1))
    array([1., 1., 0., 0.])
    >>> pauli_to_binary(qml.X(0) @ qml.X(5))
    array([1., 1., 0., 0.])

    Dimensionality higher than twice the support can be specified by ``n_qubits``,

    >>> pauli_to_binary(qml.X(0) @ qml.X(1), n_qubits=6)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> pauli_to_binary(qml.X(0) @ qml.X(5), n_qubits=6)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    For these Pauli words to have a consistent mapping to vector representation, we once again
    need to specify a ``wire_map``.

    >>> wire_map = {0:0, 1:1, 5:5}
    >>> pauli_to_binary(qml.X(0) @ qml.X(1), n_qubits=6, wire_map=wire_map)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> pauli_to_binary(qml.X(0) @ qml.X(5), n_qubits=6, wire_map=wire_map)
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    Note that if ``n_qubits`` is unspecified and ``wire_map`` is specified, the dimensionality of the
    vector representation will be inferred from the highest integer in ``wire_map.values()``.

    >>> wire_map = {0:0, 1:1, 5:5}
    >>> pauli_to_binary(qml.X(0) @ qml.X(5),  wire_map=wire_map)
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    """
    wire_map = wire_map or {w: i for i, w in enumerate(pauli_word.wires)}

    if check_is_pauli_word and not is_pauli_word(pauli_word):
        raise TypeError(f"Expected a Pauli word Observable instance, instead got {pauli_word}.")

    pw = next(iter(pauli_word.pauli_rep))

    n_qubits_min = max(wire_map.values()) + 1
    if n_qubits is None:
        n_qubits = n_qubits_min
    elif n_qubits < n_qubits_min:
        raise ValueError(
            f"n_qubits must support the highest mapped wire index {n_qubits_min},"
            f" instead got n_qubits={n_qubits}."
        )

    binary_pauli = np.zeros(2 * n_qubits)

    for wire, pauli_type in pw.items():
        if pauli_type == "X":
            binary_pauli[wire_map[wire]] = 1
        elif pauli_type == "Y":
            binary_pauli[wire_map[wire]] = 1
            binary_pauli[n_qubits + wire_map[wire]] = 1
        elif pauli_type == "Z":
            binary_pauli[n_qubits + wire_map[wire]] = 1
    return binary_pauli


def binary_to_pauli(binary_vector, wire_map=None):  # pylint: disable=too-many-branches
    """Converts a binary vector of even dimension to an Observable instance.

    This functions follows the convention that the first half of binary vector components specify
    PauliX placements while the last half specify PauliZ placements.

    Args:
        binary_vector (Union[list, tuple, array]): binary vector of even dimension representing a
            unique Pauli word
        wire_map (dict): dictionary containing all wire labels used in the Pauli word as keys, and
            unique integer labels as their values

    Returns:
        Union[Tensor, Prod]: The Pauli word corresponding to the input binary vector.
        Note that if a zero vector is input, then the resulting Pauli word will be
        an :class:`~.Identity` instance. If new operator arithmetic is enabled via
        :func:`~.pennylane.operation.enable_new_opmath`, a :class:`~.Prod` will be
        returned, else a :class:`~.Tensor` will be returned.

    Raises:
        TypeError: if length of binary vector is not even, or if vector does not have strictly
            binary components

    **Example**

    If ``wire_map`` is unspecified, the Pauli operations follow the same enumerations as the vector
    components, i.e., the ``i`` and ``N+i`` components specify the Pauli operation on wire ``i``,

    >>> binary_to_pauli([0,1,1,0,1,0])
    Tensor(Y(1), X(2))

    An arbitrary labelling can be assigned by using ``wire_map``:

    >>> wire_map = {'a': 0, 'b': 1, 'c': 2}
    >>> binary_to_pauli([0,1,1,0,1,0], wire_map=wire_map)
    Tensor(Y('b'), X('c'))

    Note that the values of ``wire_map``, if specified, must be ``0,1,..., N``,
    where ``N`` is the dimension of the vector divided by two, i.e.,
    ``list(wire_map.values())`` must be ``list(range(len(binary_vector)/2))``.
    """

    if isinstance(binary_vector, (list, tuple)):
        binary_vector = np.asarray(binary_vector)

    if len(binary_vector) % 2 != 0:
        raise ValueError(
            f"Length of binary_vector must be even, instead got vector of shape {np.shape(binary_vector)}."
        )

    if not np.array_equal(binary_vector, binary_vector.astype(bool)):
        raise ValueError(
            f"Input vector must have strictly binary components, instead got {binary_vector}."
        )

    n_qubits = len(binary_vector) // 2

    if wire_map is None:
        label_map = {i: i for i in range(n_qubits)}
    else:
        if set(wire_map.values()) != set(range(n_qubits)):
            raise ValueError(
                f"The values of wire_map must be integers 0 to N, for 2N-dimensional binary vector."
                f" Instead got wire_map values: {wire_map.values()}"
            )
        label_map = {explicit_index: wire_label for wire_label, explicit_index in wire_map.items()}

    pauli_word = None
    for i in range(n_qubits):
        operation = None
        if binary_vector[i] == 1 and binary_vector[n_qubits + i] == 0:
            operation = PauliX(wires=Wires([label_map[i]]))

        elif binary_vector[i] == 1 and binary_vector[n_qubits + i] == 1:
            operation = PauliY(wires=Wires([label_map[i]]))

        elif binary_vector[i] == 0 and binary_vector[n_qubits + i] == 1:
            operation = PauliZ(wires=Wires([label_map[i]]))

        if operation is not None:
            if pauli_word is None:
                pauli_word = operation
            else:
                pauli_word @= operation

    if pauli_word is None:
        return Identity(wires=list(label_map.values())[0])

    return pauli_word


def pauli_word_to_string(pauli_word, wire_map=None):
    """Convert a Pauli word to a string.

    A Pauli word can be either:

    * A single pauli operator (see :class:`~.PauliX` for an example).

    * A :class:`.Tensor` instance containing Pauli operators.

    * A :class:`.Prod` instance containing Pauli operators.

    * A :class:`.SProd` instance containing a Pauli operator.

    * A :class:`.Hamiltonian` instance with only one term.

    Given a Pauli in observable form, convert it into string of
    characters from ``['I', 'X', 'Y', 'Z']``. This representation is required for
    functions such as :class:`.PauliRot`.

    .. warning::

        This method ignores any potential coefficient multiplying the Pauli word:

        >>> qml.pauli.pauli_word_to_string(3 * qml.X(0) @ qml.Y(1))
        'XY'

    .. warning::

        This method assumes all Pauli operators are acting on different wires, ignoring
        any extra operators:

        >>> qml.pauli.pauli_word_to_string(qml.X(0) @ qml.Y(0) @ qml.Y(0))
        'X'

    Args:
        pauli_word (Observable): an observable, either a :class:`~.Tensor` instance or
            single-qubit observable representing a Pauli group element.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        str: The string representation of the observable in terms of ``'I'``, ``'X'``, ``'Y'``,
        and/or ``'Z'``.

    Raises:
        TypeError: if the input observable is not a proper Pauli word.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> pauli_word = qml.X('a') @ qml.Y('c')
    >>> pauli_word_to_string(pauli_word, wire_map=wire_map)
    'XIY'
    """

    if not is_pauli_word(pauli_word):
        raise TypeError(f"Expected Pauli word observables, instead got {pauli_word}")
    if isinstance(pauli_word, qml.ops.Hamiltonian):
        # hamiltonian contains only one term
        return _pauli_word_to_string_legacy(pauli_word, wire_map)

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


def _pauli_word_to_string_legacy(pauli_word, wire_map):
    """Turn a legacy Hamiltonian operator to strings"""
    # TODO: Give Hamiltonian a pauli rep to make this branch obsolete
    pauli_word = pauli_word.ops[0]

    # If there is no wire map, we must infer from the structure of Paulis
    if wire_map is None:
        wire_map = {pauli_word.wires.labels[i]: i for i in range(len(pauli_word.wires))}

    character_map = {"Identity": "I", "PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}

    n_qubits = len(wire_map)

    # Set default value of all characters to identity
    pauli_string = ["I"] * n_qubits

    # Special case is when there is a single Pauli term
    if not isinstance(pauli_word.name, list):
        if pauli_word.name != "Identity":
            wire_idx = wire_map[pauli_word.wires[0]]
            pauli_string[wire_idx] = character_map[pauli_word.name]
        return "".join(pauli_string)

    for name, wire_label in zip(pauli_word.name, pauli_word.wires):
        wire_idx = wire_map[wire_label]
        pauli_string[wire_idx] = character_map[name]

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
        .Observable: The Pauli word representing of ``pauli_string`` on the wires
        enumerated in the wire map.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> string_to_pauli_word('XIY', wire_map=wire_map)
    X('a') @ Y('c')
    """
    character_map = {"I": Identity, "X": PauliX, "Y": PauliY, "Z": PauliZ}

    if not isinstance(pauli_string, str):
        raise TypeError(f"Input to string_to_pauli_word must be string, obtained {pauli_string}")

    # String can only consist of I, X, Y, Z
    if any(char not in character_map for char in pauli_string):
        raise ValueError(
            "Invalid characters encountered in string_to_pauli_word "
            f"string {pauli_string}. Permitted characters are 'I', 'X', 'Y', and 'Z'"
        )

    # If no wire map is provided, construct one using integers based on the length of the string
    if wire_map is None:
        wire_map = {x: x for x in range(len(pauli_string))}

    if len(pauli_string) != len(wire_map):
        raise ValueError(
            "Wire map and pauli_string must have the same length to convert "
            "from string to Pauli word."
        )

    # Special case: all-identity Pauli
    if pauli_string == "I" * len(wire_map):
        first_wire = list(wire_map)[0]
        return Identity(first_wire)

    pauli_word = None

    for wire_name, wire_idx in wire_map.items():
        pauli_char = pauli_string[wire_idx]

        # Don't care about the identity
        if pauli_char == "I":
            continue

        if pauli_word is not None:
            pauli_word = pauli_word @ character_map[pauli_char](wire_name)
        else:
            pauli_word = character_map[pauli_char](wire_name)

    return pauli_word


def pauli_word_to_matrix(pauli_word, wire_map=None):
    """Convert a Pauli word from a tensor to its matrix representation.

    The matrix representation of a Pauli word has dimension :math:`2^n \\times 2^n`,
    where :math:`n` is the number of qubits provided in ``wire_map``. For wires
    that the Pauli word does not act on, identities must be inserted into the tensor
    product at the correct positions.

    Args:
        pauli_word (Observable): an observable, either a :class:`~.Tensor`, :class:`~.Prod` or
            single-qubit observable representing a Pauli group element.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        array[complex]: The matrix representation of the multi-qubit Pauli over the
        specified wire map.

    Raises:
        TypeError: if the input observable is not a proper Pauli word.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1}
    >>> pauli_word = qml.X('a') @ qml.Y('b')
    >>> pauli_word_to_matrix(pauli_word, wire_map=wire_map)
    array([[0.+0.j, 0.-0.j, 0.+0.j, 0.-1.j],
           [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j],
           [0.+0.j, 0.-1.j, 0.+0.j, 0.-0.j],
           [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    """
    if not is_pauli_word(pauli_word):
        raise TypeError(f"Expected Pauli word observables, instead got {pauli_word}")

    # If there is no wire map, we must infer from the structure of Paulis
    if wire_map is None:
        wire_map = {pauli_word.wires.labels[i]: i for i in range(len(pauli_word.wires))}

    return pauli_word.matrix(wire_map)


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
            f"Vectors a and b must be the same dimension, instead got "
            f"shapes {np.shape(pauli_vec_1)} and {np.shape(pauli_vec_2)}."
        )

    if len(pauli_vec_1) % 2 != 0:
        raise ValueError(
            f"Symplectic vector-space must have even dimension, instead got vectors of shape {np.shape(pauli_vec_1)}."
        )

    if not (
        np.array_equal(pauli_vec_1, pauli_vec_1.astype(bool))
        and np.array_equal(pauli_vec_2, pauli_vec_2.astype(bool))
    ):
        raise ValueError(
            f"Vectors a and b must have strictly binary components, instead got {pauli_vec_1} and {pauli_vec_2}"
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


def _are_pauli_words_qwc_pauli_rep(lst_pauli_words):
    """Given a list of observables assumed to be valid Pauli words, determine if they are pairwise
    qubit-wise commuting. This private method is used for operators that have a valid pauli
    representation"""
    basis = {}
    for op in lst_pauli_words:  # iterate over the list of observables
        if len(pr := op.pauli_rep) > 1:
            return False

        pw = next(iter(pr))

        for wire, pauli_type in pw.items():  # iterate over wires of the observable,
            if pauli_type != "I":
                if wire in basis and pauli_type != basis[wire]:
                    # Only non-identity paulis are in basis, so if pauli_type doesn't match
                    # it is guaranteed to not commute
                    return False

                basis[wire] = pauli_type

    return True  # if we get through all ops, then they are qwc!


def are_pauli_words_qwc(lst_pauli_words):
    """Given a list of observables assumed to be valid Pauli observables, determine if they are pairwise
    qubit-wise commuting.

    This implementation has time complexity ~ O(m * n) for m Pauli words and n wires, where n is the
    number of distinct wire labels used to represent the Pauli words.

    Args:
        lst_pauli_words (list[Observable]): List of observables (assumed to be valid Pauli words).

    Returns:
        (bool): True if they are all qubit-wise commuting, false otherwise. If any of the provided
        observables are not valid Pauli words, false is returned.
    """
    if all(op.pauli_rep is not None for op in lst_pauli_words):
        return _are_pauli_words_qwc_pauli_rep(lst_pauli_words)

    latest_op_name_per_wire = {}

    for op in lst_pauli_words:  # iterate over the list of observables
        op_names = [op.name] if not isinstance(op.name, list) else op.name
        op_wires = op.wires.tolist()

        for op_name, wire in zip(op_names, op_wires):  # iterate over wires of the observable,
            latest_op_name = latest_op_name_per_wire.get(wire, "Identity")
            if latest_op_name != op_name and (
                op_name != "Identity" and latest_op_name != "Identity"
            ):
                return False

            if op_name != "Identity":
                latest_op_name_per_wire[wire] = op_name

    return True  # if we get through all ops, then they are qwc!


def observables_to_binary_matrix(observables, n_qubits=None, wire_map=None):
    """Converts a list of Pauli words to the binary vector representation and yields a row matrix
    of the binary vectors.

    The dimension of the binary vectors will be implied from the highest wire being acted on
    non-trivially by the Pauli words in observables.

    Args:
        observables (list[Union[Identity, PauliX, PauliY, PauliZ, Tensor, Prod, SProd]]): the list
            of Pauli words
        n_qubits (int): number of qubits to specify dimension of binary vector representation
        wire_map (dict): dictionary containing all wire labels used in the Pauli words as keys, and
            unique integer labels as their values


    Returns:
        array[array[int]]: a matrix whose rows are Pauli words in binary vector representation

    **Example**

    >>> observables_to_binary_matrix([X(0) @ Y(2), Z(0) @ Z(1) @ Z(2)])
    array([[1., 1., 0., 0., 1., 0.],
           [0., 0., 0., 1., 1., 1.]])
    """

    m_cols = len(observables)

    if wire_map is None:
        all_wires = Wires.all_wires([pauli_word.wires for pauli_word in observables])
        wire_map = {i: c for c, i in enumerate(all_wires)}

    n_qubits_min = max(wire_map.values()) + 1
    if n_qubits is None:
        n_qubits = n_qubits_min
    elif n_qubits < n_qubits_min:
        raise ValueError(
            f"n_qubits must support the highest mapped wire index {n_qubits_min},"
            f" instead got n_qubits={n_qubits}."
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
        raise ValueError(f"Expected a binary array, instead got {binary_observables}")

    m_terms = np.shape(binary_observables)[0]
    adj = np.zeros((m_terms, m_terms))

    for i in range(m_terms):
        for j in range(i + 1, m_terms):
            adj[i, j] = int(not is_qwc(binary_observables[i], binary_observables[j]))
            adj[j, i] = adj[i, j]

    return adj


# from grouping/pauli.py ------------------------
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

    while element_idx < 4**n_qubits:
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

    >>> from pennylane.pauli import pauli_group
    >>> n_qubits = 3
    >>> for p in pauli_group(n_qubits):
    ...     print(p)
    ...
    I(0)
    Z(2)
    Z(1)
    Z(1) @ Z(2)
    Z(0)

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
    I('a')
    Z('c')
    Z('b')
    Z('b') @ Z('c')
    Z('a')

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

    >>> qml.pauli.partition_pauli_group(3)
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
    for string in product("FXYZ", repeat=n_qubits):
        if string not in strings:
            num_free_slots = string.count("F")

            group = []
            commuting = product("IZ", repeat=num_free_slots)

            for commuting_string in commuting:
                commuting_string = list(commuting_string)
                new_string = tuple(commuting_string.pop(0) if s == "F" else s for s in string)

                if new_string not in strings:  # only add if string has not already been grouped
                    group.append("".join(new_string))
                    strings |= {new_string}

            if len(group) > 0:
                groups.append(group)

    return groups


# from grouping/transformations.py --------------
def qwc_rotation(pauli_operators):
    """Performs circuit implementation of diagonalizing unitary for a Pauli word.

    Args:
        pauli_operators (list[Union[PauliX, PauliY, PauliZ, Identity]]): Single-qubit Pauli
            operations. No Pauli operations in this list may be acting on the same wire.
    Raises:
        TypeError: if any elements of ``pauli_operators`` are not instances of
            :class:`~.PauliX`, :class:`~.PauliY`, :class:`~.PauliZ`, or :class:`~.Identity`

    **Example**

    >>> pauli_operators = [qml.X('a'), qml.Y('b'), qml.Z('c')]
    >>> qwc_rotation(pauli_operators)
    [RY(-1.5707963267948966, wires=['a']), RX(1.5707963267948966, wires=['b'])]
    """
    paulis_with_identity = (qml.Identity, qml.X, qml.Y, qml.Z)
    if not all(isinstance(element, paulis_with_identity) for element in pauli_operators):
        raise TypeError(
            f"All values of input pauli_operators must be either Identity, PauliX, PauliY, or PauliZ instances,"
            f" instead got pauli_operators = {pauli_operators}."
        )
    ops = []
    with qml.QueuingManager.stop_recording():
        for pauli in pauli_operators:
            if isinstance(pauli, qml.X):
                ops.append(qml.RY(-np.pi / 2, wires=pauli.wires))

            elif isinstance(pauli, qml.Y):
                ops.append(qml.RX(np.pi / 2, wires=pauli.wires))
    return ops


@qml.QueuingManager.stop_recording()
def diagonalize_pauli_word(pauli_word):
    """Transforms the Pauli word to diagonal form in the computational basis.

    Args:
        pauli_word (Operator): the Pauli word to diagonalize in computational basis

    Returns:
        Operator: the Pauli word diagonalized in the computational basis

    Raises:
        TypeError: if the input is not a Pauli word, i.e., a Pauli operator,
            :class:`~.Identity`, or :class:`~.Tensor` instances thereof

    **Example**

    >>> diagonalize_pauli_word(qml.X('a') @ qml.Y('b') @ qml.Z('c'))
    Z('a') @ Z('b') @ Z('c')
    """

    if not is_pauli_word(pauli_word):
        raise TypeError(f"Input must be a Pauli word, instead got: {pauli_word}.")

    pw = next(iter(pauli_word.pauli_rep))

    # ordered as pauli_word, with identities eliminated
    components = [qml.Z(w) for w in pauli_word.wires if w in pw]
    if not components:
        return qml.Identity(wires=pauli_word.wires)

    if isinstance(pauli_word, Tensor):
        return components[0] if len(components) == 1 else Tensor(*components)

    prod = qml.prod(*components)
    coeff = pauli_word.pauli_rep[pw]
    return prod if qml.math.allclose(coeff, 1) else coeff * prod


@qml.QueuingManager.stop_recording()
def diagonalize_qwc_pauli_words(
    qwc_grouping,
):  # pylint: disable=too-many-branches, isinstance-second-argument-not-valid-type
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

    >>> qwc_group = [qml.X(0) @ qml.Z(1),
                     qml.X(0) @ qml.Y(3),
                     qml.Z(1) @ qml.Y(3)]
    >>> diagonalize_qwc_pauli_words(qwc_group)
    ([RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[3])],
     [Z(0) @ Z(1),
      Z(0) @ Z(3),
      Z(1) @ Z(3)])
    """

    full_pauli_word = {}
    new_ops = []
    for term in qwc_grouping:
        pauli_rep = term.pauli_rep
        if pauli_rep is None or len(pauli_rep) > 1 or term.name == "Hamiltonian":
            raise ValueError("This function only supports pauli words.")
        pw = next(iter(pauli_rep))
        for wire, pauli_type in pw.items():
            if wire in full_pauli_word:
                if full_pauli_word[wire] != pauli_type:
                    raise ValueError("The list of Pauli words are not qubit-wise commuting")
            else:
                full_pauli_word[wire] = pauli_type

        new_ops.append(diagonalize_pauli_word(term))

    diag_gates = []
    for w, pauli_type in full_pauli_word.items():
        if pauli_type == "X":
            diag_gates.append(qml.RY(-np.pi / 2, wires=w))
        elif pauli_type == "Y":
            diag_gates.append(qml.RX(np.pi / 2, wires=w))
    return diag_gates, new_ops


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

    >>> qwc_group_1 = [qml.X(0) @ qml.Z(1),
                       qml.X(0) @ qml.Y(3),
                       qml.Z(1) @ qml.Y(3)]
    >>> qwc_group_2 = [qml.Y(0),
                       qml.Y(0) @ qml.X(2),
                       qml.X(1) @ qml.Z(3)]
    >>> post_rotations, diag_groupings = diagonalize_qwc_groupings([qwc_group_1, qwc_group_2])
    >>> post_rotations
    [[RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[3])],
     [RX(1.5707963267948966, wires=[0]),
      RY(-1.5707963267948966, wires=[2]),
      RY(-1.5707963267948966, wires=[1])]]
    >>> diag_groupings
    [[Z(0) @ Z(1),
     Z(0) @ Z(3),
     Z(1) @ Z(3)],
    [Z(0),
     Z(0) @ Z(2),
     Z(1) @ Z(3)]]
    """

    post_rotations = []
    diag_groupings = []
    m_groupings = len(qwc_groupings)

    for i in range(m_groupings):
        diagonalizing_unitary, diag_grouping = diagonalize_qwc_pauli_words(qwc_groupings[i])
        post_rotations.append(diagonalizing_unitary)
        diag_groupings.append(diag_grouping)

    return post_rotations, diag_groupings


# from observable_hf.py -------------------------
def simplify(h, cutoff=1.0e-12):
    r"""Add together identical terms in the Hamiltonian.

    The Hamiltonian terms with identical Pauli words are added together and eliminated if the
    overall coefficient is smaller than a cutoff value.

    Args:
        h (Hamiltonian): PennyLane Hamiltonian
        cutoff (float): cutoff value for discarding the negligible terms

    Returns:
        Hamiltonian: Simplified PennyLane Hamiltonian

    **Example**

    >>> c = np.array([0.5, 0.5])
    >>> h = qml.Hamiltonian(c, [qml.X(0) @ qml.Y(1), qml.X(0) @ qml.Y(1)])
    >>> print(simplify(h))
    (1.0) [X0 Y1]
    """
    wiremap = dict(zip(h.wires, range(len(h.wires) + 1)))

    c, o = [], []
    for i, op in enumerate(h.ops):
        op = qml.operation.Tensor(op).prune()
        op = qml.pauli.pauli_word_to_string(op, wire_map=wiremap)
        if op not in o:
            c.append(h.coeffs[i])
            o.append(op)
        else:
            c[o.index(op)] += h.coeffs[i]

    coeffs, ops = [], []
    c = qml.math.convert_like(c, c[0])
    nonzero_ind = qml.math.argwhere(abs(c) > cutoff).flatten()
    for i in nonzero_ind:
        coeffs.append(c[i])
        ops.append(qml.pauli.string_to_pauli_word(o[i], wire_map=wiremap))

    try:
        coeffs = qml.math.stack(coeffs)
    except ValueError:
        pass

    return qml.Hamiltonian(qml.math.array(coeffs), ops)


pauli_mult_dict = {
    "XX": "I",
    "YY": "I",
    "ZZ": "I",
    "ZX": "Y",
    "XZ": "Y",
    "ZY": "X",
    "YZ": "X",
    "XY": "Z",
    "YX": "Z",
    "IX": "X",
    "IY": "Y",
    "IZ": "Z",
    "XI": "X",
    "YI": "Y",
    "ZI": "Z",
    "I": "I",
    "II": "I",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
}

pauli_coeff = {
    "ZX": 1.0j,
    "XZ": -1.0j,
    "ZY": -1.0j,
    "YZ": 1.0j,
    "XY": 1.0j,
    "YX": -1.0j,
}


def _binary_matrix_from_pws(terms, num_qubits, wire_map=None):
    r"""Get a binary matrix representation from a list of PauliWords where each row corresponds to a
    Pauli term, which is represented by a concatenation of Z and X vectors.

    Args:
        terms (Iterable[~.PauliWord]): operators defining the Hamiltonian
        num_qubits (int): number of wires required to define the Hamiltonian
        wire_map (dict): dictionary containing all wire labels used in the Pauli words as keys, and
            unique integer labels as their values

    Returns:
        array[int]: binary matrix representation of the Hamiltonian of shape
        :math:`len(terms) * 2*num_qubits`

    **Example**

    >>> from pennylane.pauli import PauliWord
    >>> wire_map = {'a':0, 'b':1, 'c':2, 'd':3}
    >>> terms = [PauliWord({'a': 'Z', 'b': 'X'}),
    ...          PauliWord({'a': 'Z', 'c': 'Y'}),
    ...          PauliWord({'a': 'X', 'd': 'Y'})]
    >>> _binary_matrix_from_pws(terms, 4, wire_map=wire_map)
    array([[1, 0, 0, 0, 0, 1, 0, 0],
           [1, 0, 1, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 1, 0, 0, 1]])
    """
    if wire_map is None:
        all_wires = qml.wires.Wires.all_wires([term.wires for term in terms], sort=True)
        wire_map = {i: c for c, i in enumerate(all_wires)}

    binary_matrix = np.zeros((len(terms), 2 * num_qubits), dtype=int)
    for idx, pw in enumerate(terms):
        for wire, pauli_op in pw.items():
            if pauli_op in ["X", "Y"]:
                binary_matrix[idx][wire_map[wire] + num_qubits] = 1
            if pauli_op in ["Z", "Y"]:
                binary_matrix[idx][wire_map[wire]] = 1

    return binary_matrix
