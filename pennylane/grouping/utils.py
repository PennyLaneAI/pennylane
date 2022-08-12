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
Utility functions used in Pauli partitioning and measurement reduction schemes utilizing the
symplectic vector-space representation of Pauli words. For information on the symplectic binary
representation of Pauli words and applications, see:

* `arXiv:quant-ph/9705052 <https://arxiv.org/abs/quant-ph/9705052>`_
* `arXiv:1701.08213 <https://arxiv.org/abs/1701.08213>`_
* `arXiv:1907.09386 <https://arxiv.org/abs/1907.09386>`_
"""
from functools import reduce

import numpy as np

import pennylane as qml
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane.operation import Observable, Tensor
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
    wire_map = {label: i for i, label in enumerate(wire_labels)}
    return wire_map


def is_pauli_word(observable):
    """
    Checks if an observable instance is a Pauli word.

    Args:
        observable (Observable): an observable, either a :class:`~.Tensor` instance or
            single-qubit observable.

    Returns:
        bool: true if the input observable is a Pauli word, false otherwise.

    Raises:
        TypeError: if input observable is not an Observable instance.

    **Example**

    >>> is_pauli_word(qml.Identity(0))
    True
    >>> is_pauli_word(qml.PauliX(0) @ qml.PauliZ(2))
    True
    >>> is_pauli_word(qml.PauliZ(0) @ qml.Hadamard(1))
    False
    """

    if not isinstance(observable, Observable):
        raise TypeError(f"Expected {Observable} instance, instead got {type(observable)} instance.")

    pauli_word_names = ["Identity", "PauliX", "PauliY", "PauliZ"]
    if isinstance(observable, Tensor):
        return set(observable.name).issubset(pauli_word_names)

    return observable.name in pauli_word_names


def are_identical_pauli_words(pauli_1, pauli_2):
    """Performs a check if two Pauli words have the same ``wires`` and ``name`` attributes.

    This is a convenience function that checks if two given :class:`~.Tensor` instances specify the same
    Pauli word. This function only checks if both :class:`~.Tensor` instances have the same wires and name
    attributes, and hence won't perform any simplification to identify if the two Pauli words are
    algebraically equivalent. For instance, this function will not identify
    that ``PauliX(0) @ PauliX(0) = Identity(0)``, or ``PauliX(0) @ Identity(1)
    = PauliX(0)``, or ``Identity(0) = Identity(1)``, etc.

    Args:
        pauli_1 (Union[Identity, PauliX, PauliY, PauliZ, Tensor]): the first Pauli word
        pauli_2 (Union[Identity, PauliX, PauliY, PauliZ, Tensor]): the second Pauli word

    Returns:
        bool: whether ``pauli_1`` and ``pauli_2`` have the same wires and name attributes

    Raises:
        TypeError: if ``pauli_1`` or ``pauli_2`` are not :class:`~.Identity`,
            :class:`~.PauliX`, :class:`~.PauliY`, :class:`~.PauliZ`, or
            :class:`~.Tensor` instances

    **Example**

    >>> are_identical_pauli_words(qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1))
    True
    >>> are_identical_pauli_words(qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliX(3))
    False
    """

    if not (is_pauli_word(pauli_1) and is_pauli_word(pauli_2)):
        raise TypeError(f"Expected Pauli word observables, instead got {pauli_1} and {pauli_2}.")

    paulis_with_identity = (PauliX, PauliY, PauliZ, Identity)

    # convert tensors of length 1 to plain observables
    pauli_1 = getattr(pauli_1, "prune", lambda: pauli_1)()
    pauli_2 = getattr(pauli_2, "prune", lambda: pauli_2)()

    if isinstance(pauli_1, paulis_with_identity) and isinstance(pauli_2, paulis_with_identity):
        return (pauli_1.wires, pauli_1.name) == (pauli_2.wires, pauli_2.name)

    if isinstance(pauli_1, paulis_with_identity) and isinstance(pauli_2, Tensor):
        return {(pauli_1.wires, pauli_1.name)} == set(zip(pauli_2.wires, pauli_2.name))

    if isinstance(pauli_1, Tensor) and isinstance(pauli_2, paulis_with_identity):
        return set(zip(pauli_1.wires, pauli_1.name)) == {(pauli_2.wires, pauli_2.name)}

    return set(zip(pauli_1.wires, pauli_1.name)) == set(zip(pauli_2.wires, pauli_2.name))


def pauli_to_binary(pauli_word, n_qubits=None, wire_map=None, check_is_pauli_word=True):
    """Converts a Pauli word to the binary vector representation.

    This functions follows convention that the first half of binary vector components specify
    PauliX placements while the last half specify PauliZ placements.

    Args:
        pauli_word (Union[Identity, PauliX, PauliY, PauliZ, Tensor]): the Pauli word to be
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

    >>> pauli_to_binary(qml.PauliX('a') @ qml.PauliY('b') @ qml.PauliZ('c'))
    array([1., 1., 0., 0., 1., 1.])
    >>> pauli_to_binary(qml.PauliX('c') @ qml.PauliY('a') @ qml.PauliZ('b'))
    array([1., 1., 0., 0., 1., 1.])

    The above cases have the same binary representation since they are equivalent up to a
    relabelling of the wires. To keep binary vector component enumeration consistent with wire
    labelling across multiple Pauli words, or define any arbitrary enumeration, one can use
    keyword argument ``wire_map`` to set this enumeration.

    >>> wire_map = {'a': 0, 'b': 1, 'c': 2}
    >>> pauli_to_binary(qml.PauliX('a') @ qml.PauliY('b') @ qml.PauliZ('c'), wire_map=wire_map)
    array([1., 1., 0., 0., 1., 1.])
    >>> pauli_to_binary(qml.PauliX('c') @ qml.PauliY('a') @ qml.PauliZ('b'), wire_map=wire_map)
    array([1., 0., 1., 1., 1., 0.])

    Now the two Pauli words are distinct in the binary vector representation, as the vector
    components are consistently mapped from the wire labels, rather than enumerated
    left-to-right.

    If ``n_qubits`` is unspecified, the dimensionality of the vector representation will be inferred
    from the size of support of the Pauli word,

    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(1))
    array([1., 1., 0., 0.])
    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(5))
    array([1., 1., 0., 0.])

    Dimensionality higher than twice the support can be specified by ``n_qubits``,

    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(1), n_qubits=6)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(5), n_qubits=6)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    For these Pauli words to have a consistent mapping to vector representation, we once again
    need to specify a ``wire_map``.

    >>> wire_map = {0:0, 1:1, 5:5}
    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(1), n_qubits=6, wire_map=wire_map)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(5), n_qubits=6, wire_map=wire_map)
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    Note that if ``n_qubits`` is unspecified and ``wire_map`` is specified, the dimensionality of the
    vector representation will be inferred from the highest integer in ``wire_map.values()``.

    >>> wire_map = {0:0, 1:1, 5:5}
    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(5),  wire_map=wire_map)
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    """

    if check_is_pauli_word and not is_pauli_word(pauli_word):
        raise TypeError(f"Expected a Pauli word Observable instance, instead got {pauli_word}.")

    if wire_map is None:
        num_wires = len(pauli_word.wires)
        wire_map = {pauli_word.wires[i]: i for i in range(num_wires)}

    n_qubits_min = max(wire_map.values()) + 1
    if n_qubits is None:
        n_qubits = n_qubits_min
    elif n_qubits < n_qubits_min:
        raise ValueError(
            f"n_qubits must support the highest mapped wire index {n_qubits_min},"
            f" instead got n_qubits={n_qubits}."
        )

    pauli_wires = pauli_word.wires.map(wire_map).tolist()

    binary_pauli = np.zeros(2 * n_qubits)

    paulis_with_identity = (PauliX, PauliY, PauliZ, Identity)

    if isinstance(pauli_word, paulis_with_identity):
        operations_zip = zip(pauli_wires, [pauli_word.name])
    else:
        operations_zip = zip(pauli_wires, pauli_word.name)

    for wire, name in operations_zip:

        if name == "PauliX":
            binary_pauli[wire] = 1

        elif name == "PauliY":
            binary_pauli[wire] = 1
            binary_pauli[n_qubits + wire] = 1

        elif name == "PauliZ":
            binary_pauli[n_qubits + wire] = 1

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
        Tensor: The Pauli word corresponding to the input binary vector. Note
        that if a zero vector is input, then the resulting Pauli word will be
        an :class:`~.Identity` instance.

    Raises:
        TypeError: if length of binary vector is not even, or if vector does not have strictly
            binary components

    **Example**

    If ``wire_map`` is unspecified, the Pauli operations follow the same enumerations as the vector
    components, i.e., the ``i`` and ``N+i`` components specify the Pauli operation on wire ``i``,

    >>> binary_to_pauli([0,1,1,0,1,0])
    Tensor(PauliY(wires=[1]), PauliX(wires=[2]))

    An arbitrary labelling can be assigned by using ``wire_map``:

    >>> wire_map = {'a': 0, 'b': 1, 'c': 2}
    >>> binary_to_pauli([0,1,1,0,1,0], wire_map=wire_map)
    Tensor(PauliY(wires=['b']), PauliX(wires=['c']))

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
    """Convert a Pauli word from a tensor to a string.

    Given a Pauli in observable form, convert it into string of
    characters from ``['I', 'X', 'Y', 'Z']``. This representation is required for
    functions such as :class:`.PauliRot`.

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
    >>> pauli_word = qml.PauliX('a') @ qml.PauliY('c')
    >>> pauli_word_to_string(pauli_word, wire_map=wire_map)
    'XIY'
    """

    if not is_pauli_word(pauli_word):
        raise TypeError(f"Expected Pauli word observables, instead got {pauli_word}")

    character_map = {"Identity": "I", "PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}

    # If there is no wire map, we must infer from the structure of Paulis
    if wire_map is None:
        wire_map = {pauli_word.wires.labels[i]: i for i in range(len(pauli_word.wires))}

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
    PauliX(wires=['a']) @ PauliY(wires=['c'])
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
        pauli_word (Observable): an observable, either a :class:`~.Tensor` instance or
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
    >>> pauli_word = qml.PauliX('a') @ qml.PauliY('b')
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

    n_qubits = len(wire_map)

    # If there is only a single qubit, we can return the matrix directly
    if n_qubits == 1:
        return pauli_word.matrix()

    # There may be more than one qubit in the Pauli but still only
    # one of them with anything acting on it, so take that into account
    pauli_names = [pauli_word.name] if isinstance(pauli_word.name, str) else pauli_word.name

    # Special case: the identity Pauli
    if pauli_names == ["Identity"]:
        return np.eye(2**n_qubits)

    # If there is more than one qubit, we must go through the wire map wire
    # by wire and pick out the relevant matrices
    pauli_mats = [ID_MAT for x in range(n_qubits)]

    for wire_label, wire_idx in wire_map.items():
        if wire_label in pauli_word.wires.labels:
            op_idx = pauli_word.wires.labels.index(wire_label)
            # compute_matrix() only works because we work with Paulis here
            pauli_mats[wire_idx] = getattr(qml, pauli_names[op_idx]).compute_matrix()

    return reduce(np.kron, pauli_mats)


def is_commuting(pauli_word_1, pauli_word_2, wire_map=None):
    r"""Checks if two Pauli words commute.

    To determine if two Pauli words commute, we can check the value of the
    symplectic inner product of their binary vector representations.
    For two binary vectors representing Pauli words, :math:`p_1 = [x_1, z_1]`
    and :math:`p_2 = [x_2, z_2],` the symplectic inner product is defined as
    :math:`\langle p_1, p_2 \rangle_{symp} = z_1 x_2^T + z_2 x_1^T`. If the symplectic
    product is :math:`0` they commute, while if it is :math:`1`, they don't commute.

    Args:
        pauli_word_1 (Observable): first Pauli word in commutator
        pauli_word_2 (Observable): second Pauli word in commutator
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        bool: returns True if the input Pauli commute, False otherwise

    Raises:
        TypeError: if either of the Pauli words is not valid.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
    >>> pauli_word_1 = qml.PauliX('a') @ qml.PauliY('b')
    >>> pauli_word_2 = qml.PauliZ('a') @ qml.PauliZ('c')
    >>> is_commuting(pauli_word_1, pauli_word_2, wire_map=wire_map)
    False
    """

    if not (is_pauli_word(pauli_word_1) and is_pauli_word(pauli_word_2)):
        raise TypeError(
            f"Expected Pauli word observables, instead got {pauli_word_1} and {pauli_word_2}"
        )

    if wire_map is None:
        wire_map = _wire_map_from_pauli_pair(pauli_word_1, pauli_word_2)

    n_qubits = len(wire_map)

    pauli_vec_1 = pauli_to_binary(pauli_word_1, n_qubits=n_qubits, wire_map=wire_map)
    pauli_vec_2 = pauli_to_binary(pauli_word_2, n_qubits=n_qubits, wire_map=wire_map)

    x1, z1 = pauli_vec_1[:n_qubits], pauli_vec_1[n_qubits:]
    x2, z2 = pauli_vec_2[:n_qubits], pauli_vec_2[n_qubits:]

    return (np.dot(z1, x2) + np.dot(z2, x1)) % 2 == 0


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


def are_pauli_words_qwc(lst_pauli_words):
    """Given a list of observables assumed to be valid Pauli words, determine if they are pairwise
    qubit-wise commuting.

    This implementation has time complexity ~ O(m * n) for m Pauli words and n wires, where n is the
    number of distinct wire labels used to represent the Pauli words.

    Args:
        lst_pauli_words (list[Observable]): List of observables (assumed to be valid Pauli words).

    Returns:
        (bool): True if they are all qubit-wise commuting, false otherwise.
    """
    latest_op_name_per_wire = {}

    for op in lst_pauli_words:  # iterate over the list of observables
        op_names = [op.name] if not isinstance(op.name, list) else op.name
        op_wires = op.wires.tolist()

        for op_name, wire in zip(op_names, op_wires):  # iterate over wires of the observable,
            try:
                if latest_op_name_per_wire[wire] != op_name and (
                    op_name != "Identity" and latest_op_name_per_wire[wire] != "Identity"
                ):
                    return False
                if latest_op_name_per_wire[wire] == "Identity":
                    latest_op_name_per_wire[wire] = op_name  # update name
            except KeyError:
                latest_op_name_per_wire[wire] = op_name  # add wire and name for the first time

    return True  # if we get through all ops, then they are qwc!


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
