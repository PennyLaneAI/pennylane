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
symplectic vector space representation of Pauli words. For information on the symplectic binary
representation of Pauli words and applications, see:
arXiv:quant-ph/9705052
arXiv:1701.08213
arXiv:1907.09386
"""

from pennylane import PauliX, PauliY, PauliZ, Identity
from pennylane.operation import Observable, Tensor
from pennylane.wires import Wires
import numpy as np


def get_n_qubits(observables):
    """Obtains the number of qubits needed to support the given list of observables.

    **Usage example:**

    >>> observables = [qml.PauliX(0) @ qml.PauliY(13),
                       qml.PauliZ(4) @ qml.PauliX(5)@ qml.PauliX(7)]
    >>> get_n_qubits(observables)
    14

    Args:
        observables (list[Observable]): list of observables with wires attribute.

    Returns:
        int: number of qubits.

    Raises:
        TypeError: if any item in observables list does not have the wires attribute.

    """

    for c, obs in enumerate(observables):
        if not hasattr(obs, "wires"):
            raise TypeError("Instance of {} does not have wires attribute.".format(type(obs)))
        if isinstance(obs, Tensor):
            observables[c] = obs.prune()

    n_qubits = len(
        Wires.all_wires([obs.wires for obs in observables if not isinstance(obs, Identity)])
    )

    return n_qubits


def is_pauli_word(observable):
    """
    Checks if an observable instance is a Pauli word.

    **Example usage:**

    >>> is_pauli_word(qml.Identity(0))
    True
    >>> is_pauli_word(qml.PauliX(0) @ qml.PauliZ(2))
    True
    >>> is_pauli_word(qml.PauliZ(0) @ qml.Hadamard(1))
    False

    Args:
        observable (Observable): an observable, either a Tensor instance or single-qubit
            observable.

    Returns:
        bool: true if the input observable is a Pauli word, false otherwise.

    Raises:
        TypeError: if input observable is not an Observable instance.
    """

    if not isinstance(observable, Observable):
        raise TypeError(
            "Expected {} instance, instead got {} instance.".format(Observable, type(observable))
        )

    if not isinstance(observable, Tensor):
        observable = Tensor(observable)

    if not set(observable.name).issubset(["Identity", "PauliX", "PauliY", "PauliZ"]):
        return False

    return True


def are_identical_pauli_words(pauli_1, pauli_2):
    """Performs a check if two Pauli words have the same `wires` and `name` attributes.

    This is a convenience function that checks if two given Tensor instances specify the same
    Pauli. This function only checks if both Tensor instances have the same wires and name
    attributes, and hence won't perform any simplification to identify if the two Pauli words are
    algebraically equivalent. For instance, this function will not identify that
    PauliX(0) @ PauliX(0) = Identity(0), or PauliX(0) @ Identity(1) = PauliX(0), or
    Identity(0) = Identity(1), etc.

    **Usage example:**

    >>> are_identical_pauli_words(qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1))
    True
    >>> are_identical_pauli_words(qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliX(3))
    False

    Args:
        pauli_1 (Union[Identity, PauliX, PauliY, PauliZ, Tensor]): the first Pauli word.
        pauli_2 (Union[Identity, PauliX, PauliY, PauliZ, Tensor]): the second Pauli word.

    Returns:
        Boolean: whether pauli_1 and pauli_2 have the same wires and name attributes.

    Raises:
        TypeError: if pauli_1 or pauli_2 are not Identity, PauliX, PauliY, PauliZ, or Tensor
            instances.

    """

    if not (is_pauli_word(pauli_1) and is_pauli_word(pauli_2)):
        raise TypeError(
            "Expected Pauli word observables, instead got {} and {}.".format(pauli_1, pauli_2)
        )

    if isinstance(pauli_1, (PauliX, PauliY, PauliZ, Identity)):
        pauli_1 = Tensor(pauli_1)

    if isinstance(pauli_2, (PauliX, PauliY, PauliZ, Identity)):
        pauli_2 = Tensor(pauli_2)

    return set(zip(pauli_1.wires, pauli_1.name)) == set(zip(pauli_2.wires, pauli_2.name))


def binary_symplectic_inner_prod(pauli_vec_1, pauli_vec_2):
    """Computes the symplectic inner product of two vectors a and b over the binary field.

    **Usage example:**

    >>> binary_symplectic_inner_prod([0,1,0,1,1,0],[1,0,0,1,0,0])
    1
    >>> binary_symplectic_inner_prod([1,1,0,1,1,0],[1,1,0,1,1,0])
    0

    Args:
        pauli_vec_1 (Union[list, tuple, array]): first binary vector argument in symplectic inner
            product.
        pauli_vec_2 (Union[list, tuple, array]): second binary vector argument in symplectic inner
            product.

    Returns:
        bool: binary symplectic inner product between `pauli_vec_1` and `pauli_vec_2`.

    Raises:
        ValueError: if the input vectors are of different dimension, if the vectors are not of even
        dimension, or if the vector components are not strictly binary.

    """

    if isinstance(pauli_vec_1, (list, tuple)):
        pauli_vec_1 = np.asarray(pauli_vec_1)
    if isinstance(pauli_vec_2, (list, tuple)):
        pauli_vec_2 = np.asarray(pauli_vec_2)

    if len(pauli_vec_1) != len(pauli_vec_2):
        raise ValueError(
            "Both input vectors must be the same dimension, instead got shapes {} and {}.".format(
                np.shape(pauli_vec_1), np.shape(pauli_vec_2)
            )
        )

    if len(pauli_vec_1) % 2 != 0:
        raise ValueError(
            "Symplectic vector space must have even dimension, instead got vectors of shape {}."
            .format(np.shape(pauli_vec_1))
        )

    if not (
        np.array_equal(pauli_vec_1, pauli_vec_1.astype(bool))
        and np.array_equal(pauli_vec_2, pauli_vec_2.astype(bool))
    ):
        raise ValueError(
            "Both vectors must have strictly binary components, instead got {} and {}".format(
                pauli_vec_1, pauli_vec_2
            )
        )

    dim = len(pauli_vec_1) // 2

    inner_prod = pauli_vec_1[:dim] @ pauli_vec_2[dim:] + pauli_vec_2[:dim] @ pauli_vec_1[dim:]

    return inner_prod % 2


def pauli_to_binary(pauli_word, n_qubits=None, wire_map=None):
    """Converts a Pauli word to the binary vector representation.

    This functions follows convention that the first half of binary vector components specify
    PauliX placements while the last half specify PauliZ placements.

    **Usage example:**

    If `n_qubits` and `wire_map` are both unspecified, the dimensionality of the binary vector
    will be `2 * len(pauli_word.wires)`. Regardless of wire labels, the vector components encoding
    Pauli operations will be read from left-to-right in the tensor product when `wire_map` is
    unspecified, e.g.

    >>> pauli_to_binary(qml.PauliX('a') @ qml.PauliY('b') @ qml.PauliZ('c'))
    array([1., 1., 0., 0., 1., 1.])
    >>> pauli_to_binary(qml.PauliX('c') @ qml.PauliY('a') @ qml.PauliZ('b'))
    array([1., 1., 0., 0., 1., 1.])

    The above cases have the same binary representation since they are equivalent up to a
    relabelling of the wires. To keep binary vector component enumeration consistent with wire
    labelling across multiple Pauli words, or define any arbitrary enumeration, one can use
    keyword argument `wire_map` to set this enumeration.

    >>> wire_map = {Wires('a'): 0, Wires('b'): 1, Wires('c'): 2}
    >>> pauli_to_binary(qml.PauliX('a') @ qml.PauliY('b') @ qml.PauliZ('c'), wire_map=wire_map)
    array([1., 1., 0., 0., 1., 1.])
    >>> pauli_to_binary(qml.PauliX('c') @ qml.PauliY('a') @ qml.PauliZ('b'), wire_map=wire_map)
    array([1., 0., 1., 1., 1., 0.])

    Now the two Pauli words are distinct in the binary vector representation, as the vector
    components are consistently mapped from the wire labels, rather than enumerated
    left-to-right.

    If `n_qubits` is unspecified, the dimensionality of the vector representation will be inferred
    from the size of support of the Pauli word,

    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(1))
    array([1., 1., 0., 0.])
    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(5))
    array([1., 1., 0., 0.])

    Dimensionality higher than twice the support can be specified by `n_qubits`,

    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(1), n_qubits=6)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(5), n_qubits=6)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    For these Pauli words to have a consistent mapping to vector representation, we once again
    need to specify a `wire_map`.

    >>>wire_map = {Wires(0):0, Wires(1):1, Wires(5):5}
    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(1), n_qubits=6, wire_map=wire_map)
    array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(5), n_qubits=6, wire_map=wire_map)
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    Note that if `n_qubits` is specified and `wire_map` is specified, the dimensionality of the
    vector representation will be inferred from the highest integer in `wire_map.values()`.

    >>>wire_map = {Wires(0):0, Wires(1):1, Wires(5):5}
    >>> pauli_to_binary(qml.PauliX(0) @ qml.PauliX(5),  wire_map=wire_map)
    array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    Args:
        pauli_word (Union[Identity, PauliX, PauliY, PauliZ, Tensor]): the Pauli word to be
            converted to binary vector representation.

    Keyword args:
        n_qubits (int): number of qubits to specify dimension of binary vector representation.
        wire_map (dict): dictionary containing all wire labels used in the Pauli word as keys, and
                         unique integer labels as their values

    Returns:
        array: the 2*n_qubits dimensional binary vector representation of the input Pauli word.

    Raises:
        TypeError: if the input `pauli_word` is not an instance of Identity, PauliX, PauliY,
            PauliZ or tensor products thereof.
        ValueError: if `n_qubits` is less than the number of wires acted on by the Pauli word.
    """

    if not is_pauli_word(pauli_word):
        raise TypeError(
            "Expected a Pauli word Observable instance, instead got {}.".format(pauli_word)
        )

    if isinstance(pauli_word, Tensor):
        pauli_word = pauli_word.prune()

    if not isinstance(pauli_word, Tensor):
        pauli_word = Tensor(pauli_word)

    if wire_map is None:
        num_wires = len(pauli_word.wires)
        wire_map = {pauli_word.wires[i]: i for i in range(num_wires)}

    n_qubits_min = max(wire_map.values()) + 1
    if n_qubits is None:
        n_qubits = n_qubits_min
    elif n_qubits < n_qubits_min:
        raise ValueError(
            "n_qubits must support the highest mapped wire index {},"
            " instead got n_qubits={}.".format(n_qubits_min, n_qubits)
        )

    pauli_wires = pauli_word.wires.map(wire_map).tolist()

    binary_pauli = np.zeros(2 * n_qubits)

    for wire, name in zip(pauli_wires, pauli_word.name):

        if name == "PauliX":
            binary_pauli[wire] = 1

        elif name == "PauliY":
            binary_pauli[wire] = 1
            binary_pauli[n_qubits + wire] = 1

        elif name == "PauliZ":
            binary_pauli[n_qubits + wire] = 1

    return binary_pauli


def binary_to_pauli(binary_vector, wire_map=None):
    """Converts a binary vector of even dimension to an Observable instance.

    This functions follows convention that the first half of binary vector components specify
    PauliX placements while the last half specify PauliZ placements.

    **Usage example:**

    If `wire_map` is unspecified, the Pauli operations follow the same enumerations as the vector
    components, i.e. the `i` and `N+i` components specify the Pauli operation on wire `i`,

    >>> binary_to_pauli([0,1,1,0,1,0])
    Tensor(PauliY(wires=[1]), PauliX(wires=[2]))

    An arbitrary labelling can be assigned by using `wire_map`
    >>> wire_map = {Wires('a'): 0, Wires('b'): 1, Wires('c'): 2}
    >>> binary_to_pauli([0,1,1,0,1,0], wire_map=wire_map)
    Tensor(PauliY(wires=['b']), PauliX(wires=['c']))

    Note that the values of `wire_map`, if specified, must be 0,1,..., N, where N is the dimension
    of the vector divided by two, i.e. `list(wire_map.values())` must be
    `list(range(len(binary_vector)/2))`.

    Args:
        binary_vector (Union[list, tuple, array]): binary vector of even dimension representing a
            unique Pauli word.

    Keyword args:
        wire_map (dict): dictionary containing all wire labels used in the Pauli word as keys, and
                         unique integer labels as their values

    Returns:
        Tensor(Union[Identity, PauliX, PauliY, PauliZ]): the Pauli word corresponding to the input
            binary vector.

    Raises:
        TypeError: if length of binary vector is not even, or if vector does not have strictly
            binary components.

    """

    if isinstance(binary_vector, (list, tuple)):
        binary_vector = np.asarray(binary_vector)

    if len(binary_vector) % 2 != 0:
        raise ValueError(
            "Length of binary_vector must be even, instead got vector of shape {}.".format(
                np.shape(binary_vector)
            )
        )

    if not np.array_equal(binary_vector, binary_vector.astype(bool)):
        raise ValueError(
            "Input vector must have strictly binary components, instead got {}.".format(
                binary_vector
            )
        )

    n_qubits = len(binary_vector) // 2

    if wire_map is not None:
        if list(wire_map.values()) != list(range(n_qubits)):
            raise ValueError(
                "The values of wire_map must be integers 0 to N, for 2N-dimensional binary vector."
                " Instead got wire_map values: {}".format(wire_map.values())
            )
        label_map = {explicit_index: wire_label for wire_label, explicit_index in wire_map.items()}
    else:
        label_map = {i: Wires(i) for i in range(n_qubits)}

    pauli_word = Tensor(Identity(wires=list(label_map.values())[0]))

    for i in range(n_qubits):
        if binary_vector[i] == 1 and binary_vector[n_qubits + i] == 0:
            pauli_word @= PauliX(wires=label_map[i])

        elif binary_vector[i] == 1 and binary_vector[n_qubits + i] == 1:
            pauli_word @= PauliY(wires=label_map[i])

        elif binary_vector[i] == 0 and binary_vector[n_qubits + i] == 1:
            pauli_word @= PauliZ(wires=label_map[i])

    pauli_word = pauli_word.prune()

    if not isinstance(pauli_word, Tensor):
        pauli_word = Tensor(pauli_word)

    return pauli_word


def is_qwc(pauli_vec_1, pauli_vec_2):
    """Checks if two Pauli words in the binary vector representation are qubit-wise commutative.

    **Usage example:**

    >>> is_qwc([1,0,0,1,1,0],[1,0,1,0,1,0])
    False
    >>> is_qwc([1,0,1,1,1,0],[1,0,0,1,1,0])
    True

    Args:
        pauli_vec_1 (Union[list, tuple, array]): first binary vector argument in qubit-wise
            commutator.
        pauli_vec_2 (Union[list, tuple, array]): second binary vector argument in qubit-wise
            commutator.

    Returns:
        bool: returns True if the input Pauli words are qubit-wise commutative, returns False
            otherwise.

    Raises:
        ValueError: if the input vectors are of different dimension, if the vectors are not of even
        dimension, or if the vector components are not strictly binary.

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
            "Symplectic vector space must have even dimension, instead got vectors of shape {}."
            .format(np.shape(pauli_vec_1))
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

        a_x_i = pauli_vec_1[i]
        a_z_i = pauli_vec_1[n_qubits + i]
        b_x_i = pauli_vec_2[i]
        b_z_i = pauli_vec_2[n_qubits + i]

        if not (
            ((a_x_i == a_z_i == 0) or (b_x_i == b_z_i == 0))
            or ((a_x_i == b_x_i) and (a_z_i == b_z_i))
        ):
            return False

    return True


def convert_observables_to_binary(observables, n_qubits=None, wire_map=None):
    """Converts a list of Pauli words to the binary vector representation and yields column matrix
        of the binary vectors.

    The dimension of the binary vectors will be implied from the highest wire being acted on
    non-trivially by the Pauli words in observables.

    **Usage example:**

    >>> convert_observables_to_binary([PauliX(0) @ PauliY(2), PauliZ(0) @ PauliZ(1) @ PauliZ(2)])

    array([[1., 0.],
           [1., 0.],
           [0., 0.],
           [0., 1.],
           [1., 1.],
           [0., 1.]])

    Args:
        observables (list[Union[Identity, PauliX, PauliY, PauliZ, Tensor]]): the list of Pauli
            words.

    Returns:
        array[array[bool]]: a matrix whose columns are Pauli words in binary vector representation.

    """

    m_cols = len(observables)

    for c, obs in enumerate(observables):
        if isinstance(obs, Tensor):
            observables[c] = obs.prune()
        if isinstance(obs, Identity):
            observables[c] = Identity(0)

    if wire_map == None:
        all_wires = Wires.all_wires([pauli_word.wires for pauli_word in observables])
        wire_map = {i: c for c, i in enumerate(all_wires)}

    n_qubits_min = n_qubits_min = max(wire_map.values()) + 1
    if n_qubits == None:
        n_qubits = n_qubits_min
    elif n_qubits < n_qubits_min:
        raise ValueError(
            "n_qubits must support the highest mapped wire index {},"
            " instead got n_qubits={}.".format(n_qubits_min, n_qubits)
        )

    binary_mat = np.zeros((2 * n_qubits, m_cols))

    for i in range(m_cols):
        binary_mat[:, i] = pauli_to_binary(observables[i], n_qubits=n_qubits, wire_map=wire_map)

    return binary_mat


def get_qwc_compliment_adj_matrix(binary_observables):
    """Obtains the adjacency matrix for the complementary graph of the qubit-wise commutativity
    graph for a given set of observables in the binary representation.

    The qubit-wise commutativity graph for a set of Pauli words has a vertex for each Pauli word,
    and two nodes are connected if and only if the corresponding Pauli words are qubit-wise
    commuting.

    **Usage example:**

    >>> binary_observables
    array([[1., 0., 0.],
           [0., 1., 0.],
           [1., 1., 0.],
           [0., 1., 1.],
           [0., 0., 0.],
           [1., 1., 0.]])
    >>> get_qwc_compliment_adj_matrix(binary_observables)
    array([[0., 1., 1.],
           [1., 0., 0.],
           [1., 0., 0.]])

    Args:
        binary_observables (array[array[bool]]): a matrix whose columns are the Pauli words in the
            binary vector representation.

    Returns:
        array[array[bool]]: the adjacency matrix for the complement of the qubit-wise commutativity graph.

    Raises:
        ValueError: if input binary observables contain components which are not strictly binary.

    """

    if not np.array_equal(binary_observables, binary_observables.astype(bool)):
        raise ValueError("Expected a binary array, instead got {}".format(binary_observables))

    if len(np.shape(binary_observables)) == 1:
        binary_observables = np.reshape(binary_observables, (np.shape(binary_observables), 1))

    m_terms = np.shape(binary_observables)[1]

    if isinstance(binary_observables, (list, tuple)):
        binary_observables = np.asarray(binary_observables)

    adj = np.zeros((m_terms, m_terms))
    pauli_vecs = binary_observables.T

    for i in range(m_terms):
        for j in range(i + 1, m_terms):
            adj[i, j] = int(not is_qwc(pauli_vecs[i], pauli_vecs[j]))
            adj[j, i] = adj[i, j]

    return adj
