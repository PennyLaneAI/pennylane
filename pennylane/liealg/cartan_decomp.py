# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functionality for Cartan decomposition"""
# pylint: disable= missing-function-docstring
from typing import List, Tuple, Union

import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence


def cartan_decomp(
    g: List[Union[PauliSentence, Operator]], involution: callable
) -> Tuple[List[Union[PauliSentence, Operator]], List[Union[PauliSentence, Operator]]]:
    r"""Cartan Decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}`.

    Given a Lie algebra :math:`\mathfrak{g}`, the Cartan decomposition is a decomposition
    :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}` into orthogonal complements.
    This is realized by an involution :math:`\Theta(g)` that maps each operator :math:`g \in \mathfrak{g}`
    back to itself after two consecutive applications, i.e., :math:`\Theta(\Theta(g)) = g \ \forall g \in \mathfrak{g}`.

    The ``involution`` argument can be any function that maps the operators in the provided ``g`` to a boolean output.
    ``True`` for operators that go into :math:`\mathfrak{k}` and ``False`` for operators in :math:`\mathfrak{m}`.

    The resulting subspaces fulfill the Cartan commutation relations

    .. math:: [\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k} \text{ ; } [\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m} \text{ ; } [\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k}

    Args:
        g (List[Union[PauliSentence, Operator]]): the (dynamical) Lie algebra to decompose
        involution (callable): Involution function :math:`\Theta(\cdot)` to act on the input operator, should return ``0/1`` or ``False/True``.
            E.g., :func:`~even_odd_involution` or :func:`~concurrence_involution`.

    Returns:
        Tuple(List[Union[PauliSentence, Operator]], List[Union[PauliSentence, Operator]]): Tuple ``(k, m)`` containing the even
        parity subspace :math:`\Theta(\mathfrak{k}) = \mathfrak{k}` and the odd
        parity subspace :math:`\Theta(\mathfrak{m}) = -\mathfrak{m}`.

    .. seealso:: :func:`~even_odd_involution`, :func:`~concurrence_involution`, :func:`~check_cartan_decomp`

    **Example**

    We first construct a Lie algebra.

    >>> from pennylane import X, Z
    >>> from pennylane.labs.dla import concurrence_involution, even_odd_involution, cartan_decomp
    >>> generators = [X(0) @ X(1), Z(0), Z(1)]
    >>> g = qml.lie_closure(generators)
    >>> g
    [X(0) @ X(1),
     Z(0),
     Z(1),
     -1.0 * (Y(0) @ X(1)),
     -1.0 * (X(0) @ Y(1)),
     -1.0 * (Y(0) @ Y(1))]

    We compute the Cartan decomposition with respect to the :func:`~concurrence_involution`.

    >>> k, m = cartan_decomp(g, concurrence_involution)
    >>> k, m
    ([-1.0 * (Y(0) @ X(1)), -1.0 * (X(0) @ Y(1))],
     [X(0) @ X(1), Z(0), Z(1), -1.0 * (Y(0) @ Y(1))])

    We can check the validity of the decomposition using :func:`~check_cartan_decomp`.

    >>> check_cartan_decomp(k, m)
    True

    There are other Cartan decomposition induced by other involutions. For example using :func:`~even_odd_involution`.

    >>> from pennylane.labs.dla import check_cartan_decomp
    >>> k, m = cartan_decomp(g, even_odd_involution)
    >>> k, m
    ([Z(0), Z(1)],
     [X(0) @ X(1),
      -1.0 * (Y(0) @ X(1)),
      -1.0 * (X(0) @ Y(1)),
      -1.0 * (Y(0) @ Y(1))])
    >>> check_cartan_decomp(k, m)
    True
    """
    # simple implementation assuming all elements in g are already either in k and m
    # TODO: Figure out more general way to do this when the above is not the case
    m = []
    k = []

    for op in g:
        if involution(op):  # odd parity theta(k) = k
            k.append(op)
        else:  # even parity theta(m) = -m
            m.append(op)

    return k, m


def check_commutation(ops1, ops2, vspace):
    r"""Helper function to check :math:`[\text{ops1}, \text{ops2}] \subseteq \text{vspace}`.

    .. warning:: This function is expensive to compute

    Args:
        ops1 (Iterable[PauliSentence]): First set of operators
        ops2 (Iterable[PauliSentence]): Second set of operators
        vspace (:class:`~PauliVSpace`): The vector space in form of a :class:`~PauliVSpace` that the operators should map to

    Returns:
        bool: Whether or not :math:`[\text{ops1}, \text{ops2}] \subseteq \text{vspace}`

    **Example**

    >>> from pennylane.liealg import check_commutation
    >>> ops1 = [qml.X(0)]
    >>> ops2 = [qml.Y(0)]
    >>> vspace1 = [qml.X(0), qml.Y(0)]

    Because :math:`[X_0, Y_0] = 2i Z_0`, the commutators do not map to the selected vector space.

    >>> check_commutation(ops1, ops2, vspace1)
    False

    Instead, we need the full :math:`\mathfrak{su}(2)` space.

    >>> vspace2 = [qml.X(0), qml.Y(0), qml.Z(0)]
    >>> check_commutation(ops1, ops2, vspace2)
    True
    """

    if any(isinstance(op, Operator) for op in ops1):
        ops1 = [op.pauli_rep for op in ops1]

    if any(isinstance(op, Operator) for op in ops2):
        ops2 = [op.pauli_rep for op in ops2]

    if not isinstance(vspace, qml.pauli.PauliVSpace):
        vspace = qml.pauli.PauliVSpace(vspace, dtype=complex)

    for o1 in ops1:
        for o2 in ops2:
            com = o1.commutator(o2)
            com.simplify()
            if len(com) != 0:
                if vspace.is_independent(com):
                    return False

    return True


def check_cartan_decomp(k: List[PauliSentence], m: List[PauliSentence], verbose=True):
    r"""Helper function to check the validity of a Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}.`

    Check whether of not the following properties are fulfilled.

    .. math::

            [\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k} & \text{ (subalgebra)}\\
            [\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m} & \text{ (reductive property)}\\
            [\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k} & \text{ (symmetric property)}

    .. warning:: This function is expensive to compute

    Args:
        k (List[PauliSentence]): List of operators of the vertical subspace
        m (List[PauliSentence]): List of operators of the horizontal subspace
        verbose: Whether failures to meet one of the criteria should be printed

    Returns:
        bool: Whether or not all properties are fulfilled

    .. seealso:: :func:`~cartan_decomp`

    **Example**

    We first construct a Lie algebra.

    >>> from pennylane import X, Z
    >>> from pennylane.labs.dla import concurrence_involution, even_odd_involution, cartan_decomp
    >>> generators = [X(0) @ X(1), Z(0), Z(1)]
    >>> g = qml.lie_closure(generators)
    >>> g
    [X(0) @ X(1),
     Z(0),
     Z(1),
     -1.0 * (Y(0) @ X(1)),
     -1.0 * (X(0) @ Y(1)),
     -1.0 * (Y(0) @ Y(1))]

    We compute the Cartan decomposition with respect to the :func:`~concurrence_involution`.

    >>> k, m = cartan_decomp(g, concurrence_involution)
    >>> k, m
    ([-1.0 * (Y(0) @ X(1)), -1.0 * (X(0) @ Y(1))],
     [X(0) @ X(1), Z(0), Z(1), -1.0 * (Y(0) @ Y(1))])

    We can check the validity of the decomposition using ``check_cartan_decomp``.

    >>> from pennylane.labs.dla import check_cartan_decomp
    >>> check_cartan_decomp(k, m)
    True

    """
    if any(isinstance(op, np.ndarray) for op in k):
        k = [qml.pauli_decompose(op).pauli_rep for op in k]
    if any(isinstance(op, np.ndarray) for op in m):
        m = [qml.pauli_decompose(op).pauli_rep for op in m]

    if any(isinstance(op, Operator) for op in k):
        k = [op.pauli_rep for op in k]
    if any(isinstance(op, Operator) for op in m):
        m = [op.pauli_rep for op in m]

    k_space = qml.pauli.PauliVSpace(k, dtype=complex)
    m_space = qml.pauli.PauliVSpace(m, dtype=complex)

    # Commutation relations for Cartan pair
    if not (check_kk := check_commutation(k, k, k_space)):
        _ = print("[k, k] sub k not fulfilled") if verbose else None
    if not (check_km := check_commutation(k, m, m_space)):
        _ = print("[k, m] sub m not fulfilled") if verbose else None
    if not (check_mm := check_commutation(m, m, k_space)):
        _ = print("[m, m] sub k not fulfilled") if verbose else None

    return all([check_kk, check_km, check_mm])
