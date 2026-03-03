# Copyright 2025 Xanadu Quantum Technologies Inc.

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


from pennylane import math
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliVSpace
from pennylane.typing import TensorLike


def cartan_decomp(
    g: list[PauliSentence | Operator], involution: callable
) -> tuple[list[PauliSentence | Operator], list[PauliSentence | Operator]]:
    r"""Compute the Cartan Decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}` of a Lie algebra :math:`\mathfrak{g}`.

    Given a Lie algebra :math:`\mathfrak{g}`, the Cartan decomposition is a decomposition
    :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}` into orthogonal complements.
    This is realized by an involution :math:`\Theta(g)` that maps each operator :math:`g \in \mathfrak{g}`
    back to itself after two consecutive applications, i.e., :math:`\Theta(\Theta(g)) = g \ \forall g \in \mathfrak{g}`.

    The ``involution`` argument can be any function that maps the operators in the provided ``g`` to a boolean output.
    ``True`` for operators that go into :math:`\mathfrak{k}` and ``False`` for operators in :math:`\mathfrak{m}`.
    It is assumed that all operators in the input ``g`` belong to either :math:`\mathfrak{k}` or
    :math:`\mathfrak{m}`.

    The resulting subspaces fulfill the Cartan commutation relations

    .. math:: [\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k} \text{ ; } [\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m} \text{ ; } [\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k}

    Args:
        g (List[Union[PauliSentence, Operator]]): the (dynamical) Lie algebra to decompose.
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
    >>> from pennylane.liealg import concurrence_involution, even_odd_involution, cartan_decomp
    >>> generators = [X(0) @ X(1), Z(0), Z(1)]
    >>> g = qml.lie_closure(generators)
    >>> g
    [X(0) @ X(1), Z(0), Z(1), -1.0 * (Y(0) @ X(1)), -1.0 * (X(0) @ Y(1)), Y(0) @ Y(1)]

    We compute the Cartan decomposition with respect to the :func:`~concurrence_involution`.

    >>> k, m = cartan_decomp(g, concurrence_involution)
    >>> k, m
    ([-1.0 * (Y(0) @ X(1)), -1.0 * (X(0) @ Y(1))], [X(0) @ X(1), Z(0), Z(1), Y(0) @ Y(1)])

    We can check the validity of the decomposition using :func:`~check_cartan_decomp`.

    >>> check_cartan_decomp(k, m)
    True

    There are other Cartan decomposition induced by other involutions. For example using :func:`~even_odd_involution`.

    >>> from pennylane.liealg import check_cartan_decomp
    >>> k, m = cartan_decomp(g, even_odd_involution)
    >>> k, m
     ([Z(0), Z(1)], [X(0) @ X(1), -1.0 * (Y(0) @ X(1)), -1.0 * (X(0) @ Y(1)), Y(0) @ Y(1)])
    >>> check_cartan_decomp(k, m)
    True
    """
    # simple implementation assuming all elements in g are already either in k and m
    m = []
    k = []

    for op in g:
        if involution(op):  # theta(k) = k
            k.append(op)
        else:  # theta(m) = -m
            m.append(op)

    return k, m


def check_commutation_relation(
    ops1: list[PauliSentence | TensorLike],
    ops2: list[PauliSentence | TensorLike],
    vspace: PauliVSpace | list[PauliSentence | TensorLike],
):
    r"""Helper function to check :math:`[\text{ops1}, \text{ops2}] \subseteq \text{vspace}`.

    .. warning:: This function is expensive to compute.

    Args:
        ops1 (List[Union[PauliSentence, TensorLike]]): First set of operators.
        ops2 (List[Union[PauliSentence, TensorLike]]): Second set of operators.
        vspace (Union[PauliVSpace, List[Union[PauliSentence, TensorLike]]]): The vector space in form of a :class:`~PauliVSpace` that the operators should map to.

    Returns:
        bool: Whether or not :math:`[\text{ops1}, \text{ops2}] \subseteq \text{vspace}`.

    **Example**

    >>> from pennylane.liealg import check_commutation_relation
    >>> ops1 = [qml.X(0)]
    >>> ops2 = [qml.Y(0)]
    >>> vspace1 = [qml.X(0), qml.Y(0)]

    Because :math:`[X_0, Y_0] = 2i Z_0`, the commutators do not map to the selected vector space.

    >>> check_commutation_relation(ops1, ops2, vspace1)
    False

    Instead, we need the full :math:`\mathfrak{su}(2)` space.

    >>> vspace2 = [qml.X(0), qml.Y(0), qml.Z(0)]
    >>> check_commutation_relation(ops1, ops2, vspace2)
    True
    """

    ops1_is_tensor = any(isinstance(op, TensorLike) for op in ops1)
    ops2_is_tensor = any(isinstance(op, TensorLike) for op in ops2)
    if not isinstance(vspace, PauliVSpace):
        vspace_is_tensor = any(isinstance(op, TensorLike) for op in vspace)
        if any(isinstance(op, Operator) for op in vspace):
            vspace = PauliVSpace(vspace, dtype=complex)

    else:
        vspace_is_tensor = False

    all_tensors = all((ops1_is_tensor, ops2_is_tensor, vspace_is_tensor))
    any_tensors = any((ops1_is_tensor, ops2_is_tensor, vspace_is_tensor))
    if not all_tensors and any_tensors:
        raise TypeError(
            "All inputs `ops1`, `ops2` and `vspace` to qml.liealg.check_commutation_relation need to either be iterables of operators or matrices."
        )

    if all_tensors:
        all_coms = _all_coms(ops1, ops2)
        return _is_subspace(all_coms, vspace)

    if any(isinstance(op, Operator) for op in ops1):
        ops1 = [op.pauli_rep for op in ops1]

    if any(isinstance(op, Operator) for op in ops2):
        ops2 = [op.pauli_rep for op in ops2]

    for o1 in ops1:
        for o2 in ops2:
            com = o1.commutator(o2)
            com.simplify()
            if len(com) != 0:
                if vspace.is_independent(com):
                    return False

    return True


def _all_coms(vspace1, vspace2):
    r"""Compute all commutators [Vspace1, Vspace2]"""
    chi = len(vspace1[0])

    m0m1 = math.moveaxis(math.tensordot(vspace1, vspace2, axes=[[2], [1]]), 1, 2)
    m0m1 = math.reshape(m0m1, (-1, chi, chi))

    # Implement einsum "aij,bki->abkj" by tensordot and moveaxis
    m1m0 = math.moveaxis(math.tensordot(vspace1, vspace2, axes=[[1], [2]]), 1, 3)
    m1m0 = math.reshape(m1m0, (-1, chi, chi))
    all_coms = m0m1 - m1m0
    return all_coms


def _is_subspace(subspace, vspace):
    r"""check if subspace <= vspace"""
    # Check if rank increases by adding matices from ``subspace``
    # Use matrices as vectors -> flatten matrix dimensions (chi, chi) to (chi**2,)
    vspace = math.reshape(vspace, (len(vspace), -1))
    subspace = math.reshape(subspace, (len(subspace), -1))

    rank_V = math.linalg.matrix_rank(vspace)
    rank_both = math.linalg.matrix_rank(math.vstack([vspace, subspace]))

    return rank_both <= rank_V


def check_cartan_decomp(
    k: list[PauliSentence | TensorLike],
    m: list[PauliSentence | TensorLike],
    verbose=True,
):
    r"""Helper function to check the validity of a Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}.`

    Check whether of not the following properties are fulfilled.

    .. math::

            [\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k} & \text{ (subalgebra)}\\
            [\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m} & \text{ (reductive property)}\\
            [\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k} & \text{ (symmetric property)}

    .. warning:: This function is expensive to compute

    Args:
        k (List[Union[PauliSentence, TensorLike]]): List of operators of the vertical subspace.
        m (List[Union[PauliSentence, TensorLike]]): List of operators of the horizontal subspace.
        verbose: Whether failures to meet one of the criteria should be printed.

    Returns:
        bool: Whether or not all of the Cartan commutation relations are fulfilled.

    .. seealso:: :func:`~cartan_decomp`

    **Example**

    We first construct a Lie algebra.

    >>> from pennylane import X, Z
    >>> from pennylane.liealg import concurrence_involution, even_odd_involution, cartan_decomp
    >>> generators = [X(0) @ X(1), Z(0), Z(1)]
    >>> g = qml.lie_closure(generators)
    >>> g
    [X(0) @ X(1), Z(0), Z(1), -1.0 * (Y(0) @ X(1)), -1.0 * (X(0) @ Y(1)), Y(0) @ Y(1)]

    We compute the Cartan decomposition with respect to the :func:`~concurrence_involution`.

    >>> k, m = cartan_decomp(g, concurrence_involution)
    >>> k, m
    ([-1.0 * (Y(0) @ X(1)), -1.0 * (X(0) @ Y(1))], [X(0) @ X(1), Z(0), Z(1), Y(0) @ Y(1)])

    We can check the validity of the decomposition using ``check_cartan_decomp``.

    >>> from pennylane.liealg import check_cartan_decomp
    >>> check_cartan_decomp(k, m)
    True

    """

    if any(isinstance(op, TensorLike) for op in k) or any(isinstance(op, TensorLike) for op in m):
        if not all(isinstance(op, TensorLike) for op in k) or not all(
            isinstance(op, TensorLike) for op in m
        ):
            raise TypeError(
                "All inputs `k`, `m` to check_cartan_decomp need to either be iterables of "
                f"operators or matrices. Received `k` of types {[type(op) for op in k]} and "
                f"`m` of types {[type(op) for op in m]}"
            )

    if any(isinstance(op, Operator) for op in k):
        k = [op.pauli_rep for op in k]
    if any(isinstance(op, Operator) for op in m):
        m = [op.pauli_rep for op in m]

    if any(isinstance(op, TensorLike) for op in k):
        k_space = k
    else:
        k_space = PauliVSpace(k, dtype=complex)

    if any(isinstance(op, TensorLike) for op in m):
        m_space = m
    else:
        m_space = PauliVSpace(m, dtype=complex)

    # Commutation relations for Cartan pair
    if not (check_kk := check_commutation_relation(k, k, k_space)):
        _ = print("[k, k] sub k not fulfilled") if verbose else None
    if not (check_km := check_commutation_relation(k, m, m_space)):
        _ = print("[k, m] sub m not fulfilled") if verbose else None
    if not (check_mm := check_commutation_relation(m, m, k_space)):
        _ = print("[m, m] sub k not fulfilled") if verbose else None

    return all([check_kk, check_km, check_mm])
