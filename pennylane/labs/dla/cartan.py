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
from functools import singledispatch
from typing import List, Tuple, Union

import numpy as np

import pennylane as qml
from pennylane import Y
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
        if involution(op):  # odd parity
            k.append(op)
        else:  # even parity
            m.append(op)

    return k, m


# dispatch to different input types
def even_odd_involution(op: Union[PauliSentence, np.ndarray, Operator]) -> bool:
    r"""The Even-Odd involution

    This is defined in `quant-ph/0701193 <https://arxiv.org/pdf/quant-ph/0701193>`__.
    For Pauli words and sentences, it comes down to counting non-trivial Paulis in Pauli words.

    Args:
        op ( Union[PauliSentence, np.ndarray, Operator]): Input operator

    Returns:
        bool: Boolean output ``True`` or ``False`` for odd (:math:`\mathfrak{k}`) and even parity subspace (:math:`\mathfrak{m}`), respectively

    .. seealso:: :func:`~cartan_decomp`

    **Example**

    >>> from pennylane import X, Y, Z
    >>> from pennylane.labs.dla import even_odd_involution
    >>> ops = [X(0), X(0) @ Y(1), X(0) @ Y(1) @ Z(2)]
    >>> [even_odd_involution(op) for op in ops]
    [True, False, True]

    Operators with an odd number of non-identity Paulis yield ``1``, whereas even ones yield ``0``.

    The function also works with dense matrix representations.

    >>> ops_m = [qml.matrix(op, wire_order=range(3)) for op in ops]
    >>> [even_odd_involution(op_m) for op_m in ops_m]
    [True, False, True]

    """
    return _even_odd_involution(op)


@singledispatch
def _even_odd_involution(op):  # pylint:disable=unused-argument
    return NotImplementedError(f"Involution not defined for operator {op} of type {type(op)}")


@_even_odd_involution.register(PauliSentence)
def _even_odd_involution_ps(op: PauliSentence):
    # Generalization to sums of Paulis: check each term and assert they all have the same parity
    parity = []
    for pw in op.keys():
        parity.append(len(pw) % 2)

    # only makes sense if parity is the same for all terms, e.g. Heisenberg model
    assert all(
        parity[0] == p for p in parity
    ), f"The Even-Odd involution is not well-defined for operator {op} as individual terms have different parity"
    return bool(parity[0])


@_even_odd_involution.register(np.ndarray)
def _even_odd_involution_matrix(op: np.ndarray):
    """see Table CI in https://arxiv.org/abs/2406.04418"""
    n = int(np.round(np.log2(op.shape[-1])))
    YYY = qml.prod(*[Y(i) for i in range(n)])
    YYY = qml.matrix(YYY, range(n))

    transformed = YYY @ op.conj() @ YYY
    if np.allclose(transformed, op):
        return False
    if np.allclose(transformed, -op):
        return True
    raise ValueError(f"The Even-Odd involution is not well-defined for operator {op}.")


@_even_odd_involution.register(Operator)
def _even_odd_involution_op(op: Operator):
    """use pauli representation"""
    return _even_odd_involution_ps(op.pauli_rep)


# dispatch to different input types
def concurrence_involution(op: Union[PauliSentence, np.ndarray, Operator]) -> bool:
    r"""The Concurrence Canonical Decomposition :math:`\Theta(g) = -g^T` as a Cartan involution function

    This is defined in `quant-ph/0701193 <https://arxiv.org/pdf/quant-ph/0701193>`__.
    For Pauli words and sentences, it comes down to counting Pauli-Y operators.

    Args:
        op ( Union[PauliSentence, np.ndarray, Operator]): Input operator

    Returns:
        bool: Boolean output ``True`` or ``False`` for odd (:math:`\mathfrak{k}`) and even parity subspace (:math:`\mathfrak{m}`), respectively

    .. seealso:: :func:`~cartan_decomp`

    **Example**

    >>> from pennylane import X, Y, Z
    >>> from pennylane.labs.dla import concurrence_involution
    >>> ops = [X(0), X(0) @ Y(1), X(0) @ Y(1) @ Z(2), Y(0) @ Y(2)]
    >>> [concurrence_involution(op) for op in ops]
    [False, True, True, False]

    Operators with an odd number of ``Y`` operators yield ``1``, whereas even ones yield ``0``.

    The function also works with dense matrix representations.

    >>> ops_m = [qml.matrix(op, wire_order=range(3)) for op in ops]
    >>> [even_odd_involution(op_m) for op_m in ops_m]
    [False, True, True, False]

    """
    return _concurrence_involution(op)


@singledispatch
def _concurrence_involution(op):
    return NotImplementedError(f"Involution not defined for operator {op} of type {type(op)}")


@_concurrence_involution.register(PauliSentence)
def _concurrence_involution_pauli(op: PauliSentence):
    # Generalization to sums of Paulis: check each term and assert they all have the same parity
    parity = []
    for pw in op.keys():
        result = sum(1 if el == "Y" else 0 for el in pw.values())
        parity.append(result % 2)

    # only makes sense if parity is the same for all terms, e.g. Heisenberg model
    assert all(
        parity[0] == p for p in parity
    ), f"The concurrence canonical decomposition is not well-defined for operator {op} as individual terms have different parity"
    return bool(parity[0])


@_concurrence_involution.register(Operator)
def _concurrence_involution_operator(op: Operator):
    return _concurrence_involution_matrix(op.matrix())


@_concurrence_involution.register(np.ndarray)
def _concurrence_involution_matrix(op: np.ndarray):
    if np.allclose(op, -op.T):
        return True
    if np.allclose(op, op.T):
        return False
    raise ValueError(
        f"The concurrence canonical decomposition is not well-defined for operator {op}"
    )
