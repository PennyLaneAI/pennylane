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
from typing import Union

import numpy as np

from pennylane.operation import Operator
from pennylane.pauli import PauliSentence


def cartan_decomposition(g, involution):
    r"""Cartan Decomposition g = k + m

    Args:
        g (List[Union[PauliSentence, Operator]]): the (dynamical) Lie algebra to decompose
        involution (callable): Involution function :math:`\Theta(\cdot)` to act on the input operator, should return ``0/1`` or ``True/False``.

    Returns:
        k (List[Union[PauliSentence, Operator]]): the even parity subspace :math:`\Theta(\mathfrak{k}) = \mathfrak{k}`
        m (List[Union[PauliSentence, Operator]]): the odd parity subspace :math:`\Theta(\mathfrak{m}) = \mathfrak{m}`
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


def even_odd_involution(op: PauliSentence):
    """Generalization of EvenOdd involution to sums of Paulis"""
    parity = []
    for pw in op.keys():
        parity.append(len(pw) % 2)

    assert all(
        parity[0] == p for p in parity
    )  # only makes sense if parity is the same for all terms, e.g. Heisenberg model
    return parity[0]


def concurrence_involution(op: Union[PauliSentence, np.ndarray, Operator]):
    r"""The Concurrence Canonical Decomposition :math:`\Theta(g) = -g^T` as a Cartan involution function

    This is defined in `quant-ph/0701193 <https://arxiv.org/pdf/quant-ph/0701193>`__, and for Pauli words and sentences comes down to counting Pauli-Y operators.

    This implementation is specific to ``PauliSentence`` instances

    Args:
        op ( Union[PauliSentence, np.ndarray, Operator]): Input operator

    Returns:
        bool: Boolean output ``True`` or ``False`` for even and odd parity subspace, respectively

    """
    return _concurrence_involution(op)


@singledispatch
def _concurrence_involution(op):  # pylint:disable=unused-argument
    """
    Private implementation of _concurrence_involution, to prevent all of the
    registered functions from appearing in the Sphinx docs.
    """
    return False


@_concurrence_involution.register(PauliSentence)
def _concurrence_involution_pauli(op: PauliSentence):
    parity = []
    for pw in op.keys():
        result = sum(1 if el == "Y" else 0 for el in pw.values())
        parity.append(result % 2)

    assert all(
        parity[0] == p for p in parity
    )  # only makes sense if parity is the same for all terms, e.g. Heisenberg model
    return bool(parity[0])


@_concurrence_involution.register(Operator)
def _concurrence_involution_operation(op: Operator):
    op = op.matrix()
    return np.allclose(op, -op.T)


@_concurrence_involution.register(np.ndarray)
def _concurrence_involution_matrix(op: np.ndarray):
    return np.allclose(op, -op.T)
