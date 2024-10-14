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


def CartanDecomp(g, involution):
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
