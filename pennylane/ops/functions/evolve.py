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
This module contains the qp.evolve function.
"""

from pennylane.core.operator import Operator
from pennylane.ops import Evolution


def evolve(op: Operator, coeff: float = 1):
    r"""Returns a new operator that computes the evolution of ``op``.

    .. math::

        e^{-i x \bm{O}}

    Args:
        op (.Operator): operator to evolve
        coeff (float): coefficient multiplying the exponentiated operator

    Returns:
        .Evolution: evolution operator

    **Examples**

    We can use ``qp.evolve`` to compute the evolution of any PennyLane operator:

    >>> op = qp.evolve(qp.X(0), coeff=2)
    >>> op
    Evolution(-2j PauliX)
    """
    return Evolution(op, coeff)
