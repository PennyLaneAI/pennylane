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
This module contains the qml.evolve function.
"""
from typing import Union

from pennylane.operation import Operator
from pennylane.ops import Evolution
from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian


def evolve(op: Union[Operator, ParametrizedHamiltonian]):
    """Returns a new operator that computes the evolution of ``op``.

    Args:
        op (Union[.Operator, .ParametrizedHamiltonian]): operator to evolve

    Returns:
        Union[.Evolution, ~pennylane.ops.op_math.evolve.ParametrizedEvolution]: evolution operator

    .. seealso:: :class:`.ParametrizedEvolution`
    .. seealso:: :class:`.Evolution`

    **Examples**

    We can use ``qml.evolve`` to compute the evolution of any PennyLane operator:

    >>> op = qml.s_prod(2, qml.PauliX(0))
    >>> qml.evolve(op)
    Exp((-0-1j) 2*(PauliX(wires=[0])))

    When evolving a :class:`.ParametrizedHamiltonian` class, then a :class:`.ParametrizedEvolution`
    instance is returned:

    >>> coeffs = [lambda p, t: p * t for _ in range(4)]
    >>> ops = [qml.PauliX(i) for i in range(4)]
    >>> H = qml.ops.dot(coeffs, ops)
    >>> qml.evolve(H)
    ParametrizedEvolution(wires=[0, 1, 2, 3])

    Please check the :class:`.ParametrizedEvolution` class for more information.
    """
    if isinstance(op, ParametrizedHamiltonian):
        return ParametrizedEvolution(H=op)

    return Evolution(generator=op, param=1.0)
