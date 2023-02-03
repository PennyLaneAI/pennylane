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
from functools import singledispatch

from pennylane.operation import Operator
from pennylane.ops import Evolution
from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian


@singledispatch
def evolve(op, *args, **kwargs):  # pylint: disable=unused-argument
    """Returns a new operator that computes the evolution of ``op``.

    Depending on the ``op`` type, this function is dispatched into the following functions:

    * :func:`~.evolution` if ``op`` is an :class:`~.Operator`.

    * :func:`~.parametrized_evolution` if ``op`` is a :class:`~.ParametrizedHamiltonian`.

    """


@evolve.register
def parametrized_evolution(op: ParametrizedHamiltonian):
    """Returns a new operator that computes the evolution of ``op``.

    Args:
        op (.ParametrizedHamiltonian): operator to evolve

    Returns:
        ~pennylane.ops.op_math.evolve.ParametrizedEvolution: evolution operator

    .. seealso:: :class:`.ParametrizedEvolution`

    **Examples**

    When evolving a :class:`.ParametrizedHamiltonian` class, then a :class:`.ParametrizedEvolution`
    instance is returned:

    >>> coeffs = [lambda p, t: p * t for _ in range(4)]
    >>> ops = [qml.PauliX(i) for i in range(4)]
    >>> H = qml.dot(coeffs, ops)
    >>> qml.evolve(H)
    ParametrizedEvolution(wires=[0, 1, 2, 3])

    The :class:`.ParametrizedEvolution` instance can then be called to update the needed attributes
    to compute the evolution of the :class:`.ParametrizedHamiltonian`:

    >>> qml.evolve(H)(params=[1., 2., 3.], t=[4, 10], atol=1e-6, mxstep=1)
    ParametrizedEvolution(wires=[0, 1, 2, 3])

    Please check the :class:`.ParametrizedEvolution` class for more information.
    """
    return ParametrizedEvolution(H=op)


@evolve.register
def evolution(op: Operator, coeff: float = 1):
    r"""Returns a new operator that computes the evolution of ``op``.

    Args:
        op (Union[.Operator, .ParametrizedHamiltonian]): operator to evolve
        coeff (float): coefficient multiplying the exponentiated operator: :math:`\exp\{i\bm{O}x)\}`.

    Returns:
        .Evolution: evolution operator

    .. seealso:: :class:`.Evolution`

    **Examples**

    We can use ``qml.evolve`` to compute the evolution of any PennyLane operator:

    >>> op = qml.evolve(qml.PauliX(0), coeff=2)
    >>> op
    Exp(2j PauliX)
    """
    return Evolution(op, coeff)
