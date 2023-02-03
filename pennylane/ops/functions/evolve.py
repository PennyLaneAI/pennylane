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
def evolve(*args, **kwargs):  # pylint: disable=unused-argument
    r"""This method is dispatched and its functionality depends on the type of the input ``op``.

    .. raw:: html

        <html>
            <h3>Input: Operator</h3>
            <hr>
        </html>

    Returns a new operator that computes the evolution of ``op``.

    .. math::

        \exp\{-i \times \bm{op} \times coeff)\}

    Args:
        op (.Operator): operator to evolve
        coeff (float): coefficient multiplying the exponentiated operator

    Returns:
        .Evolution: evolution operator

    .. seealso:: :class:`~.Evolution`

    **Examples**

    We can use ``qml.evolve`` to compute the evolution of any PennyLane operator:

    >>> op = qml.evolve(qml.PauliX(0), coeff=2)
    >>> op
    Exp(2j PauliX)

    .. raw:: html

        <html>
            <h3>Input: ParametrizedHamiltonian</h3>
            <hr>
        </html>

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


# pylint: disable=missing-docstring
@evolve.register
def parametrized_hamiltonian(op: ParametrizedHamiltonian):
    return ParametrizedEvolution(H=op)


# pylint: disable=missing-docstring
@evolve.register
def evolution(op: Operator, coeff: float = 1, num_steps: int = None):
    return Evolution(op, -1 * coeff, num_steps)
