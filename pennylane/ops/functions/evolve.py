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
from typing import Union, Tuple

from pennylane.operation import Operator
from pennylane.ops import Evolution
from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian
import pennylane as qml


def evolve(
    op: Union[Operator, ParametrizedHamiltonian],
    t: Union[float, Tuple[float]] = None,
    dt: float = None,
):
    r"""Returns a new operator that computes the evolution of ``op``.

    Args:
        op (Union[.Operator, .ParametrizedHamiltonian]): operator to evolve
        t (Union[float, Tuple[float]]): Time.

            * If ``op`` is a :class:`.Operator`, ``t`` corresponds to the coefficient multiplying
                the exponentiated operator: :math:`\exp\{i\bm{O}t)\}`. If ``None``, ``t=1`` is used.

            * If ``op`` is a :class:`.ParametrizedHamiltonian`, ``t`` can be either a float or a
                tuple of floats. If a float, it corresponds to the duration of the evolution (start
                time is 0). If a tuple of floats, it corresponds to the start and end time of the evolution.
                Note that such absolute times only have meaning within an instance of
                ``ParametrizedEvolution`` and will not affect other gates.

        dt (float): Time step used.

            * If ``op`` is a :class:`.Operator`, ``1/dt`` corresponds to the Trotter number, which
                is the number of time steps used in the Suzuki-Trotter decomposition of the
                :class:`.Evolution` operator.

            * If ``op`` is a :class:`.ParametrizedHamiltonian`, ``dt`` will be
                used to convert the initial and final time values into a list of values that the ``odeint``
                solver will use. The solver might perform intermediate steps if necessary. It is recommended
                to not provide a value for ``dt``.

    Returns:
        Union[.Evolution, ~pennylane.ops.op_math.evolve.ParametrizedEvolution]: evolution operator

    .. seealso:: :class:`.ParametrizedEvolution`
    .. seealso:: :class:`.Evolution`

    **Examples**

    We can use ``qml.evolve`` to compute the evolution of any PennyLane operator:

    >>> op = qml.evolve(qml.PauliX(0), t=2)
    >>> op
    Exp(2j PauliX)
    >>> op.decomposition()
    [RX((-4+0j), wires=[0])]

    When we have an exponential of a linear combination of operators, the Suzuki-Trotter algorithm
    is used to decompose the initial operator into a product of exponentials:

    >>> op = 3 * qml.PauliX(0) + 7 * qml.PauliZ(1) @ qml.PauliX(2)
    >>> ev = qml.evolve(op, dt=0.5)
    >>> ev
    Exp(1j Hamiltonian)
    >>> ev.decomposition()
    [PauliRot((-3+0j), X, wires=[0]),
    PauliRot((-7+0j), ZX, wires=[1, 2]),
    PauliRot((-3+0j), X, wires=[0]),
    PauliRot((-7+0j), ZX, wires=[1, 2])]

    When evolving a :class:`.ParametrizedHamiltonian` class, then a :class:`.ParametrizedEvolution`
    instance is returned:

    >>> coeffs = [lambda p, t: p * t for _ in range(4)]
    >>> ops = [qml.PauliX(i) for i in range(4)]
    >>> H = qml.dot(coeffs, ops)
    >>> qml.evolve(H, t=[4, 10])
    ParametrizedEvolution(wires=[0, 1, 2, 3])

    The :class:`.ParametrizedEvolution` instance can then be called to update the needed attributes
    to compute the evolution of the :class:`.ParametrizedHamiltonian`:

    >>> qml.evolve(H, t=[4, 10])(params=[1., 2., 3.], atol=1e-6, mxstep=1)
    ParametrizedEvolution(wires=[0, 1, 2, 3])

    Please check the :class:`.ParametrizedEvolution` class for more information.
    """
    if isinstance(op, ParametrizedHamiltonian):
        if t is None:
            raise ValueError("Time must be specified to evolve a ParametrizedHamiltonian.")
        t = [0, t] if qml.math.ndim(t) == 0 else t
        if dt is not None:
            t = qml.math.arange(*t, step=dt)

        return ParametrizedEvolution(H=op, t=t)

    num_steps = dt if dt is None else int(1 / dt)

    return (
        Evolution(op, num_steps=num_steps)
        if t is None
        else Evolution(op, param=t, num_steps=num_steps)
    )
