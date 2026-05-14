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
r"""
Experimental Pauli Based Computation (PBC) functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.pbc

.. autosummary::
    :toctree: api

    ~compare_circuits


Custom operators
~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.pbc

.. autosummary::
    :toctree: api

    ~ControlledPauli
    ~controlled
    ~MeasurePauliWord
    ~measure
    ~ppr

Basic usage
~~~~~~~~~~~

We want to confirm a rule such as

.. code-block::

    0: ─╭P1●─╭P3↗─┤  = ─╭iP1@P3↗───╭P1●─┤
    1: ─╰P2○─╰P4↗─┤  = ─╰iP2@P4↗───╰P2○─┤

which we assume to be true for :math:`[P1, P3] \neq 0` and :math:`[P2, P4] \neq 0`.
To check it, we set out two circuits for each side of the equality sign

.. code-block:: python

    from pennylane.labs import pbc

    def circuit1(P1, P2, P3, P4):
        pbc.controlled(P1, P2)
        pbc.measure(P3 @ P4)

    def circuit2(P1, P2, P3, P4):
        pbc.measure(-P1 @ P3 @ P2 @ P4)
        pbc.controlled(P1, P2)

Next we set concrete values of the Pauliw ords to test the identity.

>>> P1, P3, P2, P4 = X(0) @ X(1), Z(0) @ X(1), Y(2) @ X(3), X(2) @ X(3)
>>> assert qp.commutator(P1, P3) != qp.simplify(0 * P1 @ P3), "P1 and P3 need to anti-commute"
>>> assert qp.commutator(P2, P4) != qp.simplify(0 * P2 @ P4), "P2 and P4 need to anti-commute"
>>> wires = qp.wires.Wires.all_wires([P1.wires, P2.wires, P3.wires, P4.wires])

Finally, we compare the two circuits and see that both sides match.

>>> compare_circuits(
...     circuit1, circuit2, wires=wires, P1=P1, P2=P2, P3=P3, P4=P4, verbose=True
... )
(True, "exact")

"""

from .controlled import ControlledPauli, controlled
from .compare_circuits import compare_circuits
from .pauli_measure import MeasurePauliWord, measure
from .ops import ppr
