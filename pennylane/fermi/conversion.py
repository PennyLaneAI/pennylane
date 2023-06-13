# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to convert between ``~.FermiWord`` and other PennyLane formats."""

from functools import singledispatch
from typing import Union

from pennylane.operation import Operator
from pennylane.pauli import PauliWord, PauliSentence
from .fermionic import FermiWord, FermiSentence


@singledispatch
def jordan_wigner(
    fermi_operator: (Union[FermiWord, FermiSentence]), ps=False, wire_order=None
) -> Union[Operator, PauliSentence]:
    r"""Convert a fermionic operator to a qubit operator using the Jordan-Wigner mapping.

    The fermionic creation and annihilation operators are mapped to the Pauli operators as

    .. math::

        a^{\dagger}_N = Z_0 \otimes  ... \otimes Z_{N-1} \otimes \left ( \frac{X-iY}{2} \right ),
        a_N = Z_0 \otimes  ... \otimes Z_{N-1} \otimes \left ( \frac{X+iY}{2} \right ),

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators.

    Args:
        fermi_operator(FermiWord, FermiSentence): the fermionic operator
        ps (bool): whether to return the result as a PauliSentence instead of an
            Operator. Defaults to False.
        wire_map (dict): a dictionary defining how to map the oribitals of
            the Fermi operator to qubit wires. If None, the integers used to
            order the orbitals will be used as wire labels. Defaults to None.

    Returns:
        Union[PauliSentence, Operator]: a linear combination of qubit operators

    **Example**

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> jordan_wigner(w)
    (-0.25j*(PauliY(wires=[0]) @ PauliX(wires=[1]))) + ((0.25+0j)*(PauliY(wires=[0]) @ PauliY(wires=[1]))) +
    ((0.25+0j)*(PauliX(wires=[0]) @ PauliX(wires=[1]))) + (0.25j*(PauliX(wires=[0]) @ PauliY(wires=[1])))

    >>> jordan_wigner(w, ps=True)
    -0.25j * Y(0) @ X(1)
    + (0.25+0j) * Y(0) @ Y(1)
    + (0.25+0j) * X(0) @ X(1)
    + 0.25j * X(0) @ Y(1)

    >>> jordan_wigner(w, ps=True, wire_map={0: 2, 1: 3})
    -0.25j * Y(2) @ X(3)
    + (0.25+0j) * Y(2) @ Y(3)
    + (0.25+0j) * X(2) @ X(3)
    + 0.25j * X(2) @ Y(3)
    """
    raise ValueError(f"fermi_operator must be a FermiWord or FermiSentence, got: {fermi_operator}")


@jordan_wigner.register
def _(fermi_operator: FermiWord, ps=False, wire_map=None):
    wires = list(fermi_operator.wires) or [0]
    identity_wire = wires[0]

    if len(fermi_operator) == 0:
        qubit_operator = PauliSentence({PauliWord({}): 1.0})

    else:
        coeffs = {"+": -0.5j, "-": 0.5j}
        qubit_operator = PauliSentence()

        for item in fermi_operator.items():
            (_, wire), sign = item

            z_string = dict(zip(range(wire), ["Z"] * wire))
            qubit_operator *= PauliSentence(
                {
                    PauliWord({**z_string, **{wire: "X"}}): 0.5,
                    PauliWord({**z_string, **{wire: "Y"}}): coeffs[sign],
                }
            )

    if not ps:
        # wire_order specifies wires to use for Identity (PauliWord({}))
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


@jordan_wigner.register
def _(fermi_operator: FermiSentence, ps=False, wire_map=None):
    wires = list(fermi_operator.wires) or [0]
    identity_wire = wires[0]

    if len(fermi_operator) == 0:
        qubit_operator = PauliSentence({PauliWord({}): 0})  # does anything break if I remove this?

    else:
        qubit_operator = PauliSentence()

        for fw, coeff in fermi_operator.items():
            fermi_word_as_ps = jordan_wigner(fw, ps=True)

            for pw in fermi_word_as_ps:
                qubit_operator[pw] += fermi_word_as_ps[pw] * coeff

    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator
