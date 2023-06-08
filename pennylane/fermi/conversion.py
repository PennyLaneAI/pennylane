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

from pennylane.pauli import PauliWord, PauliSentence
from .fermionic import FermiWord, FermiSentence


@singledispatch
def jordan_wigner(fermi_operator: (Union[FermiWord, FermiSentence]), ps=False) -> PauliSentence:
    r"""Convert a fermionic operator to a qubit operator using the Jordan-Wigner mapping.

    The fermionic creation and annihilation operators are mapped to the Pauli operators as

    .. math::

        a^{\dagger}_N = Z_0 \otimes  ... \otimes Z_{N-1} \otimes \left ( \frac{X-iY}{2} \right ),
        a_N = Z_0 \otimes  ... \otimes Z_{N-1} \otimes \left ( \frac{X+iY}{2} \right ),

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators.

    Args:
        fermi_operator(FermiWord, FermiSentence): the fermionic operator
        ps: whether to return the result as a PauliSentence instead of an
            operation. Defaults to False.

    Returns:
        PauliSentence: a linear combination of qubit operators

    **Example**

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> jordan_wigner(w)
    -0.25j * Y(0) @ X(1)
    + (0.25+0j) * Y(0) @ Y(1)
    + (0.25+0j) * X(0) @ X(1)
    + 0.25j * X(0) @ Y(1)
    """
    raise ValueError(f"fermi_operator must be a FermiWord or FermiSentence, got: {fermi_operator}")


@jordan_wigner.register
def _(fermi_operator: FermiWord, ps=False):

    if len(fermi_operator) == 0:
        return PauliSentence({PauliWord({}): 1.0})

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

    if ps:
        return qubit_operator
    return qubit_operator.operation()


@jordan_wigner.register
def _(fermi_operator: FermiSentence, ps=False):

    if len(fermi_operator) == 0:
        return PauliSentence({PauliWord({}): 0})

    qubit_operator = PauliSentence()

    for fw, coeff in fermi_operator.items():
        fermi_word_as_ps = jordan_wigner(fw, ps=True)

        for key in fermi_word_as_ps.keys():
            qubit_operator[key] += fermi_word_as_ps[key] * coeff

    if ps:
        return qubit_operator
    return qubit_operator.operation()
