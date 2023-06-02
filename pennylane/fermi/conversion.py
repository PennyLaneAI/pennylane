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

from pennylane.pauli import PauliWord, PauliSentence
from .fermionic import FermiWord


def jw_mapping(fermi_operator: FermiWord) -> PauliSentence:
    r"""Convert a fermionic operator to a qubit operator using the Jordan-Wigner mapping.

    The fermionic creation and annihilation operators are mapped to the Pauli operators as

    .. math::

        c^{\dagger}_N = Z_0 \otimes  ... \otimes Z_{N-1} \otimes \left ( \frac{X-iY}{2} \right ),
        c_N = Z_0 \otimes  ... \otimes Z_{N-1} \otimes \left ( \frac{X+iY}{2} \right ),

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators.

    Args:
        fermi_operator(FermiWord): the fermionic operator

    Returns:
        PauliSentence: a linear combination of qubit operators

    **Example**

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> mapping(w)
    -0.25j * Y(0) @ X(1)
    + (0.25+0j) * Y(0) @ Y(1)
    + (0.25+0j) * X(0) @ X(1)
    + 0.25j * X(0) @ Y(1)
    """
    coeffs = {"+": -0.5j, "-": 0.5j}
    qubit_operator = PauliSentence()

    if len(fermi_operator) == 0:
        return PauliSentence({PauliWord({0: "I"}): 1.0 + 0.0j})

    for item in fermi_operator.items():
        (_, wire), sign = item

        z_string = dict(zip(range(wire), ["Z"] * wire))
        qubit_operator *= PauliSentence(
            {
                PauliWord({**z_string, **{wire: "X"}}): 0.5,
                PauliWord({**z_string, **{wire: "Y"}}): coeffs[sign],
            }
        )

    return qubit_operator
