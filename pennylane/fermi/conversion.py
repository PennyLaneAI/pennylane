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
"""Functions to convert a fermionic operator to the qubit basis."""

import warnings
from functools import singledispatch
from typing import Union
import numpy as np

import pennylane as qml
from pennylane.operation import Operator, Tensor
from pennylane.pauli import PauliWord, PauliSentence
from pennylane.pauli.utils import _get_pauli_map, _pauli_mult
from .fermionic import FermiWord, FermiSentence


# pylint: disable=unexpected-keyword-arg
def jordan_wigner(
    fermi_operator: (Union[FermiWord, FermiSentence]), **kwargs
) -> Union[Operator, PauliSentence]:
    r"""Convert a fermionic operator to a qubit operator using the Jordan-Wigner mapping.

    The fermionic creation and annihilation operators are mapped to the Pauli operators as

    .. math::

        a^{\dagger}_0 =  \left (\frac{X_0 - iY_0}{2}  \right ), \:\: \text{...,} \:\:
        a^{\dagger}_n = Z_0 \otimes Z_1 \otimes ... \otimes Z_{n-1} \otimes \left (\frac{X_n - iY_n}{2} \right ),

    and

    .. math::

        a_0 =  \left (\frac{X_0 + iY_0}{2}  \right ), \:\: \text{...,} \:\:
        a_n = Z_0 \otimes Z_1 \otimes ... \otimes Z_{n-1} \otimes \left (\frac{X_n + iY_n}{2}  \right ),

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
    return _jordan_wigner_dispatch(fermi_operator, **kwargs)


@singledispatch
def _jordan_wigner_dispatch(fermi_operator, **kwargs):
    """Dispatches to appropriate function if fermi_operator is a FermiWord,
    FermiSentence or list"""
    raise ValueError(f"fermi_operator must be a FermiWord or FermiSentence, got: {fermi_operator}")


@_jordan_wigner_dispatch.register
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


@_jordan_wigner_dispatch.register
def _(fermi_operator: FermiSentence, ps=False, wire_map=None):
    wires = list(fermi_operator.wires) or [0]
    identity_wire = wires[0]

    qubit_operator = PauliSentence()

    for fw, coeff in fermi_operator.items():
        fermi_word_as_ps = jordan_wigner(fw, ps=True)

        for pw in fermi_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + fermi_word_as_ps[pw] * coeff

    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


@_jordan_wigner_dispatch.register
def _jordan_wigner_legacy(op: list, notation="physicist"):  # pylint:disable=too-many-branches
    r"""Convert a fermionic operator to a qubit operator using the Jordan-Wigner mapping.

    For instance, the one-body fermionic operator :math:`a_2^\dagger a_0` should be constructed as
    [2, 0]. The two-body operator :math:`a_4^\dagger a_3^\dagger a_2 a_1` should be constructed
    as [4, 3, 2, 1] with ``notation='physicist'``. If ``notation`` is set to ``'chemist'``, the
    two-body operator [4, 3, 2, 1] is constructed as :math:`a_4^\dagger a_3 a_2^\dagger a_1`.

    Args:
        op (list[int]): the fermionic operator
        notation (str): notation specifying the order of the two-body fermionic operators

    Returns:
        tuple(list[complex], list[Operation]): list of coefficients and qubit operators

    **Example**

    >>> f  = [0, 0]
    >>> q = jordan_wigner(f)
    >>> q # corresponds to :math:`\frac{1}{2}(I_0 - Z_0)`
    ([(0.5+0j), (-0.5+0j)], [Identity(wires=[0]), PauliZ(wires=[0])])
    """

    warnings.warn(
        "List input for the jordan_wigner function is deprecated; please use the fermionic operators format. For "
        "details, see the Fermionic Operators tutorial: https://pennylane.ai/qml/demos/tutorial_fermionic_operators"
    )

    if len(op) == 1:
        op = [(op[0], 1)]

    if len(op) == 2:
        op = [(op[0], 1), (op[1], 0)]

    if len(op) == 4:
        if notation == "physicist":
            if op[0] == op[1] or op[2] == op[3]:
                return [0], [qml.Identity(wires=[min(op)])]
            op = [(op[0], 1), (op[1], 1), (op[2], 0), (op[3], 0)]
        elif notation == "chemist":
            if (op[0] == op[2] or op[1] == op[3]) and op[1] != op[2]:
                return [0], [qml.Identity(wires=[min(op)])]
            op = [(op[0], 1), (op[1], 0), (op[2], 1), (op[3], 0)]
        else:
            raise ValueError(
                f"Currently, the only supported notations for the two-body terms are 'physicist'"
                f" and 'chemist', got notation = '{notation}'."
            )

    q = [[(0, "I"), 1.0]]
    for l in op:
        z = [(index, "Z") for index in range(l[0])]
        x = z + [(l[0], "X"), 0.5]
        if l[1]:
            y = z + [(l[0], "Y"), -0.5j]
        else:
            y = z + [(l[0], "Y"), 0.5j]

        m = []
        for t1 in q:
            for t2 in [x, y]:
                q1, c1 = _pauli_mult(t1[:-1], t2[:-1])
                m.append(q1 + [c1 * t1[-1] * t2[-1]])
        q = m

    c = [p[-1] for p in q]
    o = [p[:-1] for p in q]

    for item in o:
        k = [i for i, x in enumerate(o) if x == item]
        if len(k) >= 2:
            for j in k[::-1][:-1]:
                del o[j]
                c[k[0]] = c[k[0]] + c[j]
                del c[j]

    # Pauli gates objects pregenerated for speed
    pauli_map = _get_pauli_map(np.max(op))
    for i, term in enumerate(o):
        if len(term) == 0:
            # moved function from qchem.observable_hf without any changes to tests
            # codecov complained this line is not covered
            # function to be deprecated next release
            # not going to write a test to cover this line
            o[i] = qml.Identity(0)  # pragma: no cover
        else:
            k = [pauli_map[t[0]][t[1]] for t in term]
            o[i] = Tensor(*k)

    return c, o
