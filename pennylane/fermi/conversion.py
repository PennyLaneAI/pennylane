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

from functools import singledispatch
from typing import Union

import pennylane as qml
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord

from .fermionic import FermiSentence, FermiWord


# pylint: disable=unexpected-keyword-arg
def jordan_wigner(
    fermi_operator: (Union[FermiWord, FermiSentence]),
    ps: bool = False,
    wire_map: dict = None,
    tol: float = None,
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
        tol (float): tolerance for discarding the imaginary part of the coefficients

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
    return _jordan_wigner_dispatch(fermi_operator, ps, wire_map, tol)


@singledispatch
def _jordan_wigner_dispatch(fermi_operator, ps, wire_map, tol):
    """Dispatches to appropriate function if fermi_operator is a FermiWord or FermiSentence."""
    raise ValueError(f"fermi_operator must be a FermiWord or FermiSentence, got: {fermi_operator}")


@_jordan_wigner_dispatch.register
def _(fermi_operator: FermiWord, ps=False, wire_map=None, tol=None):
    wires = list(fermi_operator.wires) or [0]
    identity_wire = wires[0]

    if len(fermi_operator) == 0:
        qubit_operator = PauliSentence({PauliWord({}): 1.0})

    else:
        coeffs = {"+": -0.5j, "-": 0.5j}
        qubit_operator = PauliSentence({PauliWord({}): 1.0})  # Identity PS to multiply PSs with

        for item in fermi_operator.items():
            (_, wire), sign = item

            z_string = dict(zip(range(wire), ["Z"] * wire))
            qubit_operator *= PauliSentence(
                {
                    PauliWord({**z_string, **{wire: "X"}}): 0.5,
                    PauliWord({**z_string, **{wire: "Y"}}): coeffs[sign],
                }
            )

    for pw in qubit_operator:
        if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
            qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    if not ps:
        # wire_order specifies wires to use for Identity (PauliWord({}))
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


@_jordan_wigner_dispatch.register
def _(fermi_operator: FermiSentence, ps=False, wire_map=None, tol=None):
    wires = list(fermi_operator.wires) or [0]
    identity_wire = wires[0]

    qubit_operator = PauliSentence()  # Empty PS as 0 operator to add Pws to

    for fw, coeff in fermi_operator.items():
        fermi_word_as_ps = jordan_wigner(fw, ps=True)

        for pw in fermi_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + fermi_word_as_ps[pw] * coeff

            if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
                qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator

def parity_transform(
    fermi_operator: (Union[FermiWord, FermiSentence]),
        n_qubits: int,
    ps: bool = False,
    wire_map: dict = None,
    tol: float = None,
) -> Union[Operator, PauliSentence]:
    r"""Convert a fermionic operator to a qubit operator using the parity mapping.
    In parity mapping, qubit j stores the parity of all j-1 qubits before it whereas in Jordan-Wigner, qubit j stores the occupation number of qubit j.

    The fermionic creation and annihilation operators are mapped to the Pauli operators as

    .. math::
        a^{\dagger}_0 =  \left (\frac{X_0 - iY_0}{2}  \right )\otimes X_1 \otimes X_2 \otimes ... X_N, \:\: \text{...,} \:\:

        a^{\dagger}_n = \left (\frac{Z_{n-1} \otimes X_n - iY_n}{2} \right ) \otimes X_{n+1} \otimes X_{n+2} \otimes ... \otimes X_N

    and

    .. math::

        a_0 =  \left (\frac{X_0 + iY_0}{2}  \right )\otimes X_1 \otimes X_2 \otimes ... X_N, \:\: \text{...,} \:\:

        a_n = \left (\frac{Z_{n-1} \otimes X_n - iY_n}{2} \right ) \otimes X_{n+1} \otimes X_{n+2} \otimes ... \otimes X_N

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators and N is the number of qubits/spin orbitals.

    Args:
        fermi_operator(FermiWord, FermiSentence): the fermionic operator
        ps (bool): whether to return the result as a PauliSentence instead of an
            Operator. Defaults to False.
        wire_map (dict): a dictionary defining how to map the oribitals of
            the Fermi operator to qubit wires. If None, the integers used to
            order the orbitals will be used as wire labels. Defaults to None.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: a linear combination of qubit operators

    **Example**

    >>> w = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> parity_transform(w, n_qubits=6)
    (-0.25j*(PauliY(wires=[0]))) + ((-0.25+0j)*(PauliX(wires=[0]) @ PauliZ(wires=[1]))) +
    ((0.25+0j)*(PauliX(wires=[0]))) + (0.25j*(PauliY(wires=[0]) @ PauliZ(wires=[1])))

    >>> parity_transform(w, n_qubits=6, ps=True)
    -0.25j * Y(0)
    + (-0.25+0j) * X(0) @ Z(1)
    + (0.25+0j) * X(0)
    + 0.25j * Y(0) @ Z(1)
    
    >>> parity_transform(w, n_qubits=6, ps=True, wire_map={0: 2, 1: 3})
    -0.25j * Y(2)
    + (-0.25+0j) * X(2) @ Z(3)
    + (0.25+0j) * X(2)
    + 0.25j * Y(2) @ Z(3)
    """
    return _parity_transform_dispatch(fermi_operator, n_qubits, ps, wire_map, tol)


@singledispatch
def _parity_transform_dispatch(fermi_operator, n_qubits, ps, wire_map, tol):
    """Dispatches to appropriate function if fermi_operator is a FermiWord or FermiSentence."""
    raise ValueError(f"fermi_operator must be a FermiWord or FermiSentence, got: {fermi_operator}")


@_parity_transform_dispatch.register
def _(fermi_operator: FermiWord, n_qubits, ps=False, wire_map=None, tol=None):
    wires = list(fermi_operator.wires) or [0]
    identity_wire = wires[0]

    if len(fermi_operator) == 0:
        qubit_operator = PauliSentence({PauliWord({}): 1.0})

    else:
        coeffs = {"+": -0.5j, "-": 0.5j}
        qubit_operator = PauliSentence({PauliWord({}): 1.0})  # Identity PS to multiply PSs with

        for item in fermi_operator.items():
            (_, wire), sign = item
            if wire >= n_qubits:
                raise ValueError(f"Creating/annihilating a particle  on qubit number {wire} for a {n_qubits} qubit system")                
            if wire==0:
                x_string = dict(zip(range(wire+1, n_qubits), ["X"] * (n_qubits-wire)))
                qubit_operator *= PauliSentence(
                    {
                        PauliWord({**{wire: "X"}, **x_string}): 0.5,
                        PauliWord({**{wire: "Y"}, **x_string}): coeffs[sign],
                    }
                )
            else:
                x_string = dict(zip(range(wire+1, n_qubits), ["X"] * (n_qubits-wire)))
                qubit_operator *= PauliSentence(
                    {
                        PauliWord({**{wire-1:"Z"}, **{wire: "X"}, **x_string}): 0.5,
                        PauliWord({**{wire: "Y"}, **x_string}): coeffs[sign],
                    }
                )

    for pw in qubit_operator:
        if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
            qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    if not ps:
        # wire_order specifies wires to use for Identity (PauliWord({}))
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


@_parity_transform_dispatch.register
def _(fermi_operator: FermiSentence, n_qubits, ps=False, wire_map=None, tol=None):
    wires = list(fermi_operator.wires) or [0]
    identity_wire = wires[0]

    qubit_operator = PauliSentence()  # Empty PS as 0 operator to add Pws to

    for fw, coeff in fermi_operator.items():
        fermi_word_as_ps = parity_transform(fw, n_qubits, ps=True)

        for pw in fermi_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + fermi_word_as_ps[pw] * coeff

            if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
                qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator
