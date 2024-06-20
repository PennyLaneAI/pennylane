# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This module contains the functions for converting between OpenFermion and PennyLane objects.
"""

from functools import singledispatch
from typing import Union

# pylint: disable= import-outside-toplevel,no-member,unused-import
import pennylane as qml
from pennylane import numpy as np
from pennylane.fermi.fermionic import FermiSentence, FermiWord
from pennylane.ops import LinearCombination, Sum
from pennylane.qchem.convert import (
    _openfermion_to_pennylane,
    _pennylane_to_openfermion,
    _process_wires,
)
from pennylane.wires import Wires


def _import_of():
    """Import openfermion."""
    try:
        # pylint: disable=import-outside-toplevel, unused-import, multiple-imports
        import openfermion
    except ImportError as Error:
        raise ImportError(
            "This feature requires openfermion. "
            "It can be installed with: pip install openfermion."
        ) from Error

    return openfermion


def _from_openfermion_qubit(of_op, tol=1.0e-16, **kwargs):
    r"""Convert OpenFermion ``QubitOperator`` to a :class:`~.LinearCombination` object in PennyLane representing a linear combination of qubit operators.

    Args:
        of_op (QubitOperator): fermionic-to-qubit transformed operator in terms of
            Pauli matrices
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable terms measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).
        tol (float): tolerance value to decide whether the imaginary part of the coefficients is retained
        return_sum (bool): flag indicating whether a ``Sum`` object is returned

    Returns:
        (pennylane.ops.Sum, pennylane.ops.LinearCombination): a linear combination of Pauli words

    **Example**

    >>> q_op = QubitOperator('X0', 1.2) + QubitOperator('Z1', 2.4)
    >>> from_openfermion_qubit(q_op)
    1.2 * X(0) + 2.4 * Z(1)

    >>> from openfermion import FermionOperator
    >>> of_op = 0.5 * FermionOperator('0^ 2') + FermionOperator('0 2^')
    >>> pl_op = from_openfermion_qubit(of_op)
    >>> print(pl_op)
        0.5 * a⁺(0) a(2)
        + 1.0 * a(0) a⁺(2)
    """
    coeffs, pl_ops = _openfermion_to_pennylane(of_op, tol=tol)
    pl_term = qml.ops.LinearCombination(coeffs, pl_ops)

    if "format" in kwargs:
        if kwargs["format"] == "Sum":
            return qml.dot(*pl_term.terms())
        if kwargs["format"] != "LinearCombination":
            f = kwargs["format"]
            raise ValueError(f"format must be a Sum or LinearCombination, got: {f}.")

    return pl_term


def from_openfermion(openfermion_op, tol=1e-16):
    r"""Convert OpenFermion
    `FermionOperator <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`__
    object to PennyLane :class:`~.fermi.FermiWord` or :class:`~.fermi.FermiSentence` objects.

    Args:
        openfermion_op (FermionOperator): OpenFermion fermionic operator
        tol (float): tolerance for discarding negligible coefficients

    Returns:
        Union[FermiWord, FermiSentence]: the fermionic operator object

    **Example**

    >>> from openfermion import FermionOperator
    >>> openfermion_op = 0.5 * FermionOperator('0^ 2') + FermionOperator('0 2^')
    >>> pl_op = from_openfermion(openfermion_op)
    >>> print(pl_op)
        0.5 * a⁺(0) a(2)
        + 1.0 * a(0) a⁺(2)
    """
    try:
        import openfermion
    except ImportError as Error:
        raise ImportError(
            "This feature requires openfermion. "
            "It can be installed with: pip install openfermion"
        ) from Error

    typemap = {0: "-", 1: "+"}

    fermi_words = []
    fermi_coeffs = []

    for ops, val in openfermion_op.terms.items():
        fw_dict = {(i, op[0]): typemap[op[1]] for i, op in enumerate(ops)}
        fermi_words.append(FermiWord(fw_dict))
        fermi_coeffs.append(val)

    if len(fermi_words) == 1 and fermi_coeffs[0] == 1.0:
        return fermi_words[0]

    pl_op = FermiSentence(dict(zip(fermi_words, fermi_coeffs)))
    pl_op.simplify(tol=tol)

    return pl_op


def to_openfermion(
    pennylane_op: Union[Sum, LinearCombination, FermiWord, FermiSentence], wires=None, tol=1.0e-16
):
    r"""Convert a PennyLane operator to a OpenFermion ``QubitOperator`` or ``FermionOperator``.

    Args:
        pennylane_op (~ops.op_math.Sum, ~ops.op_math.LinearCombination, FermiWord, FermiSentence):
            linear combination of operators
        wires (Wires, list, tuple, dict):
            Custom wire mapping used to convert the qubit operator
            to an observable terms measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        (QubitOperator, FermionOperator): an OpenFermion operator

    **Example**

    >>> w1 = qml.fermi.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = qml.fermi.FermiWord({(0, 1) : '+', (1, 2) : '-'})
    >>> s = qml.fermi.FermiSentence({w1 : 1.2, w2: 3.1})
    >>> of_op = qml.to_openfermion(s)
    >>> of_op
    1.2 [0^ 1] +
    3.1 [1^ 2]
    """
    return _to_openfermion_dispatch(pennylane_op, wires=wires, tol=tol)


@singledispatch
def _to_openfermion_dispatch(pl_op, wires=None, tol=1.0e-16):
    """Dispatches to appropriate function if pl_op is a ``Sum``, ``LinearCombination, ``FermiWord`` or ``FermiSentence``."""
    raise ValueError(
        f"pl_op must be a Sum, LinearCombination, FermiWord or FermiSentence, got: {type(pl_op)}."
    )


@_to_openfermion_dispatch.register
def _(pl_op: Sum, wires=None, tol=1.0e-16):
    coeffs, ops = pl_op.terms()
    return _pennylane_to_openfermion(np.array(coeffs), ops, wires=wires, tol=tol)


# pylint: disable=unused-argument, protected-access
@_to_openfermion_dispatch.register
def _(ops: FermiWord, wires=None, tol=1.0e-16):
    openfermion = _import_of()

    if wires:
        all_wires = Wires.all_wires(ops.wires, sort=True)
        mapped_wires = _process_wires(wires)
        if not set(all_wires).issubset(set(mapped_wires)):
            raise ValueError("Supplied `wires` does not cover all wires defined in `ops`.")

        pl_op_mapped = {}
        for loc, orbital in ops.keys():
            pl_op_mapped[(loc, mapped_wires.index(orbital))] = ops[(loc, orbital)]

        ops = FermiWord(pl_op_mapped)

    return openfermion.ops.FermionOperator(qml.fermi.fermionic._to_string(ops, of=True))


@_to_openfermion_dispatch.register
def _(pl_op: FermiSentence, wires=None, tol=1.0e-16):
    openfermion = _import_of()

    fermion_op = openfermion.ops.FermionOperator()
    for fermi_word in pl_op:
        if np.abs(pl_op[fermi_word].imag) < tol:
            fermion_op += pl_op[fermi_word].real * to_openfermion(fermi_word, wires=wires)
        else:
            fermion_op += pl_op[fermi_word] * to_openfermion(fermi_word, wires=wires)

    return fermion_op
