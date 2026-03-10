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

import pennylane as qml
from pennylane import numpy as np
from pennylane.fermi.fermionic import FermiSentence, FermiWord
from pennylane.ops import LinearCombination, Sum
from pennylane.qchem.convert import _openfermion_to_pennylane, _pennylane_to_openfermion


def _import_of():
    """Import openfermion."""
    try:
        # pylint: disable=import-outside-toplevel
        import openfermion
    except ImportError as Error:
        raise ImportError(
            "This feature requires openfermion. "
            "It can be installed with: pip install openfermion."
        ) from Error

    return openfermion


def from_openfermion(openfermion_op, wires=None, tol=1e-16):
    r"""Convert OpenFermion
    `FermionOperator <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`__
    to PennyLane :class:`~.fermi.FermiWord` or :class:`~.fermi.FermiSentence` and
    OpenFermion `QubitOperator <https://quantumai.google/reference/python/openfermion/ops/QubitOperator>`__
    to PennyLane :class:`~.LinearCombination`.

    Args:
        openfermion_op (FermionOperator, QubitOperator): OpenFermion operator.
        wires (dict): Custom wire mapping used to convert the external qubit
            operator to a PennyLane operator.
            Only dictionaries with integer keys (for qubit-to-wire conversion) are accepted.
            If ``None``, the identity map (e.g., ``0->0, 1->1, ...``) will be used.
        tol (float): Tolerance for discarding negligible coefficients.

    Returns:
        Union[~.FermiWord, ~.FermiSentence, LinearCombination]: PennyLane operator.

    **Example**

    >>> import pennylane as qml
    >>> from openfermion import FermionOperator, QubitOperator
    >>> of_op = 0.5 * FermionOperator('0^ 2') + FermionOperator('0 2^')
    >>> pl_op = qml.from_openfermion(of_op)
    >>> print(pl_op)
    0.5 * a⁺(0) a(2)
    + 1.0 * a(0) a⁺(2)

    >>> of_op = QubitOperator('X0', 1.2) + QubitOperator('Z1', 2.4)
    >>> pl_op = qml.from_openfermion(of_op)
    >>> print(pl_op)
    1.2 * X(0) + 2.4 * Z(1)
    """
    openfermion = _import_of()

    if isinstance(openfermion_op, openfermion.FermionOperator):

        if wires:
            raise ValueError("Custom wire mapping is not supported for fermionic operators.")

        typemap = {0: "-", 1: "+"}

        fermi_words = []
        fermi_coeffs = []

        for ops, val in openfermion_op.terms.items():
            fw_dict = {(i, op[0]): typemap[op[1]] for i, op in enumerate(ops)}
            fermi_words.append(FermiWord(fw_dict))
            fermi_coeffs.append(val)

        if len(fermi_words) == 1 and fermi_coeffs[0] == 1.0:
            return fermi_words[0]

        pl_op = FermiSentence(dict(zip(fermi_words, fermi_coeffs, strict=True)))
        pl_op.simplify(tol=tol)

        return pl_op

    coeffs, pl_ops = _openfermion_to_pennylane(openfermion_op, wires=wires, tol=tol)

    pennylane_op = qml.ops.LinearCombination(coeffs, pl_ops)

    return pennylane_op


def to_openfermion(
    pennylane_op: Sum | LinearCombination | FermiWord | FermiSentence, wires=None, tol=1.0e-16
):
    r"""Convert a PennyLane operator to OpenFermion
    `QubitOperator <https://quantumai.google/reference/python/openfermion/ops/QubitOperator>`__ or
    `FermionOperator <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`__.

    Args:
        pennylane_op (~ops.op_math.Sum, ~ops.op_math.LinearCombination, ~.FermiWord, ~.FermiSentence):
            PennyLane operator
        wires (dict): Custom wire mapping used to convert a PennyLane qubit operator
            to the external operator.
            Only dictionaries with integer keys (for qubit-to-wire conversion) are accepted.
            If ``None``, the identity map (e.g., ``0->0, 1->1, ...``) will be used.

    Returns:
        (QubitOperator, FermionOperator): OpenFermion operator

    **Example**

    >>> import pennylane as qml
    >>> w1 = qml.FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = qml.FermiWord({(0, 1) : '+', (1, 2) : '-'})
    >>> fermi_s = qml.FermiSentence({w1 : 1.2, w2: 3.1})
    >>> of_fermi_op = qml.to_openfermion(fermi_s)
    >>> of_fermi_op
    1.2 [0^ 1] +
    3.1 [1^ 2]

    >>> sum_op = 1.2 * qml.X(0) + 2.4 * qml.Z(1)
    >>> of_qubit_op = qml.to_openfermion(sum_op)
    >>> of_qubit_op
    (1.2+0j) [X0] +
    (2.4+0j) [Z1]
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


@_to_openfermion_dispatch.register
def _(ops: FermiWord, wires=None, tol=1.0e-16):
    # pylint: disable=protected-access
    openfermion = _import_of()

    if wires:
        raise ValueError("Custom wire mapping is not supported for fermionic operators.")

    return openfermion.ops.FermionOperator(qml.fermi.fermionic._to_string(ops, of=True))


@_to_openfermion_dispatch.register
def _(pl_op: FermiSentence, wires=None, tol=1.0e-16):
    openfermion = _import_of()

    if wires:
        raise ValueError("Custom wire mapping is not supported for fermionic operators.")

    fermion_op = openfermion.ops.FermionOperator()
    for fermi_word in pl_op:
        if np.abs(pl_op[fermi_word].imag) < tol:
            fermion_op += pl_op[fermi_word].real * to_openfermion(fermi_word)
        else:
            fermion_op += pl_op[fermi_word] * to_openfermion(fermi_word)

    return fermion_op
