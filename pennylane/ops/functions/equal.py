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
This module contains the qp.equal function.
"""

# pylint: disable=too-many-arguments,too-many-return-statements,too-many-branches, too-many-positional-arguments

from collections.abc import Iterable
from functools import singledispatch
from typing import Any

import pennylane as qp
from pennylane import math
from pennylane.core.measurements import MeasurementProcess
from pennylane.core.operator import Operator, Operator2
from pennylane.measurements.classical_shadow import ShadowExpvalMP
from pennylane.measurements.counts import CountsMP
from pennylane.measurements.mutual_info import MutualInfoMP
from pennylane.measurements.vn_entropy import VnEntropyMP
from pennylane.ops import (
    Adjoint,
    CompositeOp,
    Conditional,
    Controlled,
    Exp,
    MeasurementValue,
    MidMeasure,
    Pow,
    SProd,
)
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.pulse.parametrized_evolution import ParametrizedEvolution
from pennylane.pytrees import flatten
from pennylane.tape import QuantumScript
from pennylane.templates import SubroutineOp
from pennylane.templates.subroutines import QSVT, ControlledSequence, PrepSelPrep, Select
from pennylane.typing import TensorLike

OPERANDS_MISMATCH_ERROR_MESSAGE = "op1 and op2 have different operands because "

BASE_OPERATION_MISMATCH_ERROR_MESSAGE = "op1 and op2 have different base operations because "


def equal(
    op1: Operator | MeasurementProcess | QuantumScript | PauliWord | PauliSentence,
    op2: Operator | MeasurementProcess | QuantumScript | PauliWord | PauliSentence,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
) -> bool:
    r"""Function for determining operator, measurement, and tape equality.

    .. Warning::

        The ``qp.equal`` function is based on a comparison of the types and attributes of the
        measurements or operators, not their mathematical representations. Mathematically equivalent
        operators defined via different classes may return False when compared via ``qp.equal``.
        To be more thorough would require the matrix forms to be calculated, which may drastically
        increase runtime.

    .. Warning::

        The interfaces and trainability of data within some observables including ``Prod`` and
        ``Sum`` are sometimes ignored, regardless of what the user specifies for ``check_interface``
        and ``check_trainability``.

    Args:
        op1 (.Operator or .MeasurementProcess or .QuantumTape or .PauliWord or .PauliSentence): First object to compare
        op2 (.Operator or .MeasurementProcess or .QuantumTape or .PauliWord or .PauliSentence): Second object to compare
        check_interface (bool, optional): Whether to compare interfaces. Default: ``True``.
        check_trainability (bool, optional): Whether to compare trainability status. Default: ``True``.
        rtol (float, optional): Relative tolerance for parameters.
        atol (float, optional): Absolute tolerance for parameters.

    Returns:
        bool: ``True`` if the operators, measurement processes, or tapes are equal, else ``False``

    **Example**

    Given two operators or measurement processes, ``qp.equal`` determines their equality.

    >>> op1 = qp.RX(np.array(.12), wires=0)
    >>> op2 = qp.RY(np.array(1.23), wires=0)
    >>> qp.equal(op1, op1), qp.equal(op1, op2)
    (True, False)

    >>> prod1 = qp.X(0) @ qp.Y(1)
    >>> prod2 = qp.Y(1) @ qp.X(0)
    >>> prod3 = qp.X(1) @ qp.Y(0)
    >>> qp.equal(prod1, prod2), qp.equal(prod1, prod3)
    (True, False)

    >>> H1 = qp.Hamiltonian([0.5, 0.5], [qp.Z(0) @ qp.Y(1), qp.Y(1) @ qp.Z(0) @ qp.Identity("a")])
    >>> H2 = qp.Hamiltonian([1], [qp.Z(0) @ qp.Y(1)])
    >>> H3 = qp.Hamiltonian([2], [qp.Z(0) @ qp.Y(1)])
    >>> qp.equal(H1, H2), qp.equal(H1, H3)
    (True, False)

    >>> qp.equal(qp.expval(qp.X(0)), qp.expval(qp.X(0)))
    True
    >>> qp.equal(qp.probs(wires=(0,1)), qp.probs(wires=(1,2)))
    False
    >>> qp.equal(qp.classical_shadow(wires=[0,1]), qp.classical_shadow(wires=[0,1]))
    True
    >>> tape1 = qp.tape.QuantumScript([qp.RX(1.2, wires=0)], [qp.expval(qp.Z(0))])
    >>> tape2 = qp.tape.QuantumScript([qp.RX(1.2 + 1e-6, wires=0)], [qp.expval(qp.Z(0))])
    >>> qp.equal(tape1, tape2, rtol=0, atol=1e-7)
    False
    >>> qp.equal(tape1, tape2, rtol=0, atol=1e-5)
    True

    .. details::
        :title: Usage Details

        You can use the optional arguments to get more specific results:

        >>> op1 = qp.RX(torch.tensor(1.2), wires=0)
        >>> op2 = qp.RX(jax.numpy.array(1.2), wires=0)
        >>> qp.equal(op1, op2)
        False

        >>> qp.equal(op1, op2, check_interface=False, check_trainability=False)
        True

        >>> op3 = qp.RX(pnp.array(1.2, requires_grad=True), wires=0)
        >>> op4 = qp.RX(pnp.array(1.2, requires_grad=False), wires=0)
        >>> qp.equal(op3, op4)
        False

        >>> qp.equal(op3, op4, check_trainability=False)
        True

        >>> qp.equal(Controlled(op3, control_wires=1), Controlled(op4, control_wires=1))
        False

        >>> qp.equal(Controlled(op3, control_wires=1), Controlled(op4, control_wires=1), check_trainability=False)
        True

    """

    dispatch_result = _equal(
        op1,
        op2,
        check_interface=check_interface,
        check_trainability=check_trainability,
        atol=atol,
        rtol=rtol,
    )
    if isinstance(dispatch_result, str):
        return False
    return dispatch_result


def assert_equal(
    op1: Operator | MeasurementProcess | QuantumScript,
    op2: Operator | MeasurementProcess | QuantumScript,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
) -> None:
    """Function to assert that two operators, measurements, or tapes are equal

    Args:
        op1 (.Operator or .MeasurementProcess or .QuantumTape): First object to compare
        op2 (.Operator or .MeasurementProcess or .QuantumTape): Second object to compare
        check_interface (bool, optional): Whether to compare interfaces. Default: ``True``.
        check_trainability (bool, optional): Whether to compare trainability status. Default: ``True``.
        rtol (float, optional): Relative tolerance for parameters.
        atol (float, optional): Absolute tolerance for parameters.

    Returns:
        None

    Raises:
        AssertionError: An ``AssertionError`` is raised if the two operators are not equal.

    .. seealso::

        :func:`~.equal`

    **Example**

    >>> op1 = qp.RX(np.array(0.12), wires=0)
    >>> op2 = qp.RX(np.array(1.23), wires=0)
    >>> qp.assert_equal(op1, op2)
    Traceback (most recent call last):
        ...
    AssertionError: op1 and op2 have different data. Got (array(0.12),) and (array(1.23),)

    >>> h1 = qp.Hamiltonian([1, 2], [qp.PauliX(0), qp.PauliY(1)])
    >>> h2 = qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliY(1)])
    >>> qp.assert_equal(h1, h2)
    Traceback (most recent call last):
        ...
    AssertionError: op1 and op2 have different operands because op1 and op2 have different scalars. Got 2 and 1

    """

    dispatch_result = _equal(
        op1,
        op2,
        check_interface=check_interface,
        check_trainability=check_trainability,
        atol=atol,
        rtol=rtol,
    )
    if isinstance(dispatch_result, str):
        raise AssertionError(dispatch_result)


def _equal(
    op1,
    op2,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
) -> bool | str:
    if not isinstance(op2, type(op1)):
        return f"op1 and op2 are of different types.  Got {type(op1)} and {type(op2)}."

    dispatch_result = _equal_dispatch(
        op1,
        op2,
        check_interface=check_interface,
        check_trainability=check_trainability,
        atol=atol,
        rtol=rtol,
    )
    if not dispatch_result:
        return f"{op1} and {op2} are not equal for an unspecified reason."
    return dispatch_result


@singledispatch
def _equal_dispatch(
    op1,
    op2,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
) -> bool | str:
    raise NotImplementedError(f"Comparison of {type(op1)} and {type(op2)} not implemented")


@_equal_dispatch.register
def _equal_circuit(
    op1: qp.tape.QuantumScript,
    op2: qp.tape.QuantumScript,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    # operations
    if len(op1.operations) != len(op2.operations):
        return False
    for comparands in zip(op1.operations, op2.operations, strict=True):
        if not qp.equal(
            comparands[0],
            comparands[1],
            check_interface=check_interface,
            check_trainability=check_trainability,
            rtol=rtol,
            atol=atol,
        ):
            return False
    # measurements
    if len(op1.measurements) != len(op2.measurements):
        return False
    for comparands in zip(op1.measurements, op2.measurements, strict=True):
        if not qp.equal(
            comparands[0],
            comparands[1],
            check_interface=check_interface,
            check_trainability=check_trainability,
            rtol=rtol,
            atol=atol,
        ):
            return False

    if op1.shots != op2.shots:
        return False
    if op1.trainable_params != op2.trainable_params:
        return False
    return True


@_equal_dispatch.register
def _equal_operators(
    op1: Operator,
    op2: Operator,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    """Default function to determine whether two Operator objects are equal."""

    if isinstance(op1, qp.Identity):
        # All Identities are equivalent, independent of wires.
        # We already know op1 and op2 are of the same type, so no need to check
        # that op2 is also an Identity
        return True

    if op1.arithmetic_depth != op2.arithmetic_depth:
        return f"op1 and op2 have different arithmetic depths. Got {op1.arithmetic_depth} and {op2.arithmetic_depth}"

    if op1.arithmetic_depth > 0:
        # Other dispatches cover cases of operations with arithmetic depth > 0.
        # If any new operations are added with arithmetic depth > 0, a new dispatch
        # should be created for them.
        return f"op1 and op2 have arithmetic depth > 0. Got arithmetic depth {op1.arithmetic_depth}"

    if op1.wires != op2.wires:
        return f"op1 and op2 have different wires. Got {op1.wires} and {op2.wires}."

    if op1.hyperparameters != op2.hyperparameters:
        return (
            "The hyperparameters are not equal for op1 and op2.\n"
            f"Got {op1.hyperparameters}\n and {op2.hyperparameters}."
        )

    if any(math.is_abstract(d) for d in op1.data + op2.data):
        # assume all tracers are independent
        return "Data contains a tracer. Abstract tracers are assumed to be unique."
    if len(op1.data) != len(op2.data):
        return f"op1 and op2 have different data.\nGot {op1.data} and {op2.data}"
    if not all(
        math.allclose(d1, d2, rtol=rtol, atol=atol)
        for d1, d2 in zip(op1.data, op2.data, strict=True)
    ):
        return f"op1 and op2 have different data.\nGot {op1.data} and {op2.data}"

    if check_trainability:
        for params1, params2 in zip(op1.data, op2.data, strict=True):
            params1_train = math.requires_grad(params1)
            params2_train = math.requires_grad(params2)
            if params1_train != params2_train:
                return (
                    "Parameters have different trainability.\n "
                    f"{params1} trainability is {params1_train} and {params2} trainability is {params2_train}"
                )

    if check_interface:
        for params1, params2 in zip(op1.data, op2.data, strict=True):
            params1_interface = math.get_interface(params1)
            params2_interface = math.get_interface(params2)
            if params1_interface != params2_interface:
                return (
                    "Parameters have different interfaces.\n "
                    f"{params1} interface is {params1_interface} and {params2} interface is {params2_interface}"
                )

    return True


@_equal_dispatch.register
def _equal_operator2(
    op1: Operator2,
    op2: Operator2,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    """Check equality between Operator2 instances."""
    if type(op1) is not type(op2):
        return f"op1 and op2 are of different types. Got {type(op1)} and {type(op2)}."

    # Check static arguments
    for (sname, sval1), (_, sval2) in zip(
        op1.static_args.items(), op2.static_args.items(), strict=True
    ):
        if sval1 != sval2:
            return f"op1 and op2 have different values for '{sname}'.\nGot {sval1} and {sval2}."

    # Check static compilable arguments
    for (cname, cval1), (_, cval2) in zip(
        op1.compilable_args.items(), op2.compilable_args.items(), strict=True
    ):
        if cval1 != cval2:
            return f"op1 and op2 have different values for '{cname}'.\nGot {cval1} and {cval2}."

    # Check wire arguments
    for (wname, wval1), (_, wval2) in zip(
        op1.wire_args.items(), op2.wire_args.items(), strict=True
    ):
        if wname in op1.hybrid_argnames:
            leaves1, tree1 = flatten(wval1, is_leaf=lambda w: isinstance(w, qp.wires.Wires))
            leaves2, tree2 = flatten(wval2, is_leaf=lambda w: isinstance(w, qp.wires.Wires))

            if len(leaves1) != len(leaves2) or tree1 != tree2:
                return f"op1 and op2 have different wires for '{wname}'.\nGot {wval1} and {wval2}."
            for l1, l2 in zip(leaves1, leaves2, strict=True):
                res = _check_wire_value(wname, l1, l2)
                if isinstance(res, str):
                    return res

        else:
            res = _check_wire_value(wname, wval1, wval2)
            if isinstance(res, str):
                return res

    # Check dynamic arguments
    # Expensive: array comparison via math.allclose.
    for (dname, dval1), (_, dval2) in zip(
        op1.dynamic_args.items(), op2.dynamic_args.items(), strict=True
    ):
        res = _check_dynamic_value(
            dname,
            dval1,
            dval2,
            check_interface=check_interface,
            check_trainability=check_trainability,
            rtol=rtol,
            atol=atol,
        )
        if isinstance(res, str):
            return res

    # Check hybrid arguments
    # Most expensive: pytree flatten + recursive _equal.
    for (hname, hval1), (_, hval2) in zip(
        op1.hybrid_args.items(), op2.hybrid_args.items(), strict=True
    ):
        if hname in op1.wire_argnames:
            continue
        res = _check_pytree_value(
            hname,
            hval1,
            hval2,
            check_interface=check_interface,
            check_trainability=check_trainability,
            rtol=rtol,
            atol=atol,
        )
        if isinstance(res, str):
            return res

    return True


def _check_wire_value(wname: str, wval1: Any, wval2: Any):
    """Check for equality of a wire argument of an Operator2 instance."""
    unequal_wires = False
    abstract_wires = False

    if len(wval1) != len(wval2):
        unequal_wires = True

    else:
        for w1, w2 in zip(wval1, wval2, strict=True):
            if math.is_abstract(w1) or math.is_abstract(w2):
                unequal_wires = True
                abstract_wires = True
                break
            if w1 != w2:
                unequal_wires = True
                break

    if unequal_wires:
        if abstract_wires:
            return (
                f"At least one of op1 or op2 has one or more tracer values for '{wname}'. "
                "Abstract tracers are assumed to be unique."
            )
        return f"op1 and op2 have different wires for '{wname}'.\nGot {wval1} and {wval2}."

    return True


def _check_dynamic_value(
    dname: str,
    dval1: TensorLike,
    dval2: TensorLike,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    """Check for equality of a dynamic argument of an Operator2 instance."""
    if math.is_abstract(dval1) or math.is_abstract(dval2):
        return (
            f"At least one of op1 or op2 has a tracer value for '{dname}'. Abstract "
            "tracers are assumed to be unique."
        )

    # Need to check shape along with math.allclose to handle the case where one
    # value is broadcasted. math.allclose might pass in that case.
    if math.shape(dval1) != math.shape(dval2) or not math.allclose(
        dval1, dval2, atol=atol, rtol=rtol
    ):
        return f"op1 and op2 have different values for '{dname}'.\nGot {dval1} and {dval2}."

    if check_interface:
        interface1 = math.get_interface(dval1)
        interface2 = math.get_interface(dval2)
        if interface1 != interface2:
            return (
                f"op1 and op2 have different interfaces for '{dname}'.\n"
                f"op1.{dname}'s interface is {interface1} and op2.{dname}'s "
                f"interface is {interface2}."
            )

    if check_trainability:
        grad1 = "trainable" if math.requires_grad(dval1) else "not trainable"
        grad2 = "trainable" if math.requires_grad(dval2) else "not trainable"
        if grad1 != grad2:
            return (
                f"op1 and op2 differ in trainability for '{dname}'.\n"
                f"op1.{dname} is {grad1} and op2.{dname} is {grad2}."
            )

    return True


def _check_pytree_value(
    hname: str,
    hval1: Any,
    hval2: Any,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    """Check for equality of a hybrid argument of an Operator2 instance."""
    unequal_str = f"op1 and op2 have different values for '{hname}'.\nGot {hval1} and {hval2}."
    leaves1, tree1 = flatten(hval1, is_leaf=lambda v: isinstance(v, Operator2))
    leaves2, tree2 = flatten(hval2, is_leaf=lambda v: isinstance(v, Operator2))
    if len(leaves1) != len(leaves2) or tree1 != tree2:
        return unequal_str

    for l1, l2 in zip(leaves1, leaves2, strict=True):
        # Case 1: Both leaf values are operators
        if isinstance(l1, Operator2) and isinstance(l2, Operator2):
            ops_equal = _equal(
                l1,
                l2,
                check_interface=check_interface,
                check_trainability=check_trainability,
                atol=atol,
                rtol=rtol,
            )
            if isinstance(ops_equal, str):
                return (
                    f"op1 and op2 have different values for '{hname}'. "
                    f"The operator arguments do not match:\n{ops_equal}"
                )

        # Case 2: One leaf value is an operator and the other is not. Not valid
        elif isinstance(l1, Operator2) or isinstance(l2, Operator2):
            return unequal_str

        # Case 3: Neither leaf value is an operator. l1 and l2 are assumed to be numbers or arrays
        else:
            res = _check_dynamic_value(
                hname,
                l1,
                l2,
                check_interface=check_interface,
                check_trainability=check_trainability,
                atol=atol,
                rtol=rtol,
            )
            if isinstance(res, str):
                return res

    return True


@_equal_dispatch.register
def _equal_pauliword(
    op1: PauliWord,
    op2: PauliWord,
    **kwargs,
):
    if op1 != op2:
        if set(op1) != set(op2):
            err = "Different wires in Pauli words."
            diff12 = set(op1).difference(set(op2))
            diff21 = set(op2).difference(set(op1))
            if diff12:
                err += f" op1 has {diff12} not present in op2."
            if diff21:
                err += f" op2 has {diff21} not present in op1."
            return err
        pauli_diff = {}
        for wire in op1:
            if op1[wire] != op2[wire]:
                pauli_diff[wire] = f"{op1[wire]} != {op2[wire]}"
        return f"Pauli words agree on wires but differ in Paulis: {pauli_diff}"
    return True


@_equal_dispatch.register
def _equal_paulisentence(
    op1: PauliSentence,
    op2: PauliSentence,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    if set(op1) != set(op2):
        err = "Different Pauli words in PauliSentences."
        diff12 = set(op1).difference(set(op2))
        diff21 = set(op2).difference(set(op1))
        if diff12:
            err += f" op1 has {diff12} not present in op2."
        if diff21:
            err += f" op2 has {diff21} not present in op1."
        return err
    for pw in op1:
        param1 = op1[pw]
        param2 = op2[pw]
        if check_trainability:
            param1_train = math.requires_grad(param1)
            param2_train = math.requires_grad(param2)
            if param1_train != param2_train:
                return (
                    "Parameters have different trainability.\n "
                    f"{param1} trainability is {param1_train} and {param2} trainability is {param2_train}"
                )

        if check_interface:
            param1_interface = math.get_interface(param1)
            param2_interface = math.get_interface(param2)
            if param1_interface != param2_interface:
                return (
                    "Parameters have different interfaces.\n "
                    f"{param1} interface is {param1_interface} and {param2} interface is {param2_interface}"
                )
        if not math.allclose(param1, param2, rtol=rtol, atol=atol):
            return f"The coefficients of the PauliSentences for {pw} differ: {param1}; {param2}"
    return True


@_equal_dispatch.register
# pylint: disable=protected-access
def _equal_prod_and_sum(op1: CompositeOp, op2: CompositeOp, **kwargs):
    """Determine whether two Prod or Sum objects are equal"""
    if op1.pauli_rep is not None and (op1.pauli_rep == op2.pauli_rep):  # shortcut check
        return True

    if len(op1.operands) != len(op2.operands):
        return f"op1 and op2 have different number of operands. Got {len(op1.operands)} and {len(op2.operands)}"

    # organizes by wire indicies while respecting commutation relations
    sorted_ops1 = op1._sort(op1.operands)
    sorted_ops2 = op2._sort(op2.operands)

    for o1, o2 in zip(sorted_ops1, sorted_ops2, strict=True):
        op_check = _equal(o1, o2, **kwargs)
        if isinstance(op_check, str):
            return OPERANDS_MISMATCH_ERROR_MESSAGE + op_check

    return True


@_equal_dispatch.register
def _equal_controlled(op1: Controlled, op2: Controlled, **kwargs):
    """Determine whether two Controlled or ControlledOp objects are equal"""
    if op1.arithmetic_depth != op2.arithmetic_depth:
        return f"op1 and op2 have different arithmetic depths. Got {op1.arithmetic_depth} and {op2.arithmetic_depth}"

    # op.base.wires compared in return
    if op1.work_wires != op2.work_wires:
        return f"op1 and op2 have different work wires. Got {op1.work_wires} and {op2.work_wires}"

    if op1.work_wire_type != op2.work_wire_type:
        return f"op1 and op2 have different work wire types. Got {op1.work_wire_type} and {op2.work_wire_type}"

    # work wires and control_wire/control_value combinations compared here
    op1_control_dict = dict(zip(op1.control_wires, op1.control_values, strict=True))
    op2_control_dict = dict(zip(op2.control_wires, op2.control_values, strict=True))
    if op1_control_dict != op2_control_dict:
        return f"op1 and op2 have different control dictionaries. Got {op1_control_dict} and {op2_control_dict}"

    base_equal_check = _equal(op1.base, op2.base, **kwargs)
    if isinstance(base_equal_check, str):
        return BASE_OPERATION_MISMATCH_ERROR_MESSAGE + base_equal_check

    return True


@_equal_dispatch.register
def _equal_controlled_sequence(op1: ControlledSequence, op2: ControlledSequence, **kwargs):
    """Determine whether two ControlledSequences are equal"""
    if op1.wires != op2.wires:
        return f"op1 and op2 have different wires. Got {op1.wires} and {op2.wires}."
    if op1.arithmetic_depth != op2.arithmetic_depth:
        return f"op1 and op2 have different arithmetic depths. Got {op1.arithmetic_depth} and {op2.arithmetic_depth}"

    base_equal_check = _equal(op1.base, op2.base, **kwargs)
    if isinstance(base_equal_check, str):
        return BASE_OPERATION_MISMATCH_ERROR_MESSAGE + base_equal_check

    return True


@_equal_dispatch.register
def _equal_pow(op1: Pow, op2: Pow, **kwargs):
    """Determine whether two Pow objects are equal"""
    check_interface, check_trainability = kwargs["check_interface"], kwargs["check_trainability"]

    if check_interface:
        interface1 = math.get_interface(op1.z)
        interface2 = math.get_interface(op2.z)
        if interface1 != interface2:
            return (
                "Exponent have different interfaces.\n"
                f"{op1.z} interface is {interface1} and {op2.z} interface is {interface2}"
            )
    if check_trainability:
        grad1 = math.requires_grad(op1.z)
        grad2 = math.requires_grad(op2.z)
        if grad1 != grad2:
            return (
                "Exponent have different trainability.\n"
                f"{op1.z} interface is {grad1} and {op2.z} interface is {grad2}"
            )

    if op1.z != op2.z:
        return f"Exponent are different. Got {op1.z} and {op2.z}"

    base_equal_check = _equal(op1.base, op2.base, **kwargs)
    if isinstance(base_equal_check, str):
        return BASE_OPERATION_MISMATCH_ERROR_MESSAGE + base_equal_check

    return True


@_equal_dispatch.register
def _equal_adjoint(op1: Adjoint, op2: Adjoint, **kwargs):
    """Determine whether two Adjoint objects are equal"""
    # first line of top-level equal function already confirms both are Adjoint - only need to compare bases
    base_equal_check = _equal(op1.base, op2.base, **kwargs)
    if isinstance(base_equal_check, str):
        return BASE_OPERATION_MISMATCH_ERROR_MESSAGE + base_equal_check

    return True


@_equal_dispatch.register
def _equal_conditional(op1: Conditional, op2: Conditional, **kwargs):
    """Determine whether two Conditional objects are equal"""
    # first line of top-level equal function already confirms both are Conditionaly - only need to compare bases and meas_val
    return qp.equal(op1.base, op2.base, **kwargs) and qp.equal(op1.meas_val, op2.meas_val, **kwargs)


@_equal_dispatch.register
def _equal_measurement_value(op1: MeasurementValue, op2: MeasurementValue, **kwargs):
    """Determine whether two MeasurementValue objects are equal"""
    return op1.measurements == op2.measurements


@_equal_dispatch.register
def _equal_exp(op1: Exp, op2: Exp, **kwargs):
    """Determine whether two Exp objects are equal"""
    check_interface, check_trainability, rtol, atol = (
        kwargs["check_interface"],
        kwargs["check_trainability"],
        kwargs["rtol"],
        kwargs["atol"],
    )

    if check_interface:
        coeff1_interface = math.get_interface(op1.coeff)
        coeff2_interface = math.get_interface(op2.coeff)
        if coeff1_interface != coeff2_interface:
            return (
                "Coeffs have different interfaces.\n "
                f"{op1.coeff} interface is {coeff1_interface} and {op2.coeff} interface is {coeff2_interface}"
            )

    if check_trainability:
        coeff1_train = math.requires_grad(op1.coeff)
        coeff2_train = math.requires_grad(op2.coeff)
        if coeff1_train != coeff2_train:
            return (
                "Coeffs have different trainability.\n "
                f"{op1.coeff} trainability is {coeff1_train} and {op2.coeff} trainability is {coeff2_train}"
            )

    if qp.math.is_abstract(op1.coeff) or qp.math.is_abstract(op2.coeff):
        if op1.coeff is not op2.coeff:
            return "Data contains a tracer. Abstract tracers are assumed to be unique."

    elif not qp.math.allclose(op1.coeff, op2.coeff, rtol=rtol, atol=atol):
        return f"op1 and op2 have different coefficients. Got {op1.coeff} and {op2.coeff}"

    equal_check = _equal(op1.base, op2.base, **kwargs)
    if isinstance(equal_check, str):
        return BASE_OPERATION_MISMATCH_ERROR_MESSAGE + equal_check

    return True


@_equal_dispatch.register
def _equal_sprod(op1: SProd, op2: SProd, **kwargs):
    """Determine whether two SProd objects are equal"""
    check_interface, check_trainability, rtol, atol = (
        kwargs["check_interface"],
        kwargs["check_trainability"],
        kwargs["rtol"],
        kwargs["atol"],
    )

    if check_interface:
        scalar1_interface = math.get_interface(op1.scalar)
        scalar2_interface = math.get_interface(op2.scalar)
        if scalar1_interface != scalar2_interface:
            return (
                "Scalars have different interfaces.\n "
                f"{op1.scalar} interface is {scalar1_interface} and {op2.scalar} interface is {scalar2_interface}"
            )

    if check_trainability:
        scalar1_train = math.requires_grad(op1.scalar)
        scalar2_train = math.requires_grad(op2.scalar)
        if scalar1_train != scalar2_train:
            return (
                "Scalars have different trainability.\n "
                f"{op1.scalar} trainability is {scalar1_train} and {op2.scalar} trainability is {scalar2_train}"
            )

    if qp.math.is_abstract(op1.scalar) or qp.math.is_abstract(op2.scalar):
        if op1.scalar is not op2.scalar:
            return "Data contains a tracer. Abstract tracers are assumed to be unique."

    elif op1.pauli_rep is not None and (op1.pauli_rep == op2.pauli_rep):  # shortcut check
        return True

    # allclose only works if op1.scalar and op2.scalar are not abstract
    elif not qp.math.allclose(op1.scalar, op2.scalar, rtol=rtol, atol=atol):
        return f"op1 and op2 have different scalars. Got {op1.scalar} and {op2.scalar}"

    equal_check = _equal(op1.base, op2.base, **kwargs)
    if isinstance(equal_check, str):
        return BASE_OPERATION_MISMATCH_ERROR_MESSAGE + equal_check

    return True


@_equal_dispatch.register
def _equal_parametrized_evolution(op1: ParametrizedEvolution, op2: ParametrizedEvolution, **kwargs):
    # check times match
    if op1.t is None or op2.t is None:
        if not (op1.t is None and op2.t is None):
            return False
    elif not math.allclose(op1.t, op2.t):
        return False

    # check parameters passed to operator match
    operator_check = _equal_operators(op1, op2, **kwargs)
    if isinstance(operator_check, str):
        return False

    # check H.coeffs match
    if len(op1.H.coeffs) != len(op2.H.coeffs) or len(op1.H.ops) != len(op2.H.ops):
        return False
    if not all(c1 == c2 for c1, c2 in zip(op1.H.coeffs, op2.H.coeffs, strict=True)):
        return False

    return all(equal(o1, o2, **kwargs) for o1, o2 in zip(op1.H.ops, op2.H.ops, strict=True))


@_equal_dispatch.register
def _equal_measurements(
    op1: MeasurementProcess,
    op2: MeasurementProcess,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    """Determine whether two MeasurementProcess objects are equal"""

    if op1.obs is not None and op2.obs is not None:
        return equal(
            op1.obs,
            op2.obs,
            check_interface=check_interface,
            check_trainability=check_trainability,
            rtol=rtol,
            atol=atol,
        )

    if op1.mv is not None and op2.mv is not None:
        if isinstance(op1.mv, MeasurementValue) and isinstance(op2.mv, MeasurementValue):
            return qp.equal(op1.mv, op2.mv)

        if math.is_abstract(op1.mv) or math.is_abstract(op2.mv):
            return op1.mv is op2.mv

        if isinstance(op1.mv, Iterable) and isinstance(op2.mv, Iterable):
            if len(op1.mv) == len(op2.mv):
                return all(
                    mv1.measurements == mv2.measurements
                    for mv1, mv2 in zip(op1.mv, op2.mv, strict=True)
                )

        return False

    if op1.wires != op2.wires:
        return False

    if op1.obs is None and op2.obs is None:
        # only compare eigvals if both observables are None.
        # Can be expensive to compute for large observables
        if op1.eigvals() is not None and op2.eigvals() is not None:
            return math.allclose(op1.eigvals(), op2.eigvals(), rtol=rtol, atol=atol)

        return op1.eigvals() is None and op2.eigvals() is None

    return False


@_equal_dispatch.register
def _equal_mid_measure(op1: MidMeasure, op2: MidMeasure, **_):
    return (
        op1.wires == op2.wires
        and op1.meas_uid == op2.meas_uid
        and op1.reset == op2.reset
        and op1.postselect == op2.postselect
    )


@_equal_dispatch.register
def _equal_pauli_measure(op1: PauliMeasure, op2: PauliMeasure, **_):
    if op1.wires != op2.wires:
        return "op1 and op2 have different wires."
    if op1.postselect != op2.postselect:
        return "op1 and op2 have different postselect values."
    if op1.pauli_word != op2.pauli_word:
        return f"op1 has pauli_word {op1.pauli_word} and op2 has pauli_word {op2.pauli_word}"
    if op1.meas_uid != op2.meas_uid:
        return "op1 and op2 have different identifiers id."
    return True


@_equal_dispatch.register
def _(op1: VnEntropyMP, op2: VnEntropyMP, **kwargs):
    """Determine whether two MeasurementProcess objects are equal"""
    eq_m = _equal_measurements(op1, op2, **kwargs)
    log_base_match = op1.log_base == op2.log_base
    return eq_m and log_base_match


@_equal_dispatch.register
def _(op1: MutualInfoMP, op2: MutualInfoMP, **kwargs):
    """Determine whether two MeasurementProcess objects are equal"""
    eq_m = _equal_measurements(op1, op2, **kwargs)
    log_base_match = op1.log_base == op2.log_base
    return eq_m and log_base_match


@_equal_dispatch.register
def _equal_shadow_measurements(op1: ShadowExpvalMP, op2: ShadowExpvalMP, **_):
    """Determine whether two ShadowExpvalMP objects are equal"""

    wires_match = op1.wires == op2.wires

    if isinstance(op1.H, Operator) and isinstance(op2.H, Operator):
        H_match = equal(op1.H, op2.H)
    elif isinstance(op1.H, Iterable) and isinstance(op2.H, Iterable):
        if len(op1.H) != len(op2.H):
            return False
        H_match = all(equal(o1, o2) for o1, o2 in zip(op1.H, op2.H, strict=True))
    else:
        return False

    k_match = op1.k == op2.k

    return wires_match and H_match and k_match


@_equal_dispatch.register
def _equal_counts(op1: CountsMP, op2: CountsMP, **kwargs):
    return _equal_measurements(op1, op2, **kwargs) and op1.all_outcomes == op2.all_outcomes


@_equal_dispatch.register
def _equal_subroutineop(
    op1: SubroutineOp,
    op2: SubroutineOp,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    if op1.subroutine != op2.subroutine:
        return f"op1 is {op1.subroutine} and op2 is {op2.subroutine}"
    for static_arg in op1.subroutine.static_argnames:
        if (val1 := op1.bound_args.arguments[static_arg]) != (
            val2 := op2.bound_args.arguments[static_arg]
        ):
            return f"op1 has value {val1} and op2 has value {val2} for input {static_arg}"
    for wire_arg in op1.subroutine.wire_argnames:
        if (val1 := op1.bound_args.arguments[wire_arg]) != (
            val2 := op2.bound_args.arguments[wire_arg]
        ):
            return f"op1 has value {val1} and op2 has value {val2} for register {wire_arg}"
    for dynamic_arg in op1.subroutine.dynamic_argnames:
        vals1, tree1 = flatten(op1.bound_args.arguments[dynamic_arg])
        vals2, tree2 = flatten(op2.bound_args.arguments[dynamic_arg])
        if len(vals1) != len(vals2):
            return f"op1 and op2 return a different number of leaves for {dynamic_arg}."
        if tree1 != tree2:
            return f"op1 and op2 have different pytree structures for {dynamic_arg}."
        for v1, v2 in zip(vals1, vals2, strict=True):
            if check_trainability and math.requires_grad(v1) != math.requires_grad(v2):
                return f"{dynamic_arg} has different trainability for {v1} and {v2}."
            if check_interface and math.get_interface(v1) != math.get_interface(v2):
                return f"{dynamic_arg} has different interfaces for {v1} and {v2}."
            if not math.allclose(v1, v2, atol=atol, rtol=rtol):
                return f"{dynamic_arg} has different values {v1} and {v2}."

    return True


@_equal_dispatch.register
def _equal_hilbert_schmidt(
    op1: qp.HilbertSchmidt,
    op2: qp.HilbertSchmidt,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    if len(op1.data) != len(op2.data):
        return False
    if not all(
        math.allclose(d1, d2, rtol=rtol, atol=atol)
        for d1, d2 in zip(op1.data, op2.data, strict=True)
    ):
        return False

    if check_trainability:
        for params_1, params_2 in zip(op1.data, op2.data, strict=True):
            if math.requires_grad(params_1) != math.requires_grad(params_2):
                return False

    if check_interface:
        for params_1, params_2 in zip(op1.data, op2.data, strict=True):
            if math.get_interface(params_1) != math.get_interface(params_2):
                return False

    equal_kwargs = {
        "check_interface": check_interface,
        "check_trainability": check_trainability,
        "atol": atol,
        "rtol": rtol,
    }

    U1 = qp.prod(*op1.hyperparameters["U"])
    U2 = qp.prod(*op2.hyperparameters["U"])
    if qp.equal(U1, U2, **equal_kwargs) is False:
        return False

    V1 = qp.prod(*op1.hyperparameters["V"])
    V2 = qp.prod(*op2.hyperparameters["V"])
    if qp.equal(V1, V2, **equal_kwargs) is False:
        return False

    return True


@_equal_dispatch.register
def _equal_prep_sel_prep(op1: PrepSelPrep, op2: PrepSelPrep, **kwargs):
    """Determine whether two PrepSelPrep are equal"""
    if op1.control != op2.control:
        return f"op1 and op2 have different control wires. Got {op1.control} and {op2.control}."
    if op1.wires != op2.wires:
        return f"op1 and op2 have different wires. Got {op1.wires} and {op2.wires}."
    if not qp.equal(op1.lcu, op2.lcu):
        return f"op1 and op2 have different lcu. Got {op1.lcu} and {op2.lcu}"
    return True


@_equal_dispatch.register
def _equal_qsvt(op1: QSVT, op2: QSVT, **kwargs):
    """Determine whether two QSVT are equal"""
    if not equal(UA1 := op1.hyperparameters["UA"], UA2 := op2.hyperparameters["UA"], **kwargs):
        return f"op1 and op2 have different block encodings UA. Got {UA1} and {UA2}."
    projectors1 = op1.hyperparameters["projectors"]
    projectors2 = op2.hyperparameters["projectors"]
    if len(projectors1) != len(projectors2):
        return f"op1 and op2 have a different number of projectors. Got {projectors1} and {projectors2}."
    for i, (p1, p2) in enumerate(zip(projectors1, projectors2, strict=True)):
        try:
            assert_equal(p1, p2, **kwargs)
        except AssertionError as e:
            return f"op1 and op2 have different projectors at position {i}. Got {p1} and {p2}, which differ: {e}."
    return True


@_equal_dispatch.register
def _equal_select(op1: Select, op2: Select, **kwargs):
    """Determine whether two Select are equal"""
    if op1.control != op2.control:
        return f"op1 and op2 have different control wires. Got {op1.control} and {op2.control}."
    t1 = op1.hyperparameters["ops"]
    t2 = op2.hyperparameters["ops"]
    if len(t1) != len(t2):
        return (
            f"op1 and op2 have different number of target operators. Got {len(t1)} and {len(t2)}."
        )
    for idx, (_t1, _t2) in enumerate(zip(t1, t2, strict=True)):
        comparer = _equal(_t1, _t2, **kwargs)
        if isinstance(comparer, str):
            return f"got different operations at index {idx}: {_t1} and {_t2}. They differ because {comparer}."
    return True
