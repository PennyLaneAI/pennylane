# Copyright 2026 Xanadu Quantum Technologies Inc.

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
This module contains utilities for testing decomposition rules.
"""

from collections import defaultdict
from functools import partial, singledispatch

import jax

from pennylane import capture, math
from pennylane.core import queuing
from pennylane.core.operator import Operator, Operator1, Operator2, abstractify
from pennylane.decomposition import DecompositionRule, list_decomps
from pennylane.decomposition.decomposition_rule import _decomp_contains_mcm
from pennylane.decomposition.utils import _get_decomp_args
from pennylane.ops import Projector
from pennylane.ops import functions as op_funcs
from pennylane.ops.op_math import Conditional
from pennylane.tape import QuantumScript
from pennylane.templates.core import SubroutineOp
from pennylane.transforms import resolve_dynamic_wires

from .capture import plxpr_to_tape


def _resolve_dynamic_wires(ops, num_zeroed):
    """Apply the transform resolve_dynamic_wires to a list of operations or tape."""

    if not isinstance(ops, QuantumScript):
        tape = QuantumScript(ops)
        return _resolve_dynamic_wires(tape, num_zeroed).operations

    zeroed = range(len(ops.wires), len(ops.wires) + num_zeroed)
    return resolve_dynamic_wires([ops], zeroed=zeroed)[0][0]


def _assert_counts_match(expected, actual):
    """Assert that the gate counts expected from a resource function match the actual gate
    counts, where gates missing from either side are counted as zero."""

    mismatches = sorted(
        (str(op), actual.get(op, 0), expected.get(op, 0))
        for op in expected.keys() | actual.keys()
        if actual.get(op, 0) != expected.get(op, 0)
    )

    if not mismatches:
        return

    width = max(len("Operator"), *(len(op) for op, _, _ in mismatches))
    header = [f"{'Operator'.rjust(width)} : Actual  !=  Resource Estimate"]
    rows = [f"{op.rjust(width)} : {str(a).rjust(6)}  !=  {e}" for op, a, e in mismatches]
    table = "\n".join(header + rows)
    raise AssertionError("The actual gate counts do not match the resource estimate:\n" + table)


def decomp_rule_to_tape(op: Operator, rule: DecompositionRule) -> QuantumScript:
    """Apply a decomposition rule to an operator and record the result as a tape.

    If program capture is enabled, the rule is captured and the tape is extracted from the
    resulting plxpr with :func:`~.testing.plxpr_to_tape`. Otherwise, the rule is called within
    a queuing context, and the queued operations are returned.

    Args:
        op (Operator): the operator to decompose
        rule (DecompositionRule): a decomposition rule for ``op``

    Returns:
        QuantumScript: a tape containing the operations produced by the rule

    **Example**

    .. code-block:: python

        @qp.register_resources({qp.H: 2, qp.CZ: 1})
        def cnot_to_cz(wires):
            qp.H(wires[1])
            qp.CZ(wires)
            qp.H(wires[1])

    >>> qp.testing.decomp_rule_to_tape(qp.CNOT([0, 1]), cnot_to_cz).operations
    [H(1), CZ(wires=[0, 1]), H(1)]

    """
    if not capture.enabled():
        _, args, kwargs = _get_decomp_args(op)
        with queuing.AnnotatedQueue() as q:
            rule(*args, **kwargs)
        return QuantumScript.from_queue(q)

    # Match each operator model's capture boundary: legacy hyperparameters remain
    # closed over, while Operator2 exposes its dynamic, wire, and hybrid arguments.
    decomp, args, kwargs = _prepare_decomp_for_capture(op, rule)
    plxpr = capture.make_plxpr(decomp, autograph=False)(*args, **kwargs)
    flat_capture_args = jax.tree.leaves((args, kwargs))
    return plxpr_to_tape(plxpr.jaxpr, plxpr.consts, *flat_capture_args)


@singledispatch
# pylint: disable-next=unused-argument
def _prepare_decomp_for_capture(op: Operator, rule: DecompositionRule):
    raise NotImplementedError  # pragma: no cover


@_prepare_decomp_for_capture.register
def _prepare_for_operator1(op: Operator1, rule: DecompositionRule):
    decomp = partial(rule, **op.hyperparameters)
    capture_args = op.data
    capture_kwargs = {"wires": op.wires}
    return decomp, capture_args, capture_kwargs


@_prepare_decomp_for_capture.register
def _prepare_operator2_rule(op: Operator2, rule: DecompositionRule):
    decomp = partial(rule, **op.static_args, **op.compilable_args)
    wire_args = {k: math.array(w, like="jax") for k, w in op.wire_args.items()}
    capture_kwargs = {**op.dynamic_args, **wire_args, **op.hybrid_args}
    return decomp, (), capture_kwargs


def assert_valid_decomp_rule(
    op: Operator,
    rule: DecompositionRule,
    check_matrix: bool = True,
) -> None:
    """Check that a decomposition rule is consistent with the operator it decomposes.

    The checks are skipped if the rule is not applicable to ``op``. Otherwise, the gate counts
    declared by the rule's resource function must match the operations the rule produces, and the
    matrix of the decomposition must match the matrix of ``op``.

    Args:
        op (Operator): the operator to decompose
        rule (DecompositionRule): a decomposition rule for ``op``
        check_matrix (bool): If ``True``, the matrix of the decomposition is compared with the
            matrix of ``op``. Defaults to ``True``.

    Raises:
        AssertionError: if the rule is not consistent with ``op``

    .. seealso:: :func:`~.testing.assert_decomposable`, which runs this check for every rule
        registered for an operator and requires at least one of them to be applicable.

    **Example**

    .. code-block:: python

        def cnot_to_cz(wires):
            qp.H(wires[1])
            qp.CZ(wires)
            qp.H(wires[1])

        good_rule = qp.register_resources({qp.H: 2, qp.CZ: 1})(cnot_to_cz)
        bad_rule = qp.register_resources({qp.H: 1, qp.CZ: 1})(cnot_to_cz)

    >>> qp.testing.assert_valid_decomp_rule(qp.CNOT([0, 1]), good_rule)
    >>> qp.testing.assert_valid_decomp_rule(qp.CNOT([0, 1]), bad_rule)
    Traceback (most recent call last):
        ...
    AssertionError: The actual gate counts do not match the resource estimate:
    Operator : Actual  !=  Resource Estimate
    Hadamard :      2  !=  1

    """

    params, _, _ = _get_decomp_args(op)

    if not rule.is_applicable(**params):
        return

    # Test that the resource function is correct
    resources = rule.compute_resources(**params)
    estimated_gates = resources.gate_counts

    # Make sure all counts are int
    for gate, count in estimated_gates.items():
        assert isinstance(count, int), (
            f"Resource count for '{gate}' in '{op.name}' decomp rule '{rule.name}' must "
            f"be an integer, but got {type(count)} ({count}). "
        )

    tape = decomp_rule_to_tape(op, rule)

    total_work_wires = rule.get_work_wire_spec(**params).total
    if total_work_wires:
        tape = _resolve_dynamic_wires(tape, total_work_wires)

    actual_gates = defaultdict(int)
    for _op in tape.operations:
        if isinstance(_op, Conditional):
            _op = _op.base
        op_rep = abstractify(_op)
        actual_gates[op_rep] += 1

    # Every SubroutineOp shares one generic decomposition rule, which always claims exact resources
    # but takes its gate counts from the subroutine's own ``compute_resources``. A subroutine with
    # ``exact_resources=False`` therefore only needs to pass the weaker check below.
    is_inexact_subroutine = isinstance(op, SubroutineOp) and not op.subroutine.exact_resources
    if rule.exact_resources and not is_inexact_subroutine:
        _assert_counts_match(estimated_gates, actual_gates)
    else:
        # An inexact resource estimate must still account for every gate in the decomposition.
        missing = sorted(str(gate) for gate in actual_gates.keys() - estimated_gates.keys())
        assert not missing, f"The resources is missing gates in the actual decomposition: {missing}"

    # Tests that the decomposition produces the same matrix
    if check_matrix and op.has_matrix and not _decomp_contains_mcm(rule, params):
        # Add projector to the additional wires (work wires) on the tape
        work_wires = tape.wires - op.wires
        all_wires = op.wires + work_wires
        if work_wires:
            op = op @ Projector([0] * len(work_wires), wires=work_wires)
            tape.operations.insert(0, Projector([0] * len(work_wires), wires=work_wires))

        op_matrix = op.matrix(wire_order=all_wires)
        with capture.pause():
            decomp_matrix = op_funcs.matrix(tape, wire_order=all_wires)
        assert math.allclose(
            op_matrix, decomp_matrix
        ), "decomposition must produce the same matrix as the operator."


def assert_decomposable(op: Operator, check_matrix: bool = True) -> None:
    """Check that an operator can be decomposed with its registered decomposition rules.

    At least one rule registered for ``op`` with :func:`~.decomposition.add_decomps` must be applicable to
    ``op``, and every applicable rule is checked with :func:`~.testing.assert_valid_decomp_rule`.

    Args:
        op (Operator): the operator to decompose
        check_matrix (bool): If ``True``, the matrix of each decomposition is compared with the
            matrix of ``op``. Defaults to ``True``.

    Raises:
        AssertionError: if no registered rule is applicable to ``op``, or if an applicable rule
            is not consistent with ``op``

    **Example**

    >>> qp.testing.assert_decomposable(qp.CNOT([0, 1]))

    .. code-block:: python

        class MyOp(qp.operation.Operator):
            num_wires = 1

        op = MyOp(0)

    >>> qp.testing.assert_decomposable(op)
    Traceback (most recent call last):
        ...
    AssertionError: No decomposition rules are registered for MyOp(wires=[0]).

    """

    rules = list_decomps(op)
    assert rules, f"No decomposition rules are registered for {op}."

    params, _, _ = _get_decomp_args(op)
    applicable_rules = [rule for rule in rules if rule.is_applicable(**params)]
    assert applicable_rules, f"None of the decomposition rules registered for {op} are applicable."

    for rule in applicable_rules:
        try:
            assert_valid_decomp_rule(op, rule, check_matrix=check_matrix)
        except AssertionError as e:
            raise AssertionError(
                f"Decomposition rule {rule.name} for {op.name} fails the validity check."
            ) from e
