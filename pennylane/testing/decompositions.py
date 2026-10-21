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
from functools import partial

import jax

from pennylane import capture, math
from pennylane.core import queuing
from pennylane.core.operator import Operator1, abstractify
from pennylane.decomposition.decomposition_rule import DecompositionRule, _decomp_contains_mcm
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


def _assert_counts_match(counts_0, counts_1):

    if counts_0 == counts_1:
        return

    miscounts = [
        (op, val, counts_0[op])
        for op, val in counts_1.items()
        if op in counts_0 and val != counts_0[op]
    ]
    if miscounts:
        op_len = max([8] + [len(str(op)) for op, *_ in miscounts])
        miscounts_str = (
            f"\nThe numbers are off for following ops:"
            f"\n{'Operator'.rjust(op_len)} : Actual  !=  Resource function\n"
        )
        miscounts_str += "\n".join(
            f"{str(op).rjust(op_len)} : {str(val0).rjust(6)}  !=  {val1}"
            for op, val0, val1 in miscounts
        )
    else:
        miscounts_str = ""
    assertion_error_string = (
        f"\nGate counts expected from resource function:\n{counts_0}"
        f"\nActual gate counts:\n{dict(counts_1)}"
        f"{miscounts_str}"
        "\nMissing entirely in gate counts from resource function:\n"
        f"{[op for op in counts_1 if op not in counts_0]}\n"
        "Missing entirely in actual gate counts:\n"
        f"{[op for op in counts_0 if op not in counts_1]}"
    )
    raise AssertionError(assertion_error_string)


def _decomp_rule_to_tape(rule, args, kwargs):
    with queuing.AnnotatedQueue() as q:
        rule(*args, **kwargs)
    return QuantumScript.from_queue(q)


def _capture_decomp_rule_to_tape(rule, op):

    # Match each operator model's capture boundary: legacy hyperparameters remain
    # closed over, while Operator2 exposes its dynamic, wire, and hybrid arguments.
    if isinstance(op, Operator1):
        decomposition = partial(rule, **op.hyperparameters)
        capture_args = op.data
        capture_kwargs = {"wires": op.wires}
    else:
        decomposition = partial(rule, **op.static_args, **op.compilable_args)
        capture_args = ()
        wire_args = {k: math.array(w, like="jax") for k, w in op.wire_args.items()}
        capture_kwargs = {**op.dynamic_args, **wire_args, **op.hybrid_args}

    plxpr = capture.make_plxpr(decomposition, autograph=False)(*capture_args, **capture_kwargs)
    flat_capture_args = jax.tree.leaves((capture_args, capture_kwargs))
    return plxpr_to_tape(plxpr.jaxpr, plxpr.consts, *flat_capture_args)


def _test_decomposition_rule(op, rule: DecompositionRule, skip_decomp_matrix_check: bool = False):
    """Tests that a decomposition rule is consistent with the operator."""

    params, args, kwargs = _get_decomp_args(op)

    if not rule.is_applicable(**params):
        return

    # Test that the resource function is correct
    resources = rule.compute_resources(**params)
    estimated_gate_counts = resources.gate_counts

    # Make sure all counts are int
    for gate, count in estimated_gate_counts.items():
        assert isinstance(count, int), (
            f"Resource count for '{gate}' in '{op.name}' decomp rule '{rule.name}' must be an integer, "
            f"but got {type(count)} ({count}). "
        )

    tape = (
        _capture_decomp_rule_to_tape(rule, op)
        if capture.enabled()
        else _decomp_rule_to_tape(rule, args, kwargs)
    )

    total_work_wires = rule.get_work_wire_spec(**params).total
    if total_work_wires:
        tape = _resolve_dynamic_wires(tape, total_work_wires)

    actual_gate_counts = defaultdict(int)
    for _op in tape.operations:
        if isinstance(_op, Conditional):
            _op = _op.base
        op_rep = abstractify(_op)
        actual_gate_counts[op_rep] += 1
    actual_gate_counts = dict(sorted(actual_gate_counts.items(), key=lambda item: str(item[0])))

    if rule.exact_resources and not (
        isinstance(op, SubroutineOp) and not op.subroutine.exact_resources
    ):
        non_zero_gate_counts = {k: v for k, v in estimated_gate_counts.items() if v > 0}
        _assert_counts_match(non_zero_gate_counts, actual_gate_counts)
    else:
        # If the resource estimate is not expected to match exactly to the actual
        # decomposition, at least make sure that all gates are accounted for.
        assert all(op in estimated_gate_counts for op in actual_gate_counts), (
            "\nGate counts expected from resource function to contain actual gates:\n"
            f"{list(estimated_gate_counts.keys())}\nActual gates:\n{list(actual_gate_counts.keys())}\n"
            "Missing in gate counts from resource function:\n"
            f"{[op for op in actual_gate_counts if op not in estimated_gate_counts]}"
        )

    # Tests that the decomposition produces the same matrix
    if op.has_matrix and not skip_decomp_matrix_check and not _decomp_contains_mcm(rule, params):
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
