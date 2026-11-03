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
"""A developer tool that checks whether a circuit can be decomposed to a target gate set using only
decomposition rules that can be captured into plxpr.

.. warning::

    This module is a standalone developer tool for validating workflows. It is not exported and is
    not part of the public API.

The :func:`validate_decomp_capture` transform captures the circuit, collects every operator in it
(from every branch of every ``cond``, and every loop body), and then traces every applicable
decomposition rule for each operator into plxpr. The operators produced by each captured rule are
collected the same way and explored recursively. Finally, it checks whether every operator in the
circuit has a decomposition pathway to the target gate set that only uses rules that are supported
and capture successfully.

A decomposition rule is considered unsupported, and is not traced, if

- it dynamically allocates work wires (its ``work_wire_spec`` requests any work wires, or its
  captured body allocates wires), or
- it depends on legacy operators (its resource function produces a ``CompressedResourceOp``, or its
  captured body produces a legacy operator).

Legacy (non-``Operator2``) operators that are not in the target gate set are always unsolvable.

**Example**

.. code-block:: python

    import jax
    import pennylane as qp
    from pennylane.transforms.validate_decomp_capture import validate_decomp_capture

    jax.config.update("jax_enable_x64", True)

    @qp.register_resources({qp.RX: 1})
    def bad_ry(phi, wires, **__):
        if phi > 0:  # Python control flow on a traced parameter cannot be captured
            qp.RX(phi, wires)

    @validate_decomp_capture(
        gate_set={"RX", "RZ", "CNOT", "GlobalPhase"},
        fixed_decomps={qp.RY: bad_ry},
    )
    @qp.qnode(qp.device("default.qubit", wires=2))
    def circuit(x):
        qp.CRX(x, wires=[0, 1])
        qp.RY(x, wires=0)
        return qp.expval(qp.Z(0))

>>> report = circuit(0.5)
>>> report.is_valid
False
>>> report.unsolved_ops
[RY(AbstractArray((), float64, weak_type=True), wires=AbstractWires(1))]
>>> print(report.inspect(qp.RY(0.5, 0)))
RY: unsolved
  bad_ry: capture failed (TracerBoolConversionError)

The full error messages of the failed rules are available via ``report.details(op)``, and
``print(report)`` lists every unsolved operator encountered, along with the status of each of its
decomposition rules.

Known limitations:

- Rules that allocate work wires, or that depend on legacy operators, are not supported, so an
  operator is only solvable through its remaining rules.
- Operators are identified by their abstract form (see :func:`~pennylane.core.operator.abstractify`),
  and each rule is traced on a dummy instance of that abstract form. Dynamic values never influence
  which operators are found.
- The circuit is captured with ``autograph=False``.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial, wraps

import jax
import jax.numpy as jnp

from pennylane import capture, math
from pennylane.allocation import Allocate, Deallocate, allocate_prim, deallocate_prim
from pennylane.capture import PlxprInterpreter
from pennylane.capture.base_interpreter import jaxpr_to_jaxpr
from pennylane.capture.primitives import (
    adjoint_transform_prim,
    ctrl_transform_prim,
    measure_prim,
    pauli_measure_prim,
)
from pennylane.core.operator import Operator, Operator2, abstractify
from pennylane.decomposition import DecompositionRule, GateSet
from pennylane.decomposition.decomposition_rule import (
    _fix_decomp,
    add_decomps,
    list_decomps,
    local_decomps,
)
from pennylane.decomposition.resources import CompressedResourceOp
from pennylane.decomposition.utils import to_name
from pennylane.exceptions import DecompositionError
from pennylane.ops import Conditional, MidMeasure, PauliMeasure, adjoint, ctrl
from pennylane.pytrees import flatten, unflatten
from pennylane.templates import SubroutineOp
from pennylane.templates.core import CollectedSubroutine
from pennylane.transforms.core import transform
from pennylane.typing import AbstractArray, AbstractWires
from pennylane.wires import Wires

from .decompose import _resolve_gate_set

_IGNORED_OPS = {"Barrier", "Snapshot"}

_NOT_APPLICABLE = "not applicable"
_USES_ALLOCATION = "not supported (uses dynamic wire allocation)"
_LEGACY = "not supported (depends on legacy operators)"
_RESOURCES_FAILED = "resource function failed"
_CAPTURE_FAILED = "capture failed"
_CAPTURED = "captured"


@dataclass
class _RuleRecord:
    """The outcome of checking a single decomposition rule for an operator."""

    rule: DecompositionRule
    status: str
    error: Exception | None = None
    produced: frozenset = frozenset()
    legacy_ops: tuple = ()


@dataclass
class _OpRecord:
    """The outcome of checking all decomposition rules of an operator."""

    reason: str | None = None
    """Set when the rules of this operator could not be examined at all."""

    rules: list[_RuleRecord] = field(default_factory=list)


@dataclass
class DecompCaptureReport:
    """The result of :func:`validate_decomp_capture`."""

    circuit_ops: list
    """The abstract operators found in the circuit."""

    records: dict
    """Maps every explored abstract operator to the outcome of checking its decomposition rules."""

    solved: set
    """The abstract operators with a supported and captured decomposition pathway to the target
    gate set, including the operators in the target gate set."""

    gate_set: GateSet
    """The target gate set."""

    capture_error: Exception | None = None
    """The error raised when capturing the circuit, if any."""

    @property
    def is_valid(self) -> bool:
        """Whether every operator in the circuit can be decomposed to the target gate set using
        supported, capture-compatible decomposition rules."""
        return self.capture_error is None and not self.unsolved_ops

    @property
    def unsolved_ops(self) -> list:
        """The abstract operators in the circuit that cannot be decomposed to the target gate set
        using supported, capture-compatible decomposition rules."""
        return [op for op in self.circuit_ops if op not in self.solved]

    @property
    def all_unsolved_ops(self) -> list:
        """Every explored abstract operator, including those produced by decomposition rules, that
        cannot be decomposed to the target gate set."""
        return [op for op in self.records if op not in self.solved]

    def inspect(self, op) -> str:
        """Describe how each decomposition rule of an operator was evaluated.

        Args:
            op (Operator): an operator instance, or its abstract form

        Returns:
            str: a description of the status of every decomposition rule of the operator
        """
        key = abstractify(op)
        if key in self.gate_set:
            return f"{key}: in the target gate set"
        if key not in self.records:
            return f"{key}: not encountered in the circuit or any explored decomposition rule"
        header = f"{key}: {'solved' if key in self.solved else 'unsolved'}"
        record = self.records[key]
        if record.reason:
            return f"{header} ({record.reason})"
        if not record.rules:
            return f"{header} (no decomposition rules)"
        lines = [header] + [f"  {self._describe_rule(r)}" for r in record.rules]
        return "\n".join(lines)

    def _describe_rule(self, record: _RuleRecord) -> str:
        name = record.rule.name
        if record.status == _CAPTURED:
            blocking = [op for op in record.produced if op not in self.solved]
            if not blocking:
                return f"{name}: OK"
            blocking_str = ", ".join(sorted(str(op) for op in blocking))
            return f"{name}: captured, but blocked by unsolved operators {{{blocking_str}}}"
        if record.status in (_CAPTURE_FAILED, _RESOURCES_FAILED):
            return f"{name}: {record.status} ({type(record.error).__name__})"
        if record.status == _LEGACY and record.legacy_ops:
            legacy_str = ", ".join(sorted(str(op) for op in record.legacy_ops))
            return f"{name}: {record.status} {{{legacy_str}}}"
        return f"{name}: {record.status}"

    def details(self, op) -> str:
        """Return the full error messages of the rules of an operator that failed.

        Args:
            op (Operator): an operator instance, or its abstract form

        Returns:
            str: the error message of every rule that failed to capture or estimate resources
        """
        record = self.records.get(abstractify(op))
        if record is None:
            return ""
        return "\n\n".join(
            f"{r.rule.name}: {type(r.error).__name__}: {r.error}"
            for r in record.rules
            if r.error is not None
        )

    def __str__(self) -> str:
        if self.capture_error is not None:
            return (
                "The circuit could not be captured: "
                f"{type(self.capture_error).__name__}: {self.capture_error}"
            )
        if self.is_valid:
            return (
                f"All {len(self.circuit_ops)} operators in the circuit can be decomposed to the "
                f"target gate set {self.gate_set} using supported, capture-compatible "
                "decomposition rules."
            )
        circuit_str = ", ".join(str(op) for op in self.unsolved_ops)
        sections = "\n\n".join(self.inspect(op) for op in self.all_unsolved_ops)
        return (
            f"The following operators in the circuit cannot be decomposed to the target gate set "
            f"{self.gate_set} using supported, capture-compatible decomposition rules: "
            f"{{{circuit_str}}}\n\nAll unsolved operators:\n\n{sections}"
        )

    def __repr__(self) -> str:
        return f"<DecompCaptureReport: is_valid={self.is_valid}>"


class _CollectOpKeys(PlxprInterpreter):
    """Records the abstract form of every operator in a plxpr.

    Higher-order primitives like ``cond``, ``for_loop``, ``while_loop``, subroutines and qnodes are
    handled by the base class, which re-traces every branch and body. This interpreter must always
    be executed while tracing (e.g., with ``jaxpr_to_jaxpr``), so that nothing is executed.
    """

    def __init__(self, collected=None, wrap: Callable | None = None):
        super().__init__()
        self.collected = collected or {"ops": {}, "uses_allocation": False}
        self.wrap = wrap or (lambda op: op)

    def record(self, op):
        """Record the abstract form of an operator, after applying the modifier wrapper."""
        with capture.pause():
            self.collected["ops"][abstractify(self.wrap(op))] = None

    def interpret_operation(self, op):
        self.record(op)
        return op


@_CollectOpKeys.register_primitive(adjoint_transform_prim)
def _adjoint_transform(self, *invals, jaxpr, lazy, n_consts):
    def wrap(op):
        return self.wrap(adjoint(op, lazy=lazy))

    _CollectOpKeys(self.collected, wrap).eval(jaxpr, invals[:n_consts], *invals[n_consts:])
    return []


@_CollectOpKeys.register_primitive(ctrl_transform_prim)
def _ctrl_transform(self, *invals, jaxpr, n_consts, n_control, **ctrl_params):
    control = list(invals[-n_control:])

    def wrap(op):
        return self.wrap(ctrl(op, control=control, **ctrl_params))

    child = _CollectOpKeys(self.collected, wrap)
    child.eval(jaxpr, invals[:n_consts], *invals[n_consts:-n_control])
    return []


@_CollectOpKeys.register_primitive(measure_prim)
def _measure(self, wires, reset, postselect):
    with capture.pause():
        self.record(MidMeasure(wires=wires, reset=reset, postselect=postselect))
    return measure_prim.bind(wires, reset=reset, postselect=postselect)


@_CollectOpKeys.register_primitive(pauli_measure_prim)
def _pauli_measure(self, *wires, pauli_word="", postselect=None):
    with capture.pause():
        self.record(PauliMeasure(pauli_word, wires=list(wires), postselect=postselect))
    return pauli_measure_prim.bind(*wires, pauli_word=pauli_word, postselect=postselect)


@_CollectOpKeys.register_primitive(allocate_prim)
def _allocate(self, **params):
    self.collected["uses_allocation"] = True
    return allocate_prim.bind(**params)


@_CollectOpKeys.register_primitive(deallocate_prim)
def _deallocate(self, *wires):
    self.collected["uses_allocation"] = True
    return deallocate_prim.bind(*wires)


def _collect_ops(plxpr, args) -> tuple[list, bool]:
    """Collect the abstract operators in a plxpr, and whether it dynamically allocates wires."""
    collector = _CollectOpKeys()
    with capture.toggle_ctx(True):
        jaxpr_to_jaxpr(collector, plxpr.jaxpr, plxpr.consts, *args)
    return list(collector.collected["ops"]), collector.collected["uses_allocation"]


def _concretize(op: Operator2) -> Operator2:
    """Build a dummy instance of an abstract operator, with zeros for every dynamic argument and
    distinct integer wire labels."""

    wire_labels = itertools.count()

    def _leaf_to_dummy(leaf):
        if isinstance(leaf, AbstractArray):
            return jnp.zeros(leaf.shape, leaf.dtype)
        if isinstance(leaf, AbstractWires):
            return Wires([next(wire_labels) for _ in range(len(leaf))])
        return leaf

    leaves, struct = flatten(op, is_leaf=lambda l: isinstance(l, (AbstractArray, AbstractWires)))
    with capture.pause():
        return unflatten([_leaf_to_dummy(leaf) for leaf in leaves], struct)


def _capture_rule(rule: DecompositionRule, op: Operator2):
    """Capture a decomposition rule applied to an operator, using the same calling convention as
    ``assert_valid`` for ``Operator2``."""
    decomposition = partial(rule, **op.static_args, **op.compilable_args)
    wire_args = {k: math.array(w, like="jax") for k, w in op.wire_args.items()}
    capture_kwargs = {**op.dynamic_args, **wire_args, **op.hybrid_args}
    with capture.toggle_ctx(True):
        plxpr = capture.make_plxpr(decomposition, autograph=False)(**capture_kwargs)
    return plxpr, jax.tree.leaves(capture_kwargs)


def _check_rule_metadata(rule: DecompositionRule, op: Operator2) -> _RuleRecord | None:
    """Check whether a decomposition rule is unsupported based on its applicability, work wire
    spec and resources. Returns ``None`` if the rule should be captured."""

    params = op.arguments
    if not rule.is_applicable(**params):
        return _RuleRecord(rule, _NOT_APPLICABLE)
    if rule.get_work_wire_spec(**params).total > 0:
        return _RuleRecord(rule, _USES_ALLOCATION)
    try:
        resources = rule.compute_resources(**params).gate_counts
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _RuleRecord(rule, _RESOURCES_FAILED, error=e)
    if legacy_ops := tuple(r for r in resources if isinstance(r, CompressedResourceOp)):
        return _RuleRecord(rule, _LEGACY, legacy_ops=legacy_ops)
    return None


def _check_rule(rule: DecompositionRule, op: Operator2, dummy: Operator2) -> _RuleRecord:
    """Check whether a decomposition rule is supported and can be captured."""

    if record := _check_rule_metadata(rule, op):
        return record
    try:
        plxpr, args = _capture_rule(rule, dummy)
        produced, uses_allocation = _collect_ops(plxpr, args)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _RuleRecord(rule, _CAPTURE_FAILED, error=e)

    if uses_allocation:
        return _RuleRecord(rule, _USES_ALLOCATION)
    legacy_ops = tuple(
        p for p in produced if not isinstance(p, Operator2) and to_name(p) not in _IGNORED_OPS
    )
    if legacy_ops:
        return _RuleRecord(rule, _LEGACY, legacy_ops=legacy_ops)
    return _RuleRecord(rule, _CAPTURED, produced=frozenset(produced))


def _is_terminal(op, gate_set: GateSet) -> bool:
    return op in gate_set or to_name(op) in _IGNORED_OPS


def _check_op(op) -> _OpRecord:
    """Check all decomposition rules of an abstract operator."""

    if not isinstance(op, Operator2):
        return _OpRecord(reason="legacy operator, its decomposition rules are not supported")
    try:
        dummy = _concretize(op)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _OpRecord(reason=f"could not build a dummy instance: {type(e).__name__}: {e}")
    return _OpRecord(rules=[_check_rule(rule, op, dummy) for rule in list_decomps(op)])


def _explore(circuit_ops: Iterable, gate_set: GateSet) -> dict:
    """Breadth-first exploration of every operator reachable through supported, captured rules."""

    records = {}
    queue = deque(circuit_ops)
    seen = set(queue)
    while queue:
        op = queue.popleft()
        if _is_terminal(op, gate_set):
            continue
        records[op] = _check_op(op)
        for rule_record in records[op].rules:
            for produced in rule_record.produced:
                if produced not in seen:
                    seen.add(produced)
                    queue.append(produced)
    return records


def _solve(circuit_ops: Iterable, records: dict, gate_set: GateSet) -> set:
    """Find every operator with a decomposition pathway to the gate set, by fixed-point iteration."""

    solved = {op for op in circuit_ops if _is_terminal(op, gate_set)}
    for record in records.values():
        for rule_record in record.rules:
            solved.update(op for op in rule_record.produced if _is_terminal(op, gate_set))

    changed = True
    while changed:
        changed = False
        for op, record in records.items():
            if op in solved:
                continue
            if any(r.status == _CAPTURED and r.produced <= solved for r in record.rules):
                solved.add(op)
                changed = True
    return solved


@contextmanager
def _custom_decomps(fixed_decomps: dict | None, alt_decomps: dict | None):
    """Apply ``fixed_decomps`` and ``alt_decomps`` locally, as ``DecompositionGraph`` does."""

    fixed_decomps = fixed_decomps or {}
    alt_decomps = alt_decomps or {}
    all_rules = itertools.chain(fixed_decomps.values(), *alt_decomps.values())
    if rule := next((r for r in all_rules if not isinstance(r, DecompositionRule)), None):
        raise TypeError(
            f"{rule.__name__} is missing a resource estimate! A quantum function must be "
            "decorated with @qp.register_resources to be used as a decomposition rule."
        )

    with local_decomps():
        for op, decomps in alt_decomps.items():
            add_decomps(to_name(op), *decomps)
        for op, decomp in fixed_decomps.items():
            _fix_decomp(to_name(op), decomp)
        yield


def _validate(circuit_ops, gate_set, fixed_decomps, alt_decomps) -> DecompCaptureReport:
    gate_set, _ = _resolve_gate_set(gate_set)
    circuit_ops = list(dict.fromkeys(circuit_ops))
    with _custom_decomps(fixed_decomps, alt_decomps):
        records = _explore(circuit_ops, gate_set)
    return DecompCaptureReport(
        circuit_ops, records, _solve(circuit_ops, records, gate_set), gate_set
    )


def _finalize(report: DecompCaptureReport, raise_on_failure: bool) -> DecompCaptureReport:
    if raise_on_failure and not report.is_valid:
        raise DecompositionError(str(report))
    return report


def _ops_from_tape(operations: Iterable[Operator]) -> list:
    """Collect the abstract operators of a tape, expanding subroutines."""

    ops = []
    for op in operations:
        if isinstance(op, Conditional):
            op = op.base
        if isinstance(op, (Allocate, Deallocate)):
            continue
        if isinstance(op, (SubroutineOp, CollectedSubroutine)):
            ops.extend(_ops_from_tape(op.decomposition()))
            continue
        ops.append(abstractify(op))
    return ops


@partial(transform, is_informative=True)
def validate_decomp_capture(
    tape,
    *,
    gate_set,
    fixed_decomps: dict | None = None,
    alt_decomps: dict | None = None,
    raise_on_failure: bool = False,
):
    """Check that every operator in a circuit can be decomposed to a target gate set using only
    supported decomposition rules that can be captured into plxpr.

    When applied to a QNode, the circuit itself is captured, and the operators are collected from
    every branch of the captured program. When applied to a tape or quantum function, the operators
    of the tape are used directly.

    Args:
        tape (QuantumScript or QNode or Callable): a quantum circuit.
        gate_set (Iterable[str or type], Dict[type or str, float]): the target gate set.
        fixed_decomps (Dict[Type[Operator], DecompositionRule]): a dictionary mapping operator types
            to custom decomposition rules that replace the existing rules for the operator.
        alt_decomps (Dict[Type[Operator], List[DecompositionRule]]): a dictionary mapping operator
            types to lists of alternative custom decomposition rules.
        raise_on_failure (bool): if ``True``, raise a ``DecompositionError`` instead of returning
            the report when the validation fails.

    Returns:
        DecompCaptureReport: the outcome of the validation.

    Raises:
        DecompositionError: if ``raise_on_failure=True`` and the validation fails.
    """
    report = _validate(_ops_from_tape(tape.operations), gate_set, fixed_decomps, alt_decomps)
    _finalize(report, raise_on_failure)

    def postprocessing(_):
        return report

    return [tape], postprocessing


@validate_decomp_capture.custom_qnode_transform
def _validate_decomp_capture_qnode(_transform, qnode, _targs, tkwargs):
    """Capture the QNode and validate the operators collected from the captured program."""

    gate_set = tkwargs["gate_set"]
    fixed_decomps = tkwargs.get("fixed_decomps")
    alt_decomps = tkwargs.get("alt_decomps")
    raise_on_failure = tkwargs.get("raise_on_failure", False)

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        try:
            with capture.toggle_ctx(True):
                plxpr = capture.make_plxpr(qnode, autograph=False)(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            resolved_gate_set, _ = _resolve_gate_set(gate_set)
            report = DecompCaptureReport([], {}, set(), resolved_gate_set, capture_error=e)
            return _finalize(report, raise_on_failure)

        circuit_ops, _ = _collect_ops(plxpr, jax.tree.leaves((args, kwargs)))
        report = _validate(circuit_ops, gate_set, fixed_decomps, alt_decomps)
        return _finalize(report, raise_on_failure)

    return wrapper
