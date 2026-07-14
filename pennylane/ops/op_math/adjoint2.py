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
"""Defines the base class for the adjoint of operators."""

from collections.abc import Sequence
from textwrap import dedent

from typing_extensions import override

import pennylane as qp
from pennylane import math
from pennylane._class_property import classproperty
from pennylane.core.operator import Operator, Operator2, abstractify
from pennylane.core.operator.operator2 import operator_p, pop_op_eqns  # tach-ignore
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    DecompositionRule,
    _decomp_contains_mcm,
    list_decomps,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import (
    AbstractOperatorLike,
    CompressedResourceOp,
    adjoint_resource_rep,
)

from .symbolicop2 import SymbolicOp2


class Adjoint2(SymbolicOp2):
    """The adjoint of an operator."""

    wire_argnames = ()

    hybrid_argnames = ("base",)

    def __init__(self, base: Operator2):
        super().__init__(base)

    @property
    @override
    def pauli_rep(self):
        if not self.base.pauli_rep:
            return None
        if not self._pauli_rep:
            rep = {pw: math.conjugate(c) for pw, c in self.base.pauli_rep.items()}
            self._pauli_rep = qp.pauli.PauliSentence(rep)
        return self._pauli_rep

    def __repr__(self) -> str:
        return f"Adjoint({self.base})"

    @property
    @override
    def name(self) -> str:
        return f"Adjoint({self.base.name})"

    @override
    def label(self, decimals=None, base_label=None, cache=None) -> str:
        base_label = self.base.label(decimals, base_label, cache=cache)
        if self.base.arithmetic_depth > 0 and len(base_label) > 1:
            base_label = f"({base_label})"
        return f"{base_label}†"

    @staticmethod
    @override
    def compute_matrix(base):  # pylint: disable=arguments-differ
        base_matrix = base.matrix()
        return math.moveaxis(math.conj(base_matrix), -2, -1)

    @staticmethod
    @override
    def compute_sparse_matrix(base, format="csr"):  # pylint: disable=arguments-differ
        base_matrix = base.sparse_matrix()
        return math.transpose(math.conj(base_matrix)).asformat(format=format)

    @staticmethod
    @override
    def compute_eigvals(base):  # pylint: disable=arguments-differ
        return math.conj(base.eigvals())

    @property
    @override
    def has_diagonalizing_gates(self):  # pylint: disable=arguments-differ,invalid-overridden-method
        return self.base.has_diagonalizing_gates

    @staticmethod
    @override
    def compute_diagonalizing_gates(base):  # pylint: disable=arguments-differ
        return base.diagonalizing_gates()

    @classproperty
    @classmethod
    @override
    def has_adjoint(cls):
        return True

    @override
    def adjoint(self):
        return type(self.base)(**self.base.arguments)

    @override
    def simplify(self):
        base = self.base.simplify()
        if base.has_adjoint:
            return base.adjoint().simplify()
        return Adjoint2(base)

    @property
    @override
    def has_generator(self):  # pylint: disable=arguments-differ,invalid-overridden-method
        return self.base.has_generator

    @override
    def generator(self):
        return -1 * self.base.generator()

    @property
    def has_decomposition(self):  # pylint: disable=arguments-differ,invalid-overridden-method
        return any(rule.is_applicable(**self.arguments) for rule in list_decomps(self))

    @override
    def _bind_primitive(self):
        """Bind the operator primitive. ``Adjoint2`` has to override the method of the base
        ``Operator2`` class so that we can "edit" the original primitive."""
        if not qp.capture.enabled():
            return

        eqns = pop_op_eqns((self.base,))
        assert len(eqns) <= 1, f"Expected at most one plxpr equation for {self.base}."

        if len(eqns) == 0:
            # pylint: disable=protected-access
            self.base._bind_primitive()
            if self.base.tracer is None:
                # If the base's tracer is `None` after explicitly re-binding the primitive,
                # it means we're not in a tracing context, so we don't need to (and cannot)
                # do anything.
                return

            eqn = self.base.tracer.parent
            eqn.params["adjoint"] ^= True
            res = self.base.tracer

        else:
            params = eqns[0].params
            params["adjoint"] ^= True
            # invars during tracing will just be tracers, not `Var`s wrapping
            # abstract values
            res = operator_p.bind(*eqns[0].invars, **params)

        self.base.tracer = None
        # If we bind the primitive outside a tracing context but with program capture enabled,
        # `res`` will be a concrete operator, not an abstract tracer, so we don't save it.
        if math.is_abstract(res):
            self.tracer = res


@list_decomps.register
def _list_adjoint_decomps(
    op: Adjoint2,
    fixed_decomps: dict[str, DecompositionRule] | None = None,
    alt_decomps: dict[str, Sequence[DecompositionRule]] | None = None,
) -> DecompCollection:
    """Populates the decomposition rules for an adjoint operator."""

    abs_op = abstractify(op)

    if isinstance(abs_op.base, Adjoint2):
        return DecompCollection([cancel_adjoint])

    # Custom decomposition rules registered specifically for this adjoint operator.
    custom_rules = list_decomps.dispatch(object)(abs_op, fixed_decomps, alt_decomps)

    # Applying adjoint to the decomposition rules of the base.
    wrapped_rules = DecompCollection(
        [
            _make_adjoint_decomp(rule)
            for rule in list_decomps(abs_op.base, fixed_decomps, alt_decomps)
            # It only makes sense to wrap a decomposition rule with adjoint if the decomposition
            # does not dynamically allocate wires and does not contain mid-circuit measurements.
            if rule.get_work_wire_spec(**abs_op.base.arguments).total == 0
            and not _decomp_contains_mcm(rule, abs_op.base.arguments)
        ]
    )
    return custom_rules + wrapped_rules


def _make_adjoint_decomp(base_rule: DecompositionRule):
    """Wraps a decomposition rule with adjoint."""

    def _condition_fn(base):
        return base_rule.is_applicable(**base.arguments)

    def _resource_fn(base):
        base_res = base_rule.compute_resources(**base.arguments)
        base_gates = base_res.gate_counts
        return {_adjoint_abstract(op): count for op, count in base_gates.items()}

    base_source = base_rule._source

    # pylint: disable=protected-access
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=base_rule._work_wire_spec,
        exact=base_rule.exact_resources,
        name=f"adjoint({base_rule.name})",
    )
    def _impl(base):
        # pylint: disable=protected-access
        qp.adjoint(base_rule._impl)(**base.arguments)

    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere base_decomposition is defined as:\n\n"
        + dedent(base_source).strip()
    )

    return _impl


def _adjoint_abstract(op: AbstractOperatorLike | type[Operator]):
    op = abstractify(op)
    if isinstance(op, CompressedResourceOp):
        return adjoint_resource_rep(op.op_type, op.params)
    return qp.adjoint(op)


def _cancel_adjoint_resources(base):
    assert isinstance(base, Adjoint2)
    inner_base = base.base
    return {inner_base: 1}


@register_resources(_cancel_adjoint_resources)
def cancel_adjoint(base):
    """Decompose the adjoint of the adjoint of an operator."""
    assert isinstance(base, Adjoint2)
    type(base.base)(**base.base.arguments)
