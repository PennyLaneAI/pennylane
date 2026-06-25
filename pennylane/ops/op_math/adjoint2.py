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

from textwrap import dedent

from typing_extensions import override

import pennylane as qp
from pennylane import math
from pennylane._class_property import classproperty
from pennylane.core.operator import Operator2
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    DecompositionRule,
    list_decomps,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import (
    AbstractOperatorLike,
    CompressedResourceOp,
    adjoint_resource_rep,
    resource_rep,
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


@list_decomps.register
def _list_adjoint_decomps(op: Adjoint2) -> DecompCollection:
    if isinstance(op.base, Adjoint2):
        return DecompCollection([cancel_adjoint])
    custom_rules = list_decomps.dispatch(object)(op)
    wrapped_rules = DecompCollection(
        [
            _make_adjoint_decomp(rule)
            for rule in list_decomps(op.base)
            # It only makes sense to wrap a decomposition rule with adjoint if the decomposition
            # does not dynamically allocate wires and does not contain mid-circuit measurements.
            if rule.get_work_wire_spec(**op.base.arguments).total == 0
            and not _decomp_contains_mcm(rule, op.base.arguments)
        ]
    )
    return custom_rules + wrapped_rules


def _decomp_contains_mcm(rule, params):
    resources = rule.compute_resources(**params).gate_counts
    mcm = resource_rep(qp.ops.MidMeasure)
    ppm = resource_rep(qp.ops.PauliMeasure)
    return mcm in resources or ppm in resources


def _make_adjoint_decomp(base_rule: DecompositionRule):
    """Wraps a decomposition rule with adjoint."""

    def _condition_fn(base):
        return base_rule.is_applicable(**base.arguments)

    def _resource_fn(base):
        base_res = base_rule.compute_resources(**base.arguments)
        base_gates = base_res.gate_counts
        return {_adjoint(op): count for op, count in base_gates.items()}

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


def _adjoint(op: AbstractOperatorLike):
    if isinstance(op, CompressedResourceOp):
        return adjoint_resource_rep(op.op_type, op.params)
    return Adjoint2(op)


def _cancel_adjoint_resources(base):
    assert isinstance(base, Adjoint2)
    inner_base = base.base
    return {inner_base: 1}


@register_resources(_cancel_adjoint_resources)
def cancel_adjoint(base):
    """Decompose the adjoint of the adjoint of an operator."""
    assert isinstance(base, Adjoint2)
    type(base.base)(**base.base.arguments)
