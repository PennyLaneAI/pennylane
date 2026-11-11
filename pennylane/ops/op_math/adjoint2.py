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

from importlib import import_module
from typing import override

import pennylane as qp
from pennylane import math
from pennylane._class_property import classproperty
from pennylane.capture.op_eqns import pop_op_eqns
from pennylane.core.operator import Operator, Operator2, abstractify
from pennylane.core.operator.operator2 import operator_p  # tach-ignore
from pennylane.core.queuing import remove_from_program
from pennylane.decomposition.decomposition_rule import list_decomps
from pennylane.decomposition.resources import (
    AbstractOperatorLike,
    CompressedResourceOp,
    adjoint_resource_rep,
)
from pennylane.decomposition.utils import to_name

from .symbolicop2 import SymbolicOp2


def get_traced_and_non_traced_args(op: Operator2) -> tuple[dict, dict]:
    """Split an ``Operator2``'s bound arguments into non-dynamic (static and compilable)
    and traced (everything else, including hybrid) argument dicts."""
    non_traced_argnames = op.static_argnames + op.compilable_argnames
    non_traced_args = {k: v for k, v in op.arguments.items() if k in non_traced_argnames}
    traced_args = {k: v for k, v in op.arguments.items() if k not in non_traced_argnames}
    return non_traced_args, traced_args


class Adjoint2(SymbolicOp2):
    """The adjoint of an operator."""

    wire_argnames = ()

    hybrid_argnames = ("base",)

    def __init__(self, base: Operator2):
        super().__init__(base)

    @property
    @override
    def data(self):
        return self.base.data

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
        new_base = self.base.simplify()
        if new_base is not self.base:
            remove_from_program(new_base)  # remove intermediate op from program
        if new_base.has_adjoint:
            adjoint_base = new_base.adjoint()
            simplified_base = adjoint_base.simplify()
            if simplified_base is not adjoint_base:
                remove_from_program(adjoint_base)  # remove intermediate op from program
            return simplified_base
        return Adjoint2(new_base)

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
        if math.is_tracer(res):
            self.tracer = res


def _adjoint_abstract(op: AbstractOperatorLike | type[Operator]):
    op = abstractify(op)
    if isinstance(op, CompressedResourceOp):
        return adjoint_resource_rep(op.op_type, op.params)
    return qp.adjoint(op)


@to_name.register
def _adjoint2_to_name(op: Adjoint2):
    return f"Adjoint({to_name(op.base)})"


def __getattr__(name):
    """Resolve the existing public names of relocated decomposition rules."""
    if name in {"cancel_adjoint", "adjoint_rotation"}:
        rules = import_module("pennylane.decomposition_rules.op_math.adjoint2")

        return getattr(rules, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
