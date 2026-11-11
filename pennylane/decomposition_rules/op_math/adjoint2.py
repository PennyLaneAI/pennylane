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

"""Built-in decomposition rules for op_math.adjoint2."""

from functools import partial
from textwrap import dedent

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    DecompositionRule,
    _decomp_contains_mcm,
    _get_registered_decomps,
    _list_decomps,
    get_fixed_decomp,
    list_decomps,
    register_condition,
    register_resources,
)
from pennylane.decomposition.utils import to_name
from pennylane.decomposition_rules.op_math.symbolicop2 import self_adjoint
from pennylane.ops.op_math.adjoint2 import (
    Adjoint2,
    _adjoint_abstract,
    get_traced_and_non_traced_args,
)


@_list_decomps.register
def _list_adjoint_decomps(op: Adjoint2) -> DecompCollection:

    abs_op = abstractify(op)

    # fixed_decomps would override everything.
    if fixed_rule := get_fixed_decomp(op):
        return DecompCollection([fixed_rule])

    # special case of cancelling nested adjoints.
    if isinstance(abs_op.base, Adjoint2):
        return DecompCollection([cancel_adjoint])

    # Custom decomposition rules registered specifically to the adjoint operator
    custom_rules = _get_registered_decomps(to_name(abs_op))

    # Shortcut that ignores general rules if an operator is self-adjoint
    if self_adjoint in custom_rules:
        return custom_rules

    # Decomposition rules populated by applying adjoint on the base decomp rules
    wrapped_rules = DecompCollection(
        [
            _make_adjoint_decomp(rule)
            for rule in list_decomps(abs_op.base)
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

    def _wrap_work_wire_spec(base_rule):

        # pylint: disable=protected-access
        if isinstance(base_rule._work_wire_spec, dict):
            return base_rule._work_wire_spec

        def _wrapper(base, *_, **__):
            return base_rule._work_wire_spec(**base.arguments)

        return _wrapper

    base_source = base_rule._source

    # pylint: disable=protected-access
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=_wrap_work_wire_spec(base_rule),
        exact=base_rule.exact_resources,
        name=f"adjoint({base_rule.name})",
    )
    def _impl(base):
        # pylint: disable=protected-access
        non_traced_args, traced_args = get_traced_and_non_traced_args(base)
        impl = partial(base_rule._impl, **non_traced_args)
        qp.adjoint(impl)(**traced_args)

    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere base_decomposition is defined as:\n\n"
        + dedent(base_source).strip()
    )

    return _impl


def _cancel_adjoint_resources(base):
    assert isinstance(base, Adjoint2)
    inner_base = base.base
    return {inner_base: 1}


@register_resources(_cancel_adjoint_resources)
def cancel_adjoint(base):
    """Decompose the adjoint of the adjoint of an operator."""
    assert isinstance(base, Adjoint2)
    type(base.base)(**base.base.arguments)


def _adjoint_rotation_resource(base):
    return {abstractify(base): 1}


@register_resources(_adjoint_rotation_resource)
def adjoint_rotation(base):
    """Decompose the adjoint of a rotation operator by inverting the angle."""
    # A rotation should only have 1 dynamic parameter
    assert len(base.dynamic_argnames) == 1
    angle = tuple(base.dynamic_args.values())[0]
    qp.ops.functions.bind_new_parameters(base, (-angle,))
