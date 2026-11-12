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

"""Built-in decomposition rules for op_math.controlled2."""

# pylint: disable=unused-argument,protected-access,no-value-for-parameter

from functools import partial
from textwrap import dedent

import pennylane as qp
from pennylane import allocation, capture, compiler, math
from pennylane.core.operator import abstractify
from pennylane.decomposition.decomposition_rule import (
    DecompCollection,
    DecompositionRule,
    _decomp_contains_mcm,
    _get_registered_decomps,
    _list_decomps,
    add_decomps,
    get_fixed_decomp,
    list_decomps,
    register_condition,
    register_resources,
)
from pennylane.decomposition.utils import to_name
from pennylane.ops.op_math.adjoint2 import Adjoint2, get_traced_and_non_traced_args
from pennylane.ops.op_math.controlled2 import ControlledOp2, _ctrl_abstract
from pennylane.typing import Bool, Complex, Wire
from pennylane.wires import concatenate_wires


@_list_decomps.register
def _list_controlled_decomps(op: ControlledOp2) -> DecompCollection:
    """Get all the decomposition rules applicable to this operator."""

    op = abstractify(op)

    # fixed_decomps should override everything
    if fixed_rule := get_fixed_decomp(op):
        return DecompCollection([fixed_rule])

    # Special case for flipping the order of control and adjoint. We prefer to have adjoint
    # wrapping a controlled operator instead of the other way around because there is more
    # likely custom decomposition rules registered for the controlled version.
    if isinstance(op.base, Adjoint2):
        return DecompCollection([flip_control_adjoint])

    # Retrieve only rules registered directly for this controlled operator.
    custom_rules = _get_registered_decomps(to_name(op))

    # The specialized ChangeOpBasis rule leaves the compute and uncompute operations
    # uncontrolled, so it must always take precedence over general controlled fallbacks.
    if isinstance(op.base, (qp.ops.ChangeOpBasis, qp.ops.ChangeOpBasis2)):
        return custom_rules

    # Get general fallback rules.
    general_rules = DecompCollection(
        [rule for rule in list_decomps("ControlledOp") if rule.is_applicable(**op.arguments)]
    )

    # Populate controlled versions of the base decomposition rules.
    wrapped_rules = DecompCollection(
        [
            _make_controlled_decomp(rule)
            for rule in list_decomps(op.base)
            if not _decomp_contains_mcm(rule, op.base.arguments)
        ]
    )

    return custom_rules + wrapped_rules + general_rules


def _make_controlled_decomp(base_rule: DecompositionRule):

    def _condition_fn(base, *_, **__):
        return base_rule.is_applicable(**base.arguments)

    def _resource_fn(base, control_wires, control_values, work_wires, work_wire_type):
        base_counts = base_rule.compute_resources(**base.arguments).gate_counts
        # TODO: we need a better startegy for control values, but for now
        #       we're assuming that half the control values are 0s
        gate_counts = {
            _ctrl_abstract(op, control_wires, work_wires, work_wire_type): count
            for op, count in base_counts.items()
        }
        gate_counts[qp.X] = len(control_wires)
        return gate_counts

    def _wrap_work_wire_spec(base_rule):

        if isinstance(base_rule._work_wire_spec, dict):
            return base_rule._work_wire_spec

        def _wrapper(base, *_, **__):
            return base_rule._work_wire_spec(**base.arguments)

        return _wrapper

    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=_wrap_work_wire_spec(base_rule),
        exact=False,  # TODO: no reliable way to tell whether control values has 0s.
        name=f"controlled({base_rule.name})",
    )
    def _impl(base, control_wires, control_values, work_wires, work_wire_type):

        _cvals = control_values
        _cwires = control_wires
        if qp.capture.enabled() or qp.compiler.active():
            _cvals = qp.math.array(control_values, like="jax")
            _cwires = qp.math.array(control_wires, like="jax")

        @qp.for_loop(0, len(control_values))
        def _x_flips(i):
            qp.cond(qp.math.logical_not(_cvals[i]), qp.X)(_cwires[i])

        _x_flips()
        non_traced_args, traced_args = get_traced_and_non_traced_args(base)
        impl = partial(base_rule._impl, **non_traced_args)
        qp.ctrl(
            impl,
            control=control_wires,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )(**traced_args)
        _x_flips()

    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere base_decomposition is defined as:\n\n"
        + dedent(base_rule._source).strip()
    )
    return _impl


def _flip_control_adjoint_resource(base, control_wires, control_values, work_wires, work_wire_type):
    return {
        Adjoint2(
            qp.ctrl(
                base.base,
                control=control_wires,
                control_values=control_values,
                work_wires=work_wires,
                work_wire_type=work_wire_type,
            )
        ): 1
    }


@register_resources(_flip_control_adjoint_resource)
def flip_control_adjoint(base, control_wires, control_values, work_wires, work_wire_type):
    """Decompose the control of an adjoint by applying control to the base of the adjoint
    and taking the adjoint of the control."""
    qp.adjoint(
        qp.ctrl(
            base.base,
            control=control_wires,
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
    )


def _to_controlled_qu_resource(base, control_wires, control_values, work_wires, work_wire_type):
    num_control_wires = len(control_wires)
    return {
        qp.ControlledQubitUnitary(
            Complex[2, 2],
            wires=Wire[num_control_wires + 1],
            control_values=Bool[num_control_wires],
            work_wires=Wire[len(work_wires)],
            work_wire_type=work_wire_type,
        ): 1
    }


@register_condition(lambda base, *_, **__: base.has_matrix and len(base.wires) == 1)
@register_resources(_to_controlled_qu_resource)
def to_controlled_unitary(base, control_wires, control_values, work_wires, work_wire_type):
    """Convert a controlled operator to a controlled qubit unitary."""
    qp.ControlledQubitUnitary(
        base.matrix(),
        wires=concatenate_wires(control_wires, base.wires),
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


def flip_zero_control(rule: DecompositionRule, name: str = "") -> DecompositionRule:
    """Wraps a decomposition for a controlled operator with X gates to flip zero control wires."""

    def _condition_fn(*args, **kwargs):
        return rule.is_applicable(*args, **kwargs)

    def _resource_fn(base, control_wires, control_values, work_wires, work_wire_type):
        base_resources = rule.compute_resources(
            base,
            control_wires,
            control_values=None,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        gate_counts = base_resources.gate_counts
        base_x_count = gate_counts.get(qp.X, 0)
        gate_counts[qp.X] = base_x_count + len(control_values)
        return gate_counts

    # pylint: disable=protected-access
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=rule._work_wire_spec,
        exact=False,
        name=name or f"flip_zero_ctrl_values({rule.name})",
    )
    def _impl(base, control_wires, control_values, work_wires, work_wire_type):

        _cwires = control_wires
        _cvals = control_values
        if compiler.active() or capture.enabled():
            # We perform the cast on these temporary variables for the sole purpose
            # of indexing into them with tracers. the inner wrapper rule may still
            # depend on control_wires being a Wires object. Ideally we should have a
            # strictly enforced convention for what form the wires argument takes.
            _cwires = math.array(_cwires, like="jax")
            _cvals = math.array(_cvals, like="jax")

        @qp.for_loop(0, len(_cwires))
        def _x_flips(i):
            qp.cond(qp.math.logical_not(_cvals[i]), qp.X)(_cwires[i])

        _x_flips()
        rule(
            base,
            control_wires,
            control_values=None,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
        _x_flips()

    base_source = rule._source
    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere inner_decomp is defined as:\n\n"
        + dedent(base_source).strip()
    )
    return _impl


def _ctrl_single_work_wire_resource(base, control_wires, *_, **__):
    # NOTE: No need to pass work_wire information into the resources below
    # as this rule assumes that no work wires are *explicitly* provided and instead
    # the only work wire comes from allocation.
    return {
        qp.ctrl(base, control=Wire[1], control_values=Bool[1]): 1,
        qp.MultiControlledX(Wire[len(control_wires) + 1]): 2,
    }


@register_condition(lambda base, control_wires, *_, **__: len(control_wires) > 2)
@register_resources(_ctrl_single_work_wire_resource, work_wires={"zeroed": 1})
def _ctrl_single_work_wire(base, control_wires, *_, **__):
    """Implements Lemma 7.11 from https://arxiv.org/abs/quant-ph/9503016."""
    with allocation.allocate(1, state="zero", restored=True) as aux:
        qp.MultiControlledX([*control_wires, aux[0]])
        qp.ctrl(base, control=aux[0])
        qp.MultiControlledX([*control_wires, aux[0]])


ctrl_single_work_wire = flip_zero_control(_ctrl_single_work_wire, name="ctrl_single_work_wire")


def _ctrl_many_zeroed_work_wires_resource(base, control_wires, *_, **__):
    num_control_wires = len(control_wires)
    return {
        qp.TemporaryAND: num_control_wires - 1,
        qp.adjoint(qp.TemporaryAND(Wire[3])): num_control_wires - 1,
        qp.ctrl(base, control=Wire[1], control_values=Bool[1]): 1,
    }


def _ctrl_many_zeroed_work_wires_condition(control_wires, work_wires, work_wire_type, **_):
    return (
        len(control_wires) > 1
        and len(work_wires) >= len(control_wires) - 1
        and work_wire_type == "zeroed"
    )


@register_condition(_ctrl_many_zeroed_work_wires_condition)
@register_resources(_ctrl_many_zeroed_work_wires_resource)
def _ctrl_many_zeroed_work_wires(base, control_wires, work_wires, *_, **__):
    # pylint: disable=import-outside-toplevel
    from pennylane.decomposition_rules.op_math.prod2 import _multi_temporary_and_all_ones

    effective_control = _multi_temporary_and_all_ones(control_wires, work_wires)
    qp.ctrl(base, control=effective_control)
    qp.adjoint(_multi_temporary_and_all_ones)(control_wires, work_wires)


ctrl_many_zeroed_work_wires = flip_zero_control(
    _ctrl_many_zeroed_work_wires, name="ctrl_many_zeroed_work_wires"
)

add_decomps(
    "ControlledOp",
    to_controlled_unitary,
    ctrl_many_zeroed_work_wires,
    ctrl_single_work_wire,
)
