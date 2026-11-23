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


"""Built-in decomposition rules for op_math.controlled."""

from __future__ import annotations

from importlib import import_module
from textwrap import dedent

from pennylane import allocation
from pennylane.core.operator import abstractify
from pennylane.decomposition.decomposition_rule import (
    DecompositionRule,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import adjoint_resource_rep, controlled_resource_rep
from pennylane.typing import Bool, Complex, Wire


def make_controlled_decomp(base_decomposition: DecompositionRule):
    """Create a decomposition rule for the control of a decomposition rule."""

    def _condition_fn(base_params, **_):
        return base_decomposition.is_applicable(**base_params)

    def _resource_fn(
        base_params, num_control_wires, num_zero_control_values, num_work_wires, work_wire_type, **_
    ):
        ops = import_module("pennylane.ops")

        # pylint: disable-next=import-outside-toplevel
        from pennylane.ops.op_math.controlled2 import _ctrl_abstract

        base_resources = base_decomposition.compute_resources(**base_params)
        gate_counts = {
            _ctrl_abstract(
                base_op_rep,
                Wire[num_control_wires],
                Wire[num_work_wires],
                work_wire_type,
            ): count
            for base_op_rep, count in base_resources.gate_counts.items()
        }
        # None of the other gates in gate_counts will be X, because they are all
        # controlled operations. So we can safely set the X gate counts here.
        gate_counts[abstractify(ops.PauliX)] = num_zero_control_values * 2
        return gate_counts

    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=base_decomposition._work_wire_spec,  # pylint: disable=protected-access
        exact=base_decomposition.exact_resources,
        name=f"controlled({base_decomposition.name})",
    )
    # pylint: disable-next=too-many-arguments
    def _impl(*params, wires, control_wires, control_values, work_wires, work_wire_type, base, **_):
        ops = import_module("pennylane.ops")
        zero_control_wires = [
            w for w, val in zip(control_wires, control_values, strict=True) if not val
        ]
        for w in zero_control_wires:
            ops.PauliX(w)
        # We're extracting control wires and base wires from the wires argument instead
        # of directly using control_wires and base.wires, `wires` is properly traced, but
        # `control_wires` and `base.wires` are not.
        ops.ctrl(
            base_decomposition._impl,  # pylint: disable=protected-access
            control=wires[: len(control_wires)],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )(*params, wires=wires[-len(base.wires) :], **base.hyperparameters)
        for w in zero_control_wires:
            ops.PauliX(w)

    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere base_decomposition is defined as:\n\n"
        + dedent(base_decomposition._source).strip()
    )
    return _impl


def flip_zero_control(inner_decomp: DecompositionRule, name: str = "") -> DecompositionRule:
    """Wraps a decomposition for a controlled operator with X gates to flip zero control wires."""

    def _condition_fn(**resource_params):
        new_params = resource_params.copy()
        new_params["num_zero_control_values"] = 0
        return inner_decomp.is_applicable(**new_params)

    def _resource_fn(**resource_params):
        ops = import_module("pennylane.ops")
        new_params = resource_params.copy()
        new_params["num_zero_control_values"] = 0
        inner_resource = inner_decomp.compute_resources(**new_params)
        num_x = resource_params["num_zero_control_values"]
        gate_counts = inner_resource.gate_counts.copy()
        # Add the counts of the flipping X gates to the gate count
        gate_counts[abstractify(ops.X)] = gate_counts.get(abstractify(ops.X), 0) + num_x * 2
        return gate_counts

    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=inner_decomp._work_wire_spec,  # pylint: disable=protected-access
        exact=inner_decomp.exact_resources,
        name=name or f"flip_zero_ctrl_values({inner_decomp.name})",
    )
    def _impl(*params, wires, control_wires, control_values, **kwargs):
        ops = import_module("pennylane.ops")
        zero_control_wires = [
            w for w, val in zip(control_wires, control_values, strict=True) if not val
        ]
        for w in zero_control_wires:
            ops.PauliX(w)
        inner_decomp(
            *params,
            wires=wires,
            control_wires=control_wires,
            control_values=[1] * len(control_wires),  # all control values are 1 now
            **kwargs,
        )
        for w in zero_control_wires:
            ops.PauliX(w)

    base_source = inner_decomp._source
    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere inner_decomp is defined as:\n\n"
        + dedent(base_source).strip()
    )
    return _impl


def _controlled_flip_adjoint_resources(  # pylint: disable=too-many-arguments,unused-argument
    base_class,
    base_params,
    num_control_wires,
    num_zero_control_values,
    num_work_wires,
    work_wire_type,
):
    # base class is adjoint, and the base of the base is the target class
    target_class, target_params = base_params["base_class"], base_params["base_params"]
    inner_rep = controlled_resource_rep(
        base_class=target_class,
        base_params=target_params,
        num_control_wires=num_control_wires,
        num_zero_control_values=num_zero_control_values,
        num_work_wires=num_work_wires,
        work_wire_type=work_wire_type,
    )
    return {adjoint_resource_rep(inner_rep.op_type, inner_rep.params): 1}


@register_resources(_controlled_flip_adjoint_resources, name="flip_adjoint")
def _controlled_flip_adjoint(  # pylint: disable=too-many-arguments
    *_, wires, control_wires, control_values, work_wires, work_wire_type, base, **__
):
    """Decompose the control of an adjoint by applying control to the base of the adjoint
    and taking the adjoint of the control."""
    ops = import_module("pennylane.ops")
    ops.adjoint(
        ops.ctrl(
            base.base,
            control=wires[: len(control_wires)],
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
    )


def _controlled_single_work_wire_resources(base_class, base_params, num_control_wires, **__):
    ops = import_module("pennylane.ops")
    return {
        ops.MultiControlledX(Wire[num_control_wires + 1]): 2,
        controlled_resource_rep(base_class, base_params, 1): 1,
    }


@register_condition(lambda num_control_wires, **_: num_control_wires > 2)
@register_resources(
    _controlled_single_work_wire_resources, work_wires={"zeroed": 1}, name="single_work_wire"
)
# pylint: disable-next=unused-argument
def _controlled_single_work_wire(*params, wires, control_wires, base, **__):
    """Implements Lemma 7.11 from https://arxiv.org/abs/quant-ph/9503016."""
    ops = import_module("pennylane.ops")
    with allocation.allocate(1, state="zero", restored=True) as work_wires:
        ops.MultiControlledX([*control_wires, work_wires[0]])
        ops.ctrl(base, control=work_wires[0])
        ops.MultiControlledX([*control_wires, work_wires[0]])


ctrl_single_work_wire = flip_zero_control(_controlled_single_work_wire, name="single_work_wire")


def _to_controlled_qu_condition(base_class, **__):
    return base_class.has_matrix and base_class.num_wires == 1


def _controlled_to_unitary_resources(  # pylint: disable=unused-argument
    num_control_wires,
    num_zero_control_values,
    num_work_wires,
    work_wire_type,
    **__,
):
    ops = import_module("pennylane.ops")
    return {
        ops.ControlledQubitUnitary(
            Complex[2, 2],
            wires=Wire[num_control_wires + 1],
            control_values=Bool[num_control_wires],
            work_wires=Wire[num_work_wires],
            work_wire_type=work_wire_type,
        ): 1
    }


@register_condition(_to_controlled_qu_condition)
@register_resources(_controlled_to_unitary_resources, name="to_unitary")
def _controlled_to_unitary(*_, wires, control_values, work_wires, work_wire_type, base, **__):
    """Convert a controlled operator to a controlled qubit unitary."""
    ops = import_module("pennylane.ops")
    matrix = base.matrix()
    ops.ControlledQubitUnitary(
        matrix,
        wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
