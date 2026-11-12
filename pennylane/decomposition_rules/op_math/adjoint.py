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

"""Built-in decomposition rules for op_math.adjoint."""

from __future__ import annotations

from textwrap import dedent

import pennylane as qp
from pennylane.decomposition.decomposition_rule import (
    DecompositionRule,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import resource_rep


def make_adjoint_decomp(base_decomposition: DecompositionRule):
    """Create a decomposition rule for the adjoint of a decomposition rule."""

    def _condition_fn(base_class, base_params):  # pylint: disable=unused-argument
        return base_decomposition.is_applicable(**base_params)

    def _resource_fn(base_class, base_params):  # pylint: disable=unused-argument

        # pylint: disable=import-outside-toplevel
        from pennylane.ops.op_math.adjoint2 import _adjoint_abstract

        base_resources = base_decomposition.compute_resources(**base_params)
        return {
            _adjoint_abstract(decomp_op): count
            for decomp_op, count in base_resources.gate_counts.items()
        }

    base_source = base_decomposition._source

    # pylint: disable=protected-access
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=base_decomposition._work_wire_spec,
        exact=base_decomposition.exact_resources,
        name=f"adjoint({base_decomposition.name})",
    )
    def _impl(*params, wires, base):
        # pylint: disable=protected-access
        qp.adjoint(base_decomposition._impl)(*params, wires=wires, **base.hyperparameters)

    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere base_decomposition is defined as:\n\n"
        + dedent(base_source).strip()
    )

    return _impl


def _cancel_adjoint_resource(base_class, base_params):  # pylint:disable=unused-argument
    # The base of a nested adjoint is an adjoint, so the base of the base is the original operator,
    # and the "base_params" of base_params are the resource params of the original operator.
    base_class, base_params = base_params["base_class"], base_params["base_params"]
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_cancel_adjoint_resource)
def cancel_adjoint(*params, wires, base):
    """Decompose the adjoint of the adjoint of an operator."""
    qp.apply(base.base)


def _adjoint_rotation_resource(base_class, base_params):
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_adjoint_rotation_resource)
def adjoint_rotation(phi, wires, base):
    """Decompose the adjoint of a rotation operator by inverting the angle."""
    qp.ops.functions.bind_new_parameters(base, (-phi,))
