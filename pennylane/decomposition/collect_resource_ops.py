# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines an interpreter that extracts a set of resource reps from a plxpr"""

from copy import copy

from pennylane.capture.base_interpreter import FlattenedInterpreter
from pennylane.capture.primitives import adjoint_transform_prim, cond_prim, ctrl_transform_prim

from .resources import adjoint_resource_rep, controlled_resource_rep, resource_rep


class CollectResourceOps(FlattenedInterpreter):
    """Collects a set of unique resource ops from a plxpr."""

    def __init__(self):
        super().__init__()
        self.state = {"ops": set()}

    def interpret_operation(self, op):
        self.state["ops"].add(resource_rep(type(op), **op.resource_params))
        return op


@CollectResourceOps.register_primitive(adjoint_transform_prim)
def _(self, *invals, jaxpr, lazy, n_consts):  # pylint: disable=unused-argument
    """Collect all operations in the base plxpr and create adjoint resource ops with them."""
    consts = invals[:n_consts]
    args = invals[n_consts:]
    child = CollectResourceOps()
    child.eval(jaxpr, consts, *args)
    for op in child.state["ops"]:
        self.state["ops"].add(adjoint_resource_rep(op.op_type, op.params))
    return []


@CollectResourceOps.register_primitive(ctrl_transform_prim)
def _(self, *invals, n_control, jaxpr, n_consts, **params):
    """Collect all operations in the target plxpr and create controlled resource ops with them."""

    consts = invals[:n_consts]
    args = invals[n_consts:-n_control]
    control = invals[-n_control:]
    child = CollectResourceOps()
    child.eval(jaxpr, consts, *args)

    # Extract the resource parameters of this control transform
    control_values = params.get("control_values")
    work_wires = params.get("work_wires")
    num_control_wires = len(control)
    num_zero_control_values = sum(1 for v in control_values if not v) if control_values else 0
    num_work_wires = len(work_wires) if work_wires else 0

    # Create resource reps
    for op in child.state["ops"]:
        self.state["ops"].add(
            controlled_resource_rep(
                op.op_type, op.params, num_control_wires, num_zero_control_values, num_work_wires
            )
        )

    return []


@CollectResourceOps.register_primitive(cond_prim)
def explore_all_branches(self, *invals, jaxpr_branches, consts_slices, args_slice):
    """Handle the cond primitive by a flattened python strategy."""
    n_branches = len(jaxpr_branches)
    conditions = invals[:n_branches]
    args = invals[args_slice]
    outvals = ()
    for _, jaxpr, consts_slice in zip(conditions, jaxpr_branches, consts_slices):
        consts = invals[consts_slice]
        dummy = copy(self).eval(jaxpr, consts, *args)
        # The cond_prim may or may not expect outvals, so we need to check whether
        # the first branch returns something significant. If so, we use the return
        # value of the first branch as the outvals of this cond_prim.
        if dummy and not outvals:
            outvals = dummy
    return outvals
