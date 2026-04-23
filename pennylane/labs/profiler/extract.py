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
r"""
This file contains the core algorithm for extracting the profile.
"""

from collections import defaultdict
from collections.abc import Callable
from functools import singledispatch, wraps

from pennylane.estimator.estimate import _get_resource_decomposition, _ops_to_compressed_reps
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount, ResourceOperator
from pennylane.estimator.resources_base import DefaultGateSet, Resources
from pennylane.queuing import AnnotatedQueue
from pennylane.workflow.qnode import QNode

from pennylane.labs.estimator_beta.resource_config import LabsResourceConfig
from pennylane.labs.estimator_beta.wires_manager.wire_counting import estimate_wires_from_circuit

from .resource_profile import ProfileNode, add_dicts, mul_dict

# pylint: disable=too-many-arguments


def profile(
    workflow: Callable | ResourceOperator | Resources | QNode,
    gate_set: set[str] | None = None,
    zeroed_wires: int = 0,
    any_state_wires: int = 0,
    tight_wires_budget: bool = False,
    config: LabsResourceConfig | None = None,
) -> Resources | Callable[..., Resources]:
    r"""Profile the quantum resources required to implement a circuit or operator in terms of a given gateset.
    """
    return _profile_resources_dispatch(
        workflow, gate_set, zeroed_wires, any_state_wires, tight_wires_budget, config
    )


@singledispatch
def _profile_resources_dispatch(
    workflow: Callable | ResourceOperator | Resources | QNode,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_wires_budget: bool = False,
    config: LabsResourceConfig | None = None,
) -> Resources | Callable[..., Resources]:
    """Internal singledispatch function for resource estimation."""
    raise TypeError(
        f"Could not obtain resources for workflow of type {type(workflow)}. workflow must be one of Resources, Callable, ResourceOperator, or list"
    )


@_profile_resources_dispatch.register
def _profile_from_qfunc(
    workflow: Callable,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_wires_budget: bool = False,
    config: LabsResourceConfig | None = None,
) -> Callable[..., Resources]:
    """Generate a resource profile for a quantum function which queues operators"""
    config = config or LabsResourceConfig()
    gate_set = gate_set or DefaultGateSet

    if isinstance(workflow, QNode):
        workflow = workflow.func

    @wraps(workflow)
    def wrapper(*args, **kwargs):
        with AnnotatedQueue() as q:
            workflow(*args, **kwargs)

        # Obtain resources in the gate_set
        root_node = ProfileNode()
        compressed_res_ops_list = _ops_to_compressed_reps(q.queue)
        
        for cmp_rep_op in compressed_res_ops_list:
            child_node = _extract_gate_counts_from_compressed_res_op(
                cmp_rep_op, gate_set=gate_set, config=config,
            )
            root_node.children.append(child_node)
            add_dicts(root_node.gate_data, child_node.gate_data)  # Updates base dict inplace

        algo_qubits, final_any_state, final_zeroed = estimate_wires_from_circuit(
            circuit_as_lst=q.queue,
            gate_set=gate_set,
            config=config,
            zeroed=zeroed,
            any_state=any_state,
        )

        if tight_wires_budget:
            if (final_zeroed + final_any_state) > (zeroed + any_state):
                raise ValueError(
                    f"Allocated more wires than the prescribed wire budget. Allocated {final_zeroed + final_any_state} qubits with a budget of {zeroed + any_state}"
                )
        
        resources = Resources(
            zeroed_wires=final_zeroed,
            any_state_wires=final_any_state,
            algo_wires=algo_qubits,
            gate_types=root_node.gate_data,
        )
        return (root_node, resources)

    return wrapper


def _extract_gate_counts_from_compressed_res_op(
    comp_res_op: CompressedResourceOp,
    scalar: int = 1,
    gate_set: set[str] | None = None,
    config: LabsResourceConfig | None = None,
):
    """Recurrsive algorithm for building the Profile graph"""
    if gate_set is None:
        gate_set = DefaultGateSet

    if config is None:
        config = LabsResourceConfig()

    # Initialize ProfileNode:
    leaf_node = ProfileNode(comp_res_op, scalar)

    # If its already in the gateset: 
    if comp_res_op.name in gate_set:
        leaf_node.gate_data[comp_res_op] += scalar
        return leaf_node
    
    # Else decompose: 
    resource_decomp = _get_resource_decomposition(comp_res_op, config)

    for action in resource_decomp:
        if isinstance(action, GateCount):
            child_node = _extract_gate_counts_from_compressed_res_op(
                action.gate, scalar=action.count, gate_set=gate_set, config=config,
            )
            leaf_node.children.append(child_node)
            scaled_gate_counts = mul_dict(child_node.gate_data, scalar)
            add_dicts(leaf_node.gate_data, scaled_gate_counts)   # Updates base dict inplace

    return leaf_node
