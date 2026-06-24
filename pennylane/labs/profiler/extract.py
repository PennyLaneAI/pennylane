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

from collections.abc import Callable
from functools import singledispatch, wraps

from pennylane.core.operator import Operation
from pennylane.estimator.estimate import _get_resource_decomposition, _ops_to_compressed_reps
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount, ResourceOperator
from pennylane.estimator.resources_base import DefaultGateSet, Resources
from pennylane.labs.estimator_beta import _map_to_resource_op
from pennylane.labs.estimator_beta.resource_config import LabsResourceConfig
from pennylane.labs.estimator_beta.wires_manager.wire_counting import (
    estimate_wires_from_circuit,
    estimate_wires_from_resources,
)
from pennylane.queuing import AnnotatedQueue
from pennylane.workflow.qnode import QNode

from .resource_profile import ProfileNode, add_dicts

# pylint: disable=too-many-arguments


def profile(
    workflow: Callable | ResourceOperator | Resources | QNode,
    gate_set: set[str] | None = None,
    zeroed_wires: int = 0,
    any_state_wires: int = 0,
    tight_wires_budget: bool = False,
    config: LabsResourceConfig | None = None,
) -> tuple[ProfileNode, Resources] | Callable:
    r"""Profile the quantum resources required to implement a circuit or operator in terms of a given gate set.

    In addition to the aggregated :class:`~.pennylane.estimator.resources_base.Resources`, this function returns the
    root :class:`~.ProfileNode` of a call graph that records how each high-level operator
    decomposes into the target gate set. This tree can be exported with
    :func:`~.export_flame_graph_data` to visualize where the cost of a circuit comes from.

    Args:
        workflow (Callable | ResourceOperator | pennylane.estimator.resources_base.Resources | QNode): the workflow to profile.
            This may be a quantum function (or :class:`~pennylane.QNode`) that queues
            operators, a single :class:`~pennylane.estimator.resource_operator.ResourceOperator`, or a precomputed
            :class:`~.pennylane.estimator.resources_base.Resources` object.
        gate_set (set[str] | None): the set of operator names that the workflow should be
            decomposed into. If ``None``, the estimator's default gate set is used.
        zeroed_wires (int): the number of available auxiliary wires that are guaranteed to be
            in the zero state. Defaults to ``0``.
        any_state_wires (int): the number of available auxiliary wires that may be in any
            state. Defaults to ``0``.
        tight_wires_budget (bool): if ``True``, a :class:`ValueError` is raised when the
            workflow allocates more auxiliary wires than the budget specified by
            ``zeroed_wires`` and ``any_state_wires``. Defaults to ``False``.
        config (LabsResourceConfig | None): the configuration specifying the decomposition
            rules and precisions to use. If ``None``, a default
            :class:`~.estimator_beta.LabsResourceConfig` is used.

    Returns:
        tuple[ProfileNode, pennylane.estimator.resources_base.Resources] | Callable: when ``workflow`` is a
        :class:`~.pennylane.estimator.resource_operator.ResourceOperator` or :class:`~.pennylane.estimator.resources_base.Resources`, a tuple of
        the root :class:`~.ProfileNode` and the aggregated :class:`~.pennylane.estimator.resources_base.Resources` is
        returned. When ``workflow`` is a quantum function or :class:`~pennylane.QNode`, a
        wrapped callable is returned which produces that tuple when called with the workflow's
        arguments.

    Raises:
        TypeError: if ``workflow`` is not one of the supported types.
        ValueError: if ``tight_wires_budget`` is ``True`` and the allocated auxiliary wires
            exceed the supplied budget.

    **Example**

    ``profile`` can be used in just the same way as :func:`~.pennylane.labs.estimator_beta.estimate`:

    >>> import pennylane.labs.estimator_beta as qre
    >>> from pennylane.labs.profiler import profile
    >>> def circuit():
    ...     for w in range(5):
    ...         qre.Hadamard()
    ...         qre.RZ(1e-9)
    ...
    ...     qre.QPE(qre.RX(precision=1e-3), 4)
    ...     qre.QFT(4)
    >>>
    >>> gate_set = {"T", "Hadamard", "CNOT"}
    >>> res_profile, resources = profile(circuit, gate_set)()
    >>> print(resources)
    --- Resources: ---
     Total wires: 5
       algorithmic wires: 5
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 2.041E+3
       'T': 1.972E+3,
       'CNOT': 44,
       'Hadamard': 25

    However, we additionally have access to a resource profile, which can be processed to produce flame graph
    type visualizations.

    >>> from pennylane.labs.profiler import export_flame_graph_data
    >>> extracted_info = export_flame_graph_data(res_profile)
    >>> ids, names, values, parents = extracted_info
    >>>
    >>> import plotly.graph_objects as go  # visualization library
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Icicle(
    ...     ids=ids,
    ...     labels=names,
    ...     parents=parents,
    ...     values=values, # T cost
    ...     branchvalues="total",
    ...     root_color="lightgrey",)
    ... )
    >>> fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    >>> fig.show()

    .. figure:: ../../../_static/profiler_plotly_display.png
        :align: center
        :target: javascript:void(0);

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
) -> tuple[ProfileNode, Resources] | Callable:
    """Internal singledispatch function for resource profiling."""
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
) -> Callable:
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
                cmp_rep_op,
                gate_set=gate_set,
                config=config,
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


@_profile_resources_dispatch.register
def _profile_from_resource(
    workflow: Resources,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_wires_budget: bool = False,
    config: LabsResourceConfig | None = None,
) -> tuple[ProfileNode, Resources] | Callable:
    """Generate a resource profile from a Resources object (i.e. a Resources object that
    contains high-level operators can be analyzed with respect to a lower-level gate set)."""
    config = config or LabsResourceConfig()
    gate_set = gate_set or DefaultGateSet

    root_node = ProfileNode()
    for cmpr_rep_op, count in workflow.gate_types.items():
        child_node = _extract_gate_counts_from_compressed_res_op(
            cmpr_rep_op,
            local_scalar=count,
            cumulative_scalar=count,
            gate_set=gate_set,
            config=config,
        )

        root_node.children.append(child_node)
        add_dicts(root_node.gate_data, child_node.gate_data)  # Updates base dict inplace

    new_any_state, new_zeroed = estimate_wires_from_resources(
        workflow=workflow,
        gate_set=gate_set,
        config=config,
        zeroed=zeroed,
        any_state=any_state,
    )

    if tight_wires_budget:
        if (new_zeroed + new_any_state) > (zeroed + any_state):
            raise ValueError(
                f"Allocated more wires than the prescribed wire budget. Allocated {new_zeroed + new_any_state} qubits with a budget of {zeroed + any_state}"
            )

    resources = Resources(
        zeroed_wires=new_zeroed,
        any_state_wires=new_any_state,
        algo_wires=workflow.algo_wires,
        gate_types=root_node.gate_data,
    )
    return (root_node, resources)


@_profile_resources_dispatch.register
def _profile_from_resource_operator(
    workflow: ResourceOperator,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_wires_budget: bool = False,
    config: LabsResourceConfig | None = None,
) -> tuple[ProfileNode, Resources] | Callable:
    """Extract resource profile from a resource operator."""
    resources = 1 * workflow
    return _profile_from_resource(
        workflow=resources,
        gate_set=gate_set,
        zeroed=zeroed,
        any_state=any_state,
        tight_wires_budget=tight_wires_budget,
        config=config,
    )


@_profile_resources_dispatch.register
def _profile_from_pl_ops(
    workflow: Operation,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_wires_budget: bool = False,
    config: LabsResourceConfig | None = None,
) -> tuple[ProfileNode, Resources] | Callable:
    """Extract resource profile from a pl operator."""
    workflow = _map_to_resource_op(workflow)
    resources = 1 * workflow
    return _profile_from_resource(
        workflow=resources,
        gate_set=gate_set,
        zeroed=zeroed,
        any_state=any_state,
        tight_wires_budget=tight_wires_budget,
        config=config,
    )


def _extract_gate_counts_from_compressed_res_op(
    comp_res_op: CompressedResourceOp,
    local_scalar: int = 1,
    cumulative_scalar: int = 1,
    gate_set: set[str] | None = None,
    config: LabsResourceConfig | None = None,
) -> ProfileNode:
    """Recursive algorithm for building the Profile graph"""
    if gate_set is None:
        gate_set = DefaultGateSet

    if config is None:
        config = LabsResourceConfig()

    # Initialize ProfileNode:
    leaf_node = ProfileNode(comp_res_op, local_scalar)

    # If its already in the gateset:
    if comp_res_op.name in gate_set:
        leaf_node.gate_data[comp_res_op] += cumulative_scalar
        return leaf_node

    # Else decompose:
    resource_decomp = _get_resource_decomposition(comp_res_op, config)

    for action in resource_decomp:
        if isinstance(action, GateCount):
            child_node = _extract_gate_counts_from_compressed_res_op(
                action.gate,
                local_scalar=action.count,
                cumulative_scalar=action.count * cumulative_scalar,
                gate_set=gate_set,
                config=config,
            )
            leaf_node.children.append(child_node)
            add_dicts(leaf_node.gate_data, child_node.gate_data)  # Updates base dict inplace

    return leaf_node
