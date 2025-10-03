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
r"""Core resource estimation logic."""
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import singledispatch, wraps

from pennylane.estimator.ops.op_math.symbolic import Adjoint, Controlled, Pow
from pennylane.measurements.measurements import MeasurementProcess
from pennylane.operation import Operation, Operator
from pennylane.queuing import AnnotatedQueue, QueuingManager
from pennylane.wires import Wires
from pennylane.workflow.qnode import QNode

from .resource_config import ResourceConfig
from .resource_mapping import _map_to_resource_op
from .resource_operator import CompressedResourceOp, GateCount, ResourceOperator
from .resources_base import DefaultGateSet, Resources
from .wires_manager import Allocate, Deallocate, WireResourceManager

# pylint: disable=too-many-arguments


def estimate(
    workflow: Callable | ResourceOperator | Resources | QNode,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget: bool = False,
    config: ResourceConfig | None = None,
) -> Resources | Callable[..., Resources]:
    r"""Estimate the quantum resources required by a circuit or operator
    with respect to a given gateset.

    Args:
        workflow (Callable | :class:`~.pennylane.estimator.resource_operator.ResourceOperator` | :class:`~.pennylane.estimator.resources_base.Resources` | QNode): The quantum circuit or operator
            for which to estimate resources.
        gate_set (set[str] | None): A set of names (strings) of the fundamental operators to track
            counts for throughout the quantum workflow.
        zeroed (int | None): Number of zeroed state work wires. Default is ``0``.
        any_state (int | None): Number of work wires in an unknown state. Default is ``0``.
        tight_budget (bool | None): Determines whether extra zeroed state wires may be allocated when they
            exceed the available amount. The default is ``False``.
        config (:class:`~.pennylane.estimator.resource_config.ResourceConfig` | None): A ResourceConfig object which modifies default behaviour in the estimation pipeline.

    Returns:
        :class:`~.pennylane.estimator.resources_base.Resources` | Callable[..., Resources]: The estimated quantum resources required to execute the circuit.

    Raises:
        TypeError: could not obtain resources for workflow of type :code:`type(workflow)`

    **Example**

    The resources of a quantum workflow can be estimated by passing the quantum function describing the
    workflow.

    .. code-block:: python

        from pennylane import estimator as qre

        def my_circuit():
            for w in range(2):
                qre.Hadamard(wires=w)
            qre.CNOT(wires=[0,1])
            qre.RX(wires=0)
            qre.RY(wires=1)
            qre.QFT(num_wires=3, wires=[0, 1, 2])
            return

    The resources for this workflow are then obtained by:

    >>> from pennylane import estimator as qre
    >>> res = qre.estimate(my_circuit)()
    >>> print(res)
    --- Resources: ---
     Total wires: 3
        algorithmic wires: 3
        allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 499
      'T': 484,
      'CNOT': 10,
      'Hadamard': 5

    .. details::
        :title: Usage Details

        :func:`~.estimator.estimate.estimate` also offers mapping functionality, allowing resource estimation for
        programs written with standard PennyLane operators (:class:`~.Operation`).

        .. code-block:: python
        
            import pennylane as qml
            from pennylane import estimator as qre

            dev = qml.device("null.qubit")

            @qml.qnode(dev)
            def circ():
                for w in range(2):
                    qml.Hadamard(wires=w)
                qml.CNOT(wires=[0,1])
                qml.RX(1.23*np.pi, wires=0)
                qml.RY(1.23*np.pi, wires=1)
                qml.QFT(wires=[0, 1, 2])
                return

        .. code-block:: pycon

            >>> res = qre.estimate(circ)()
            >>> print(res)
            --- Resources: ---
             Total wires: 3
                algorithmic wires: 3
                allocated wires: 0
                 zero state: 0
                 any state: 0
             Total gates : 499
              'T': 484,
              'CNOT': 10,
              'Hadamard': 5

    """
    return _estimate_resources_dispatch(workflow, gate_set, zeroed, any_state, tight_budget, config)


@singledispatch
def _estimate_resources_dispatch(
    workflow: Callable | ResourceOperator | Resources | QNode,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget: bool = False,
    config: ResourceConfig | None = None,
) -> Resources | Callable[..., Resources]:
    """Internal singledispatch function for resource estimation."""
    raise TypeError(
        f"Could not obtain resources for workflow of type {type(workflow)}. workflow must be one of Resources, Callable, ResourceOperator, or list"
    )


@_estimate_resources_dispatch.register
def _resources_from_qfunc(
    workflow: Callable,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget: bool = False,
    config: ResourceConfig | None = None,
) -> Callable[..., Resources]:
    """Estimate resources for a quantum function which queues operators"""

    if isinstance(workflow, QNode):
        workflow = workflow.func

    @wraps(workflow)
    def wrapper(*args, **kwargs):
        with AnnotatedQueue() as q:
            workflow(*args, **kwargs)

        wire_manager = WireResourceManager(zeroed, any_state, 0, tight_budget)
        num_algo_qubits = 0
        circuit_wires = []
        for op in q.queue:
            if isinstance(op, (ResourceOperator, Operator, MeasurementProcess)):
                if op.wires:
                    circuit_wires.append(op.wires)
                elif op.num_wires:
                    num_algo_qubits = max(num_algo_qubits, op.num_wires)
            else:
                raise ValueError(
                    f"Queued object '{op}' is not a ResourceOperator or Operator, and cannot be processed."
                )
        num_algo_qubits += len(Wires.all_wires(circuit_wires))
        wire_manager.algo_wires = num_algo_qubits
        # Obtain resources in the gate_set
        compressed_res_ops_list = _ops_to_compressed_reps(q.queue)
        gate_counts = defaultdict(int)
        for cmp_rep_op in compressed_res_ops_list:
            _update_counts_from_compressed_res_op(
                cmp_rep_op, gate_counts, wire_manager=wire_manager, gate_set=gate_set, config=config
            )
        return Resources(
            zeroed=wire_manager.zeroed,
            any_state=wire_manager.any_state,
            algo_wires=wire_manager.algo_wires,
            gate_types=gate_counts,
        )

    return wrapper


@_estimate_resources_dispatch.register
def _resources_from_resource(
    workflow: Resources,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget: bool = None,
    config: ResourceConfig | None = None,
) -> Resources:
    """Further process resources from a Resources object (i.e. a Resources object that
    contains high-level operators can be analyzed with respect to a lower-level gate set)."""

    wire_manager = WireResourceManager(zeroed, any_state, workflow.algo_wires, tight_budget)
    gate_counts = defaultdict(int)
    for cmpr_rep_op, count in workflow.gate_types.items():
        _update_counts_from_compressed_res_op(
            cmpr_rep_op,
            gate_counts,
            wire_manager=wire_manager,
            gate_set=gate_set,
            scalar=count,
            config=config,
        )

    return Resources(
        zeroed=wire_manager.zeroed,
        any_state=wire_manager.any_state,
        algo_wires=wire_manager.algo_wires,
        gate_types=gate_counts,
    )


@_estimate_resources_dispatch.register
def _resources_from_resource_operator(
    workflow: ResourceOperator,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget: bool = None,
    config: ResourceConfig | None = None,
) -> Resources:
    """Extract resources from a resource operator."""
    resources = 1 * workflow
    return _resources_from_resource(
        workflow=resources,
        gate_set=gate_set,
        zeroed=zeroed,
        any_state=any_state,
        tight_budget=tight_budget,
        config=config,
    )


@_estimate_resources_dispatch.register
def _resources_from_pl_ops(
    workflow: Operation,
    gate_set: set[str] | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget: bool = None,
    config: ResourceConfig | None = None,
) -> Resources:
    """Extract resources from a pl operator."""
    workflow = _map_to_resource_op(workflow)
    resources = 1 * workflow
    return _resources_from_resource(
        workflow=resources,
        gate_set=gate_set,
        zeroed=zeroed,
        any_state=any_state,
        tight_budget=tight_budget,
        config=config,
    )


def _update_counts_from_compressed_res_op(
    comp_res_op: CompressedResourceOp,
    gate_counts_dict: dict,
    wire_manager: WireResourceManager,
    gate_set: set[str] | None = None,
    scalar: int = 1,
    config: ResourceConfig | None = None,
) -> None:
    """Modifies the `gate_counts_dict` argument by adding the (scaled) resources of the operator provided.

    Args:
        comp_res_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): operator in compressed representation to extract resources from
        gate_counts_dict (dict): base dictionary to modify with the resource counts
        wire_manager (:class:`~.pennylane.estimator.wires_manager.WireResourceManager`): the `WireResourceManager` that tracks and manages the
            `zeroed`, `any_state`, and `algo_wires` wires.
        gate_set (set[str]): the set of operators to track resources with respect to
        scalar (int | None): optional scalar to multiply the counts. Defaults to 1.
        config (dict | None): additional parameters to specify the resources from an operator. Defaults to :class:`pennylane.estimator.resource_config.ResourceConfig`.
    """
    if gate_set is None:
        gate_set = DefaultGateSet

    if config is None:
        config = ResourceConfig()

    ## Early return if compressed resource operator is already in our defined gate set
    if comp_res_op.name in gate_set:
        gate_counts_dict[comp_res_op] += scalar
        return

    ## Otherwise need to use its resource decomp to extract the resources
    decomp_func, kwargs = _get_decomposition(comp_res_op, config)

    params = {key: value for key, value in comp_res_op.params.items() if value is not None}
    filtered_kwargs = {key: value for key, value in kwargs.items() if key not in params}
    resource_decomp = decomp_func(**params, **filtered_kwargs)
    qubit_alloc_sum = _sum_allocated_wires(resource_decomp)

    for action in resource_decomp:
        if isinstance(action, GateCount):
            _update_counts_from_compressed_res_op(
                action.gate,
                gate_counts_dict,
                wire_manager=wire_manager,
                scalar=scalar * action.count,
                gate_set=gate_set,
                config=config,
            )
            continue

        if isinstance(action, Allocate):
            # When qubits are allocated and deallocate in equal numbers, we allocate and deallocate
            # in series, meaning we don't need to apply the scalar
            if qubit_alloc_sum != 0:
                wire_manager.grab_zeroed(action.num_wires * scalar)
            else:
                wire_manager.grab_zeroed(action.num_wires)
        if isinstance(action, Deallocate):
            if qubit_alloc_sum != 0:
                wire_manager.free_wires(action.num_wires * scalar)
            else:
                wire_manager.free_wires(action.num_wires)

    return


def _sum_allocated_wires(decomp):
    """Sum together the allocated and released wires in a decomposition."""
    s = 0
    for action in decomp:
        if isinstance(action, Allocate):
            s += action.num_wires
        if isinstance(action, Deallocate):
            s -= action.num_wires
    return s


@QueuingManager.stop_recording()
def _ops_to_compressed_reps(
    ops: Iterable[Operator | ResourceOperator],
) -> list[CompressedResourceOp]:
    """Convert the sequence of operators to a list of compressed resource ops.

    Args:
        ops (Iterable[Union[Operator, :class:`~.pennylane.estimator.resource_operator.ResourceOperator`]]): set of operators to convert

    Returns:
        List[CompressedResourceOp]: set of converted compressed resource ops
    """
    cmp_rep_ops = []
    for op in ops:  # Skipping measurement processes here
        if isinstance(op, ResourceOperator):
            cmp_rep_ops.append(op.resource_rep_from_op())
        elif isinstance(op, Operator):
            cmp_rep_ops.append(_map_to_resource_op(op).resource_rep_from_op())

    return cmp_rep_ops


def _get_decomposition(
    comp_res_op: CompressedResourceOp, config: ResourceConfig
) -> tuple[Callable, dict]:
    """
    Selects the appropriate decomposition function and kwargs from a config object.

    This helper function centralizes the logic for choosing a decomposition,
    handling standard, custom, and symbolic operator rules using a mapping.

    Args:
        comp_res_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): The operator to find the decomposition for.
        config (:class:`~.pennylane.estimator.resource_config.ResourceConfig`): The configuration object containing decomposition rules.

    Returns:
        A tuple containing the decomposition function and its associated kwargs.
    """
    op_type = comp_res_op.op_type

    _SYMBOLIC_DECOMP_MAP = {
        Adjoint: "_adj_custom_decomps",
        Controlled: "_ctrl_custom_decomps",
        Pow: "_pow_custom_decomps",
    }

    lookup_op_type = op_type
    custom_decomp_dict = config.custom_decomps

    if op_type in _SYMBOLIC_DECOMP_MAP:
        decomp_attr_name = _SYMBOLIC_DECOMP_MAP[op_type]
        custom_decomp_dict = getattr(config, decomp_attr_name)
        lookup_op_type = comp_res_op.params["base_cmpr_op"].op_type

    kwargs = config.resource_op_precisions.get(lookup_op_type, {})
    decomp_func = custom_decomp_dict.get(lookup_op_type, op_type.resource_decomp)

    return decomp_func, kwargs
