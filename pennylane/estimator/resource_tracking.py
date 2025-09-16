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
r"""Core resource tracking logic."""
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import singledispatch, wraps

from pennylane.operation import Operation
from pennylane.queuing import AnnotatedQueue, QueuingManager
from pennylane.wires import Wires

from .resource_config import ResourceConfig
from .resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
)
from .resources_base import DefaultGateSet, Resources
from .wires_manager import Allocate, Deallocate, WireResourceManager

# pylint: disable=protected-access,too-many-arguments


def map_to_resource_op():  # TODO: Import this function instead when the mapping PR is merged
    pass


def estimate(
    obj: ResourceOperator | Callable | Resources | list,
    gate_set: set | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget: bool = False,
    config: ResourceConfig | None = None,
) -> Resources | Callable:
    r"""Estimate the quantum resources required from a circuit or operation in terms of the gates
    provided in the gateset.

    Args:
        obj (ResourceOperator | Callable | Resources | list): The quantum circuit or operation
            to obtain resources from.
        gate_set (set | None): A set of names (strings) of the fundamental operations to track
            counts for throughout the quantum workflow.
        zeroed (int | None): Number of zeroed state work wires. Default is ``0``.
        any_state (int | None): Number of work wires in an unknown state. Default is ``0``.
        tight_budget (bool | None): Determines whether extra zeroed state wires can be allocated when they
            exceed the available amount. The default is ``False``.
        config (ResourceConfig | None): A ResourceConfig object of additional parameters which sets default values
            when they are not specified on the operator.

    Returns:
        Resources: the quantum resources required to execute the circuit

    Raises:
        TypeError: could not obtain resources for obj of type :code:`type(obj)`

    **Example**

    The resources of a quantum workflow can be tracked by passing the quantum function defining the
    workflow directly into this function.

    .. code-block:: python

        import pennylane.estimator as qre

        def my_circuit():
            for w in range(2):
                qre.Hadamard(wires=w)

            qre.CNOT(wires=[0,1])
            qre.RX(wires=0)
            qre.RY(wires=1)

            qre.QFT(num_wires=3, wires=[0, 1, 2])
            return

    Note that a python function is passed here, not a :class:`~.QNode`. The resources for this
    workflow are then obtained by:

    >>> import pennylane.estimator as qre
    >>> config = qre.ResourceConfig()
    >>> config.set_single_qubit_rot_precision(1e-4)
    >>> res = qre.estimate(
    ...     my_circuit,
    ...     gate_set = qre.DefaultGateSet,
    ...     config=config,
    ... )()
    ...
    >>> print(res)
    --- Resources: ---
    Total qubits: 3
    Total gates : 279
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
     {'Hadamard': 5, 'CNOT': 10, 'T': 264}

    """
    return _estimate_resources_dispatch(obj, gate_set, zeroed, any_state, tight_budget, config)


@singledispatch
def _estimate_resources_dispatch(
    obj: ResourceOperator | Callable | Resources | list,
    gate_set: set | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget: bool = False,
    config: ResourceConfig | None = None,
) -> Resources | Callable:
    """Internal singledispatch function for resource estimation."""
    raise TypeError(
        f"Could not obtain resources for obj of type {type(obj)}. obj must be one of Resources, Callable, ResourceOperator, or list"
    )


@_estimate_resources_dispatch.register
def _resources_from_qfunc(
    obj: Callable,
    gate_set: set | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget=False,
    config: ResourceConfig | None = None,
) -> Callable:
    """Get resources from a quantum function which queues operations"""

    @wraps(obj)
    def wrapper(*args, **kwargs):
        with AnnotatedQueue() as q:
            obj(*args, **kwargs)

        wire_manager = WireResourceManager(zeroed, any_state, 0, tight_budget)
        # Get algorithm wires:
        num_algo_qubits = 0
        circuit_wires = []
        for op in q.queue:
            if isinstance(op, (ResourceOperator, Operation)):
                if op.wires:
                    circuit_wires.append(op.wires)
                elif op.num_wires:
                    num_algo_qubits = max(num_algo_qubits, op.num_wires)
        num_algo_qubits += len(Wires.all_wires(circuit_wires))
        wire_manager.algo_wires = num_algo_qubits  # set the algorithmic qubits in the qubit manager
        # Obtain resources in the gate_set
        compressed_res_ops_lst = _ops_to_compressed_reps(q.queue)
        gate_counts = defaultdict(int)
        for cmp_rep_op in compressed_res_ops_lst:
            _update_counts_from_compressed_res_op(
                cmp_rep_op, gate_counts, wire_mngr=wire_manager, gate_set=gate_set, config=config
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
    obj: Resources,
    gate_set: set | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget=None,
    config: ResourceConfig | None = None,
) -> Resources:
    """Further process resources from a Resources object."""

    wire_manager = WireResourceManager(zeroed, any_state, obj.algo_wires, tight_budget)
    gate_counts = defaultdict(int)
    for cmpr_rep_op, count in obj.gate_types.items():
        _update_counts_from_compressed_res_op(
            cmpr_rep_op,
            gate_counts,
            wire_mngr=wire_manager,
            gate_set=gate_set,
            scalar=count,
            config=config,
        )

    # Update:
    return Resources(
        zeroed=wire_manager.zeroed,
        any_state=wire_manager.any_state,
        algo_wires=wire_manager.algo_wires,
        gate_types=gate_counts,
    )


@_estimate_resources_dispatch.register
def _resources_from_resource_ops(
    obj: ResourceOperator,
    gate_set: set | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget=None,
    config: ResourceConfig | None = None,
) -> Resources:
    """Extract resources from a resource operator."""

    return _resources_from_resource(
        1 * obj,
        gate_set,
        zeroed,
        any_state,
        tight_budget,
        config,
    )


@_estimate_resources_dispatch.register
def _resources_from_pl_ops(
    obj: Operation,
    gate_set: set | None = None,
    zeroed: int = 0,
    any_state: int = 0,
    tight_budget=None,
    config: ResourceConfig | None = None,
) -> Resources:
    """Extract resources from a PennyLane operator."""
    obj = map_to_resource_op(obj)
    return _resources_from_resource(
        1 * obj,
        gate_set,
        zeroed,
        any_state,
        tight_budget,
        config,
    )


def _update_counts_from_compressed_res_op(
    cp_rep: CompressedResourceOp,
    gate_counts_dict,
    wire_mngr: WireResourceManager,
    gate_set: set | None = None,
    scalar: int = 1,
    config: ResourceConfig | None = None,
) -> None:
    """Modifies the `gate_counts_dict` argument by adding the (scaled) resources of the operation provided.

    Args:
        cp_rep (CompressedResourceOp): operation in compressed representation to extract resources from
        gate_counts_dict (dict): base dictionary to modify with the resource counts
        wire_mngr (WireResourceManager): the `WireResourceManager` that tracks and manages the 
            `zeroed`, `any_state`, and `algo_wires` wires.
        gate_set (set): the set of operations to track resources with respect to
        scalar (int | None): optional scalar to multiply the counts. Defaults to 1.
        config (dict | None): additional parameters to specify the resources from an operator. Defaults to ResourceConfig.
    """
    if gate_set is None:
        gate_set = DefaultGateSet

    if config is None:
        config = ResourceConfig()

    ## If op in gate_set add to resources
    if cp_rep.name in gate_set:
        gate_counts_dict[cp_rep] += scalar
        return

    ## Else decompose cp_rep using its resource decomp [cp_rep --> list[GateCounts]] and extract resources
    decomp_func, kwargs = _get_decomposition(cp_rep, config)

    params = {key: value for key, value in cp_rep.params.items() if value is not None}
    filtered_kwargs = {key: value for key, value in kwargs.items() if key not in params}
    resource_decomp = decomp_func(**params, **filtered_kwargs)
    qubit_alloc_sum = _sum_allocated_wires(resource_decomp)

    for action in resource_decomp:
        if isinstance(action, GateCount):
            _update_counts_from_compressed_res_op(
                action.gate,
                gate_counts_dict,
                wire_mngr=wire_mngr,
                scalar=scalar * action.count,
                gate_set=gate_set,
                config=config,
            )
            continue

        if isinstance(action, Allocate):
            if qubit_alloc_sum != 0 and scalar > 1:
                wire_mngr.grab_zeroed(action.num_wires * scalar)
            else:
                wire_mngr.grab_zeroed(action.num_wires)
        if isinstance(action, Deallocate):
            if qubit_alloc_sum != 0 and scalar > 1:
                wire_mngr.free_wires(action.num_wires * scalar)
            else:
                wire_mngr.free_wires(action.num_wires)

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
    ops: Iterable[Operation | ResourceOperator],
) -> list[CompressedResourceOp]:
    """Convert the sequence of operations to a list of compressed resource ops.

    Args:
        ops (Iterable[Union[Operation, ResourceOperator]]): set of operations to convert

    Returns:
        List[CompressedResourceOp]: set of converted compressed resource ops
    """
    cmp_rep_ops = []
    for op in ops:  # Skipping measurement processes here
        if isinstance(op, ResourceOperator):
            cmp_rep_ops.append(op.resource_rep_from_op())

        if isinstance(op, Operation):
            cmp_rep_ops.append(map_to_resource_op(op).resource_rep_from_op())

    return cmp_rep_ops


def _get_decomposition(
    cp_rep: CompressedResourceOp, config: ResourceConfig
) -> tuple[Callable, dict]:
    """
    Selects the appropriate decomposition function and kwargs from a config object.

    This helper function centralizes the logic for choosing a decomposition,
    handling standard, custom, and symbolic operator rules using a mapping.

    Args:
        cp_rep (CompressedResourceOp): The operator to find the decomposition for.
        config (ResourceConfig): The configuration object containing decomposition rules.

    Returns:
        A tuple containing the decomposition function and its associated kwargs.
    """
    op_type = cp_rep.op_type
    _SYMBOLIC_DECOMP_MAP = {
        # TODO: Uncomment this when symbolic resource operators are merged.
        # ResourceAdjoint: "_adj_custom_decomps",
        # ResourceControlled: "_ctrl_custom_decomps",
        # ResourcePow: "_pow_custom_decomps",
    }

    if op_type in _SYMBOLIC_DECOMP_MAP:
        decomp_attr_name = _SYMBOLIC_DECOMP_MAP[op_type]
        custom_decomp_dict = getattr(config, decomp_attr_name)

        base_op_type = cp_rep.params["base_cmpr_op"].op_type
        kwargs = config.resource_op_precisions.get(base_op_type, {})
        decomp_func = custom_decomp_dict.get(base_op_type, op_type.resource_decomp)

    else:
        kwargs = config.resource_op_precisions.get(op_type, {})
        decomp_func = config._custom_decomps.get(op_type, op_type.resource_decomp)

    return decomp_func, kwargs
