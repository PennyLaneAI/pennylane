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
r"""Core resource estimation logic."""
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import singledispatch, wraps

from pennylane.estimator.ops.op_math.symbolic import Adjoint, Controlled, Pow
from pennylane.measurements.measurements import MeasurementProcess
from pennylane.operation import Operation, Operator
from pennylane.queuing import AnnotatedQueue, QueuingManager
from pennylane.wires import Wires
from pennylane.workflow.qnode import QNode

from pennylane.estimator.estimate import _update_counts_from_compressed_res_op, _ops_to_compressed_reps, _get_resource_decomposition
from pennylane.estimator.wires_manager import WireResourceManager, Allocate, Deallocate
from pennylane.estimator.resource_config import ResourceConfig
from pennylane.estimator.resource_mapping import _map_to_resource_op
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount, ResourceOperator
from pennylane.estimator.resources_base import DefaultGateSet, Resources
from .wires_manager import estimate_wires_from_circuit

# pylint: disable=too-many-arguments


def estimate(
    workflow: Callable | ResourceOperator | Resources | QNode,
    gate_set: set[str] | None = None,
    zeroed_wires: int = 0,
    any_state_wires: int = 0,
    tight_wires_budget: bool = False,
    config: ResourceConfig | None = None,
) -> Resources | Callable[..., Resources]:
    r"""Estimate the quantum resources required to implement a circuit or operator in terms of a given gateset.

    Args:
        workflow (Callable | :class:`~.pennylane.estimator.resource_operator.ResourceOperator` | :class:`~.pennylane.estimator.resources_base.Resources` | QNode):
            The quantum circuit or operator for which to estimate resources.
        gate_set (set[str] | None): A set of names (strings) of the fundamental operators to count
            throughout the quantum workflow. If not provided, the default gate set will be used,
            i.e., ``{'Toffoli', 'T', 'CNOT', 'X', 'Y', 'Z', 'S', 'Hadamard'}``.
        zeroed_wires (int): Number of work wires pre-allocated in the zeroed state. Default is ``0``.
        any_state_wires (int): Number of work wires pre-allocated in an unknown state. Default is ``0``.
        tight_wires_budget (bool): If True, extra work wires may not be allocated in addition to the pre-allocated ones. The default is ``False``.
        config (:class:`~.pennylane.estimator.resource_config.ResourceConfig` | None): Configurations for the resource estimation pipeline.

    Returns:
        :class:`~.pennylane.estimator.resources_base.Resources` | Callable[..., :class:`~.pennylane.estimator.resources_base.Resources`]:
            The estimated quantum resources required to execute the circuit.

    Raises:
        TypeError: If the ``workflow`` is of an invalid type.

    **Example**

    The resources of a quantum workflow can be estimated by supplying a quantum function describing
    the workflow. The function can be written in terms of resource operators:

    .. code-block:: python

        import pennylane.estimator as qre

        def circuit():
            qre.Hadamard()
            qre.CNOT()
            qre.QFT(num_wires=4)

    >>> res = qre.estimate(circuit)()
    >>> print(res)
    --- Resources: ---
     Total wires: 4
       algorithmic wires: 4
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 816
       'T': 792,
       'CNOT': 19,
       'Hadamard': 5

    The resource estimation can be performed with respect to an alternative gate set:

    >>> res = qre.estimate(circuit, gate_set={"RX", "RZ", "Hadamard", "CNOT"})()
    >>> print(res)
    --- Resources: ---
     Total wires: 4
       algorithmic wires: 4
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 42
       'RZ': 18,
       'CNOT': 19,
       'Hadamard': 5

    .. details::
        :title: Usage Details

        Most PennyLane operators have a corresponding resource operator defined in the ``pennylane.estimator``
        module. The resource operator is a lightweight representation of an operator that contains the
        minimum information required to perform resource estimation. For most basic operators, it is simply
        the type of the operator. For more complex operators and templates, you may be required to provide
        more information as specified in the operator's ``resource_params``, such as the number of wires.

        .. code-block:: python

            import pennylane.estimator as qre

            def circuit():
                qre.CNOT()
                qre.MultiRZ(num_wires=3)
                qre.CNOT()
                qre.MultiRZ(num_wires=3)

        >>> res = qre.estimate(circuit)()
        >>> print(res)
        --- Resources: ---
         Total wires: 3
           algorithmic wires: 3
           allocated wires: 0
             zero state: 0
             any state: 0
         Total gates : 98
           'T': 88,
           'CNOT': 10

        The ``estimate`` function returns a :class:`~pennylane.estimator.resources_base.Resources`
        object, which contains an estimate of the total number of gates (after decomposing to the
        fundamental gate set) and the total number of wires that the gates in this circuit act on
        (i.e., the "algorithmic wires"). When explicit wire labels are not provided, the operators
        are assumed to be overlapping, which may lead to an underestimate. For a more accurate
        estimate of the number of wires used by a circuit, you may optionally provide explicit
        wire labels via the ``wires`` argument:

        .. code-block:: python

            import pennylane.estimator as qre

            def circuit():
                qre.CNOT()
                qre.MultiRZ(wires=[0, 1, 2])
                qre.CNOT()
                qre.MultiRZ(wires=[2, 3, 4])

        >>> res = qre.estimate(circuit)()
        >>> print(res)
        --- Resources: ---
         Total wires: 7
           algorithmic wires: 7
           allocated wires: 0
             zero state: 0
             any state: 0
         Total gates : 98
           'T': 88,
           'CNOT': 10

        For a detailed explanation of the "allocated wires", see the "Dynamic work wire allocation
        in decompositions" section below.

    .. details::
        :title: Dynamic work wire allocation in decompositions

        Some operators require additional auxiliary wires (work wires) to decompose. These wires
        are not part of the operator's definition, so they will be dynamically allocated when
        performing the operator's decomposition. The ``estimate`` function also tracks the usage
        of these dynamically allocated wires.

        .. code-block:: python

            import pennylane.estimator as qre

            def circuit():
                qre.Hadamard()
                qre.CNOT()
                qre.AliasSampling(num_coeffs=3)

        >>> res = qre.estimate(circuit)()
        >>> print(res)
        --- Resources: ---
         Total wires: 123
           algorithmic wires: 2
           allocated wires: 121
             zero state: 58
             any state: 63
         Total gates : 1.150E+3
           'Toffoli': 64,
           'T': 88,
           'CNOT': 589,
           'X': 192,
           'Hadamard': 217

        In the above example, a total of 121 work wires were allocated (in the zeroed state) to
        perform the decomposition of the ``AliasSampling``, 58 of which were restored to the
        original zeroed state before deallocation, and the rest were deallocated in an unknown
        state. You may also pre-allocate work wires:

        >>> res = qre.estimate(circuit, zeroed_wires=150)()
        >>> print(res)
        --- Resources: ---
         Total wires: 152
           algorithmic wires: 2
           allocated wires: 150
             zero state: 87
             any state: 63
         Total gates : 1.150E+3
           'Toffoli': 64,
           'T': 88,
           'CNOT': 589,
           'X': 192,
           'Hadamard': 217

        In this case, you have the option to treat this pre-allocated pool of work wires as the
        only work wires available, by setting ``tight_wires_budget=True``, then an error is
        raised if the required number of wires exceeds the number of pre-allocated wires.

    .. details::
        :title: Estimate the resources of a standard PennyLane circuit

        The ``estimate`` function can also be used to estimate the resources of a standard PennyLane circuit.

        .. code-block:: python

            import pennylane as qml
            import pennylane.estimator as qre

            @qml.qnode(qml.device("default.qubit"))
            def circuit():
                qml.Hadamard(0)
                qml.CNOT(wires=[0, 1])
                qml.QFT(wires=[0, 1, 2, 3])

        >>> res = qre.estimate(circuit)()
        >>> print(res)
        --- Resources: ---
         Total wires: 4
           algorithmic wires: 4
           allocated wires: 0
             zero state: 0
             any state: 0
         Total gates : 816
           'T': 792,
           'CNOT': 19,
           'Hadamard': 5

    """
    return _estimate_resources_dispatch(
        workflow, gate_set, zeroed_wires, any_state_wires, tight_wires_budget, config
    )


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
        zeroed_wires=wire_manager.zeroed,
        any_state_wires=wire_manager.any_state,
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


@_estimate_resources_dispatch.register
def _resources_from_qfunc(
    workflow: Callable,
    gate_set: set[str] | None = None,
    config: ResourceConfig | None = None,
) -> Callable[..., Resources]:
    """Estimate resources for a quantum function which queues operators"""

    if isinstance(workflow, QNode):
        workflow = workflow.func

    @wraps(workflow)
    def wrapper(*args, **kwargs):
        with AnnotatedQueue() as q:
            workflow(*args, **kwargs)

        # Obtain resources in the gate_set
        compressed_res_ops_list = _ops_to_compressed_reps(q.queue)
        gate_counts = defaultdict(int)
        for cmp_rep_op in compressed_res_ops_list:
            _update_counts_from_compressed_res_op(
                cmp_rep_op, gate_counts, gate_set=gate_set, config=config
            )

        algo_qubits, any_state, zeroed = estimate_wires_from_circuit(
            q.queue, gate_set, config,
        )
        return Resources(
            zeroed_wires=zeroed,
            any_state_wires=any_state,
            algo_wires=algo_qubits,
            gate_types=gate_counts,
        )

    return wrapper


def _estimate_wires(
    list_actions: Iterable[GateCount, Allocate, Deallocate],
    scalar: int = 1,
    gate_set: set[str] | None = None,
    config: ResourceConfig | None = None,
) -> Iterable:
    if scalar == 0: return 0, 0, 0
    if config is None: config = ResourceConfig()
    if gate_set is None: gate_set = DefaultGateSet

    total = 0
    max_alloc = 0
    max_dealloc = 0

    for action in list_actions:
        if isinstance(action, GateCount):
            if action.gate.name in gate_set:
                continue

            resource_decomp = _get_resource_decomposition(action.gate, config)
            sub_max_alloc, sub_max_dealloc, sub_total = _estimate_wires(
                resource_decomp, action.count, gate_set, config
            )

            if sub_max_dealloc < 0:
                total += sub_max_dealloc  # sub_max_dealloc < 0
                if total < max_dealloc: max_dealloc = total
            total -= sub_max_dealloc  # reset

            if sub_max_alloc > 0:
                total += sub_max_alloc
                if total > max_alloc: max_alloc = total
            total -= sub_max_alloc  # reset

            total += sub_total
            continue

        elif isinstance(action, Allocate):
            total += action.num_wires

        elif isinstance(action, Deallocate):
            total -= action.num_wires
        
        if total > max_alloc: max_alloc = total
        if total < max_dealloc: max_dealloc = total

    if scalar == 1:
        return max_alloc, max_dealloc, total
    
    if total > 0: max_alloc += (scalar - 1) * total
    if total < 0: max_dealloc += (scalar - 1) * total
    total *= scalar
    
    return max_alloc, max_dealloc, total


