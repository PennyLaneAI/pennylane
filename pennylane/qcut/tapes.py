# Copyright 2022 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Functions handling quantum tapes for circuit cutting, and their auxillary functions.
"""

import copy
from itertools import product
from typing import Callable, List, Sequence, Tuple, Union

from networkx import MultiDiGraph

import pennylane as qml
from pennylane import expval
from pennylane.measurements import ExpectationMP, MeasurementProcess, SampleMP
from pennylane.operation import Operator, Tensor
from pennylane.ops.meta import WireCut
from pennylane.pauli import string_to_pauli_word
from pennylane.queuing import WrappedObj
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.wires import Wires

from .utils import MeasureNode, PrepareNode


def tape_to_graph(tape: QuantumTape) -> MultiDiGraph:
    """
    Converts a quantum tape to a directed multigraph.

    .. note::

        This operation is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        tape (QuantumTape): tape to be converted into a directed multigraph

    Returns:
        nx.MultiDiGraph: a directed multigraph that captures the circuit structure
        of the input tape. The nodes of the graph are formatted as ``WrappedObj(op)``, where
        ``WrappedObj.obj`` is the operator.

    **Example**

    Consider the following tape:

    .. code-block:: python

        ops = [
            qml.RX(0.4, wires=0),
            qml.RY(0.9, wires=0),
            qml.CNOT(wires=[0, 1]),
        ]
        measurements = [qml.expval(qml.Z(1))]
        tape = qml.tape.QuantumTape(ops,)

    Its corresponding circuit graph can be found using

    >>> qml.qcut.tape_to_graph(tape)
    <networkx.classes.multidigraph.MultiDiGraph at 0x7fe41cbd7210>
    """
    graph = MultiDiGraph()

    wire_latest_node = {w: None for w in tape.wires}

    for order, op in enumerate(tape.operations):
        _add_operator_node(graph, op, order, wire_latest_node)

    order += 1  # pylint: disable=undefined-loop-variable
    for m in tape.measurements:
        obs = getattr(m, "obs", None)
        if obs is not None and isinstance(obs, (Tensor, qml.ops.Prod)):
            if isinstance(m, SampleMP):
                raise ValueError(
                    "Sampling from tensor products of observables "
                    "is not supported in circuit cutting"
                )

            for o in obs.operands if isinstance(obs, qml.ops.op_math.Prod) else obs.obs:
                m_ = m.__class__(obs=o)
                _add_operator_node(graph, m_, order, wire_latest_node)

        elif isinstance(m, SampleMP) and obs is None:
            for w in m.wires:
                s_ = qml.sample(qml.Projector([1], wires=w))
                _add_operator_node(graph, s_, order, wire_latest_node)
        else:
            _add_operator_node(graph, m, order, wire_latest_node)
            order += 1

    return graph


# pylint: disable=protected-access
def graph_to_tape(graph: MultiDiGraph) -> QuantumTape:
    """
    Converts a directed multigraph to the corresponding :class:`~.QuantumTape`.

    To account for the possibility of needing to perform mid-circuit measurements, if any operations
    follow a :class:`MeasureNode` operation on a given wire then these operations are mapped to a
    new wire.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        graph (nx.MultiDiGraph): directed multigraph to be converted to a tape

    Returns:
        QuantumTape: the quantum tape corresponding to the input graph

    **Example**

    Consider the following circuit:

    .. code-block:: python

        ops = [
            qml.RX(0.4, wires=0),
            qml.RY(0.5, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.qcut.MeasureNode(wires=1),
            qml.qcut.PrepareNode(wires=1),
            qml.CNOT(wires=[1, 0]),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    This circuit contains operations that follow a :class:`~.MeasureNode`. These operations will
    subsequently act on wire ``2`` instead of wire ``1``:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> tape = qml.qcut.graph_to_tape(graph)
    >>> print(tape.draw())
    0: ──RX──────────╭●──────────────╭X─┤  <Z>
    1: ──RY──────────╰X──MeasureNode─│──┤
    2: ──PrepareNode─────────────────╰●─┤

    """

    wires = Wires.all_wires([n.obj.wires for n in graph.nodes])

    ordered_ops = sorted(
        [(order, op.obj) for op, order in graph.nodes(data="order")], key=lambda x: x[0]
    )
    wire_map = {w: w for w in wires}
    reverse_wire_map = {v: k for k, v in wire_map.items()}

    copy_ops = [copy.copy(op) for _, op in ordered_ops if not isinstance(op, MeasurementProcess)]
    copy_meas = [copy.copy(op) for _, op in ordered_ops if isinstance(op, MeasurementProcess)]
    observables = []

    operations_from_graph = []
    measurements_from_graph = []
    for op in copy_ops:
        op = qml.map_wires(op, wire_map=wire_map, queue=False)
        operations_from_graph.append(op)
        if isinstance(op, MeasureNode):
            assert len(op.wires) == 1
            measured_wire = op.wires[0]

            new_wire = _find_new_wire(wires)
            wires += new_wire

            original_wire = reverse_wire_map[measured_wire]
            wire_map[original_wire] = new_wire
            reverse_wire_map[new_wire] = original_wire

    if copy_meas:
        measurement_types = {type(meas) for meas in copy_meas}
        if len(measurement_types) > 1:
            raise ValueError(
                "Only a single return type can be used for measurement nodes in graph_to_tape"
            )
        measurement_type = measurement_types.pop()

        if measurement_type not in {SampleMP, ExpectationMP}:
            raise ValueError(
                "Invalid return type. Only expectation value and sampling measurements "
                "are supported in graph_to_tape"
            )

        for meas in copy_meas:
            meas = qml.map_wires(meas, wire_map=wire_map)
            obs = meas.obs
            observables.append(obs)

            if measurement_type is SampleMP:
                measurements_from_graph.append(meas)

        if measurement_type is ExpectationMP:
            if len(observables) > 1:
                prod_type = qml.prod if qml.operation.active_new_opmath() else Tensor
                measurements_from_graph.append(qml.expval(prod_type(*observables)))
            else:
                measurements_from_graph.append(qml.expval(obs))

    return QuantumScript(ops=operations_from_graph, measurements=measurements_from_graph)


def _add_operator_node(graph: MultiDiGraph, op: Operator, order: int, wire_latest_node: dict):
    """
    Helper function to add operators as nodes during tape to graph conversion.
    """
    node = WrappedObj(op)
    graph.add_node(node, order=order)
    for wire in op.wires:
        if wire_latest_node[wire] is not None:
            parent_node = wire_latest_node[wire]
            graph.add_edge(parent_node, node, wire=wire)
        wire_latest_node[wire] = node


def _find_new_wire(wires: Wires) -> int:
    """Finds a new wire label that is not in ``wires``."""
    ctr = 0
    while ctr in wires:
        ctr += 1
    return ctr


def _create_prep_list():
    """
    Creates a predetermined list for converting PrepareNodes to an associated Operation for use
    within the expand_fragment_tape function.
    """

    def _prep_zero(wire):
        return [qml.Identity(wire)]

    def _prep_one(wire):
        return [qml.X(wire)]

    def _prep_plus(wire):
        return [qml.Hadamard(wire)]

    def _prep_iplus(wire):
        return [qml.Hadamard(wire), qml.S(wires=wire)]

    return [_prep_zero, _prep_one, _prep_plus, _prep_iplus]


PREPARE_SETTINGS = _create_prep_list()


def expand_fragment_tape(
    tape: QuantumTape,
) -> Tuple[List[QuantumTape], List[PrepareNode], List[MeasureNode]]:
    """
    Expands a fragment tape into a sequence of tapes for each configuration of the contained
    :class:`MeasureNode` and :class:`PrepareNode` operations.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        tape (QuantumTape): the fragment tape containing :class:`MeasureNode` and
            :class:`PrepareNode` operations to be expanded

    Returns:
        Tuple[List[QuantumTape], List[PrepareNode], List[MeasureNode]]: the
        tapes corresponding to each configuration and the order of preparation nodes and
        measurement nodes used in the expansion

    **Example**

    Consider the following circuit, which contains a :class:`~.MeasureNode` and
    :class:`~.PrepareNode` operation:

    .. code-block:: python

        ops = [
            qml.qcut.PrepareNode(wires=0),
            qml.RX(0.5, wires=0),
            qml.qcut.MeasureNode(wires=0),
        ]
        tape = qml.tape.QuantumTape(ops)

    We can expand over the measurement and preparation nodes using:

    >>> tapes, prep, meas = qml.qcut.expand_fragment_tape(tape)
    >>> for t in tapes:
    ...     print(qml.drawer.tape_text(t, decimals=1))
    0: ──I──RX(0.5)─┤  <I>  <Z>
    0: ──I──RX(0.5)─┤  <X>
    0: ──I──RX(0.5)─┤  <Y>
    0: ──X──RX(0.5)─┤  <I>  <Z>
    0: ──X──RX(0.5)─┤  <X>
    0: ──X──RX(0.5)─┤  <Y>
    0: ──H──RX(0.5)─┤  <I>  <Z>
    0: ──H──RX(0.5)─┤  <X>
    0: ──H──RX(0.5)─┤  <Y>
    0: ──H──S──RX(0.5)─┤  <I>  <Z>
    0: ──H──S──RX(0.5)─┤  <X>
    0: ──H──S──RX(0.5)─┤  <Y>
    """
    prepare_nodes = [o for o in tape.operations if isinstance(o, PrepareNode)]
    measure_nodes = [o for o in tape.operations if isinstance(o, MeasureNode)]

    wire_map = {mn.wires[0]: i for i, mn in enumerate(measure_nodes)}

    n_meas = len(measure_nodes)
    if n_meas >= 1:
        measure_combinations = qml.pauli.partition_pauli_group(len(measure_nodes))
    else:
        measure_combinations = [[""]]

    tapes = []

    for prepare_settings in product(range(len(PREPARE_SETTINGS)), repeat=len(prepare_nodes)):
        for measure_group in measure_combinations:
            if n_meas >= 1:
                group = [
                    string_to_pauli_word(paulis, wire_map=wire_map) for paulis in measure_group
                ]
            else:
                group = []

            prepare_mapping = {
                id(n): PREPARE_SETTINGS[s] for n, s in zip(prepare_nodes, prepare_settings)
            }

            ops_list = []

            with qml.QueuingManager.stop_recording():
                for op in tape.operations:
                    if isinstance(op, PrepareNode):
                        w = op.wires[0]
                        ops_list.extend(prepare_mapping[id(op)](w))
                    elif not isinstance(op, MeasureNode):
                        ops_list.append(op)
                measurements = _get_measurements(group, tape.measurements)

            qs = qml.tape.QuantumScript(ops=ops_list, measurements=measurements)
            tapes.append(qs)

    return tapes, prepare_nodes, measure_nodes


def _get_measurements(
    group: Sequence[Operator], measurements: Sequence[MeasurementProcess]
) -> List[MeasurementProcess]:
    """Pairs each observable in ``group`` with the circuit ``measurements``.

    Only a single measurement of an expectation value is currently supported
    in ``measurements``.

    Args:
        group (Sequence[Operator]): a collection of observables
        measurements (Sequence[MeasurementProcess]): measurements from the circuit

    Returns:
        List[MeasurementProcess]: the expectation values of ``g @ obs``, where ``g`` is iterated
        over ``group`` and ``obs`` is the observable composing the single measurement
        in ``measurements``
    """
    if len(group) == 0:
        # This ensures the measurements of the original tape are carried over to the
        # following tape configurations in the absence of any MeasureNodes in the fragment
        return measurements

    n_measurements = len(measurements)
    if n_measurements > 1:
        raise ValueError(
            "The circuit cutting workflow only supports circuits with a single output "
            "measurement"
        )
    if n_measurements == 0:
        return [expval(g) for g in group]

    measurement = measurements[0]

    if not isinstance(measurement, ExpectationMP):
        raise ValueError(
            "The circuit cutting workflow only supports circuits with expectation "
            "value measurements"
        )

    obs = measurement.obs

    return [expval(copy.copy(obs) @ g) for g in group]


def _qcut_expand_fn(
    tape: QuantumTape,
    max_depth: int = 1,
    auto_cutter: Union[bool, Callable] = False,
):
    """Expansion function for circuit cutting.

    Expands operations until reaching a depth that includes :class:`~.WireCut` operations.
    """

    for op in tape.operations:
        if isinstance(op, WireCut):
            return tape

    if max_depth > 0:
        return _qcut_expand_fn(tape.expand(), max_depth=max_depth - 1, auto_cutter=auto_cutter)

    if not (auto_cutter is True or callable(auto_cutter)):
        raise ValueError(
            "No WireCut operations found in the circuit. Consider increasing the max_depth value if "
            "operations or nested tapes contain WireCut operations."
        )

    return tape
