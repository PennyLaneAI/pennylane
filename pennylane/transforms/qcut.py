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
This module provides the circuit cutting functionality that allows large
circuits to be distributed across multiple devices.
"""

from types import NoneType
import warnings
from typing import List, Tuple, Dict, Any, Union
from dataclasses import dataclass
from networkx import MultiDiGraph
from pennylane.measure import MeasurementProcess
from pennylane.operation import Operation, Operator, Tensor
from pennylane.ops.qubit.non_parametric_ops import WireCut
from pennylane.tape import QuantumTape


class MeasureNode(Operation):
    """Placeholder node for measurement operations"""

    num_wires = 1
    grad_method = None


class PrepareNode(Operation):
    """Placeholder node for state preparations"""

    num_wires = 1
    grad_method = None


def replace_wire_cut_node(node: WireCut, graph: MultiDiGraph):
    """
    Replace a :class:`~.WireCut` node in the graph with a :class:`~.MeasureNode`
    and :class:`~.PrepareNode`.

    Args:
        node (WireCut): the  :class:`~.WireCut` node to be replaced with a :class:`~.MeasureNode`
            and :class:`~.PrepareNode`
        graph (MultiDiGraph): the graph containing the node to be replaced

    **Example**

    Consider the following circuit with a manually-placed wire cut:

    .. code-block:: python

        from pennylane.transforms import qcut

        wire_cut = qml.WireCut(wires=0)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.apply(wire_cut)
            qml.RY(0.5, wires=0)
            qml.expval(qml.PauliZ(0))

    We can find the circuit graph and remove the wire cut node using:

    >>> graph = qcut.tape_to_graph(tape)
    >>> qcut.replace_wire_cut_node(wire_cut, graph)
    """
    predecessors = graph.pred[node]
    successors = graph.succ[node]

    predecessor_on_wire = {}
    for op, data in predecessors.items():
        for d in data.values():
            wire = d["wire"]
            predecessor_on_wire[wire] = op

    successor_on_wire = {}
    for op, data in successors.items():
        for d in data.values():
            wire = d["wire"]
            successor_on_wire[wire] = op

    order = graph.nodes[node]["order"]
    graph.remove_node(node)

    for wire in node.wires:
        predecessor = predecessor_on_wire.get(wire, None)
        successor = successor_on_wire.get(wire, None)

        meas = MeasureNode(wires=wire)
        prep = PrepareNode(wires=wire)

        # We are introducing a degeneracy in the order of the measure and prepare nodes
        # here but the order can be inferred as MeasureNode always precedes
        # the corresponding PrepareNode
        graph.add_node(meas, order=order)
        graph.add_node(prep, order=order)

        graph.add_edge(meas, prep, wire=wire)

        if predecessor is not None:
            graph.add_edge(predecessor, meas, wire=wire)
        if successor is not None:
            graph.add_edge(prep, successor, wire=wire)


def replace_wire_cut_nodes(graph: MultiDiGraph):
    """
    Replace each :class:`~.WireCut` node in the graph with a
    :class:`~.MeasureNode` and :class:`~.PrepareNode`.

    Args:
        graph (MultiDiGraph): The graph containing the :class:`~.WireCut` nodes
            to be replaced

    **Example**

    Consider the following circuit with manually-placed wire cuts:

    .. code-block:: python

        from pennylane.transforms import qcut

        wire_cut_0 = qml.WireCut(wires=0)
        wire_cut_1 = qml.WireCut(wires=1)
        multi_wire_cut = qml.WireCut(wires=[0, 1])

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.apply(wire_cut_0)
            qml.RY(0.5, wires=0)
            qml.apply(wire_cut_1)
            qml.CNOT(wires=[0, 1])
            qml.apply(multi_wire_cut)
            qml.RZ(0.6, wires=1)
            qml.expval(qml.PauliZ(0))

    We can find the circuit graph and remove all the wire cut nodes using:

    >>> graph = qcut.tape_to_graph(tape)
    >>> qcut.replace_wire_cut_nodes(graph)

    """
    for op in list(graph.nodes):
        if isinstance(op, WireCut):
            replace_wire_cut_node(op, graph)


def _add_operator_node(graph: MultiDiGraph, op: Operator, order: int, wire_latest_node: dict):
    """
    Helper function to add operators as nodes during tape to graph conversion.
    """
    graph.add_node(op, order=order)
    for wire in op.wires:
        if wire_latest_node[wire] is not None:
            parent_op = wire_latest_node[wire]
            graph.add_edge(parent_op, op, wire=wire)
        wire_latest_node[wire] = op


def tape_to_graph(tape: QuantumTape) -> MultiDiGraph:
    """
    Converts a quantum tape to a directed multigraph.

    Args:
        tape (QuantumTape): tape to be converted into a directed multigraph

    Returns:
        graph (MultiDiGraph): a directed multigraph that captures the circuit
        structure of the input tape

    **Example**

    Consider the following tape:

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.9, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(1))

    Its corresponding circuit graph can be found using

    >>> tape_to_graph(tape)
    <networkx.classes.multidigraph.MultiDiGraph at 0x7fe41cbd7210>
    """
    graph = MultiDiGraph()

    wire_latest_node = {w: None for w in tape.wires}

    for order, op in enumerate(tape.operations):
        _add_operator_node(graph, op, order, wire_latest_node)

    order += 1  # pylint: disable=undefined-loop-variable
    for m in tape.measurements:
        obs = getattr(m, "obs", None)
        if obs is not None and isinstance(obs, Tensor):
            for o in obs.obs:
                m_ = MeasurementProcess(m.return_type, obs=o)

                _add_operator_node(graph, m_, order, wire_latest_node)

        else:
            _add_operator_node(graph, m, order, wire_latest_node)
            order += 1

    return graph


@dataclass(frozen=True)
class CutConfig:
    """
    Container wrapping partitioning parameters for passing to the automatic cutter.
    """

    k: int
    imbalance: float
    fragment_wires: List[int] = None
    fragment_gates: List[int] = None

    @staticmethod
    def infer_imbalance(k, num_wires, num_gates, max_wires, max_gates, imbalance_tolerance=None):
        """Helper function for determining best imbalance limit."""
        avg_fragment_wires = (num_wires - 1) // k + 1
        avg_fragment_gates = (num_gates - 1) // k + 1
        wire_imbalance = max_wires / avg_fragment_wires - 1
        gate_imbalance = max_gates / avg_fragment_gates - 1
        imbalance = min(gate_imbalance, wire_imbalance)
        if imbalance_tolerance is not None:
            imbalance = min(imbalance, imbalance_tolerance)
        return imbalance

    @staticmethod
    def validate_params(
        num_tape_wires,
        num_tape_gates,
        max_device_wires,
        max_device_gates,
        fragment_wires,
        fragment_gates,
    ):
        """Helper parameter checker."""
        assert isinstance(num_tape_wires, int)
        assert isinstance(num_tape_gates, int)
        assert isinstance(max_device_wires, int)
        assert isinstance(max_device_gates, int)
        if fragment_gates is not None:
            assert isinstance(fragment_gates, (list, tuple))
            assert all(isinstance(i, int) for i in fragment_gates)
            assert all(i <= max_device_gates for i in fragment_gates)
        if fragment_wires is not None:
            assert isinstance(fragment_wires, (list, tuple))
            assert all(isinstance(i, int) for i in fragment_wires)
            assert all(i <= max_device_wires for i in fragment_wires)
        if fragment_wires is not None and fragment_gates is not None:
            assert len(fragment_wires) == len(fragment_gates)

    @classmethod
    def infer_default_configs(  # pylint: disable=too-many-arguments
        cls,
        num_tape_wires,
        num_tape_gates,
        max_device_wires,
        max_device_gates=None,
        min_device_wires=None,
        min_device_gates=None,
        k_lower=None,
        k_upper=None,
        imbalance_tolerance=None,
        fragment_wires=None,
        fragment_gates=None,
    ):
        """
        Helper function for deriving the best default partitioning parameters for the automatic
        circuit cutter. Arguments can be either user provided or inferred from devices.

        Args:
            num_tape_wires (int): Number of wires in the circuit tape to be partitioned.
            num_tape_gates (int): Number of gates in the circuit tape to be partitioned.
            max_device_wires (int): Number of wires for the largest available device.
            max_device_gates (int): Maximum allowed circuit depth for the deepest available device.
                Optional, defaults to unlimited depth.
            min_device_wires (int): Number of wires for the smallest available device.
                Optional, defaults to `max_device_wires`.
            min_device_gates (int): Maximum allowed circuit depth for the shallowest available device.
                Optional, defaults to `max_device_gates`.
            k_lower (int): Lowest number of fragments to attempt for the automatic circuit cutter.
                Optional, defaults to the scenario where each fragment is executed on the largest
                available device, satisfying both wire and gate constraints.
                If supplied, can be higher than said default if the the desired k is known.
            k_upper (int): Largest number of fragments to attempt for the automatic circuit cutter.
                Optional, defaults to the scenario where each fragment is executed on the smallest
                available device, satisfying both wire and gate constraints.
                If supplied, can be higher than said default to encourage exploration.
            imbalance_tolerance (float): Maximum allowed imbalance for all partition attempts.
                Can be used to globally override the imbalance constraint parameter.
                Optional, defaults to a loose upper bound to avoid being too restrictive.
            fragment_wires (List[int]): User provided predetermined list of fragment wire limits.
                If supplied, k will be derived from it and exploration of other ks will not be made.
            fragment_gates (List[int]): User provided predetermined list of fragment gate limits.
                If supplied, k will be derived from it and exploration of other ks will not be made.

        Returns:
            cut_configs (List[CutConfig]): A list of cut configurations for passing to the automatic
                circuit cutter to attempt.

        """

        # Assumes unlimited gate depth if not supplied.
        max_device_gates = max_device_gates or num_tape_gates

        cls.validate_params(
            num_tape_wires,
            num_tape_gates,
            max_device_wires,
            max_device_gates,
            fragment_wires,
            fragment_gates,
        )

        # Assumes same number of wires/gates across all devices if min_device_* not provided.
        min_device_wires = min_device_wires or max_device_wires
        min_device_gates = min_device_gates or max_device_gates

        # The lower bound of k corresponds to executing each fragment on the largest available device.
        k_lb = 1 + min(
            (num_tape_wires - 1) // max_device_wires,  # wire limited
            (num_tape_gates - 1) // max_device_gates,  # gate limited
        )
        # The upper bound of k corresponds to executing each fragment on the smallest available device.
        k_ub = 1 + max(
            (num_tape_wires - 1) // min_device_wires,  # wire limited
            (num_tape_gates - 1) // min_device_gates,  # gate limited
        )

        # The global imbalance tolerance, if not given, defaults to a very loose upper bound:
        imbalance_tolerance = imbalance_tolerance or k_ub
        if not (isinstance(imbalance_tolerance, (float, int)) and imbalance_tolerance >= 0):
            raise ValueError(
                "The global `imbalance_tolerance` is expected to be a non-negative number, "
                f"got {type(imbalance_tolerance)} with value {imbalance_tolerance}."
            )

        if fragment_gates is None and fragment_wires is None:
            cut_configs = []

            # k_lower, when supplied by a user, can be higher than k_lb if the the desired k is known:
            k_lower = int(k_lower or k_lb)
            # k_upper, when supplied by a user, can be higher than k_ub to encourage exploration:
            k_upper = int(k_upper or k_ub)

            if k_lower < k_lb:
                warnings.warn(
                    f"The provided `k_lower={k_lower}` is less than the lowest allowed value, "
                    f"will override and set `k_lower={k_lb}`."
                )
                k_lower = k_lb
            if k_upper < k_lower:
                warnings.warn(
                    f"The provided `k_upper={k_upper}` is less than `k_lower={k_lower}`, "
                    f"will override and set `k_upper={k_lower}`. "
                    "Note this will result in only one partitioning attempt, with the number of fragments "
                    f"fixed at {k_lower}, rather than exploring into higher number of fragments."
                )
                k_upper = k_lower

            HIGH_NUM_FRAGMENTS = 20
            if k_lower > HIGH_NUM_FRAGMENTS:
                warnings.warn(
                    f"The attempted number of fragments seems high with lower bound at {k_lower}, "
                    "are you sure?"
                )

            # Prepare the list of ks to explore:
            ks = list(range(k_lower, k_upper + 1))

            HIGH_PARTITION_ATTEMPS = 20
            if len(ks) > HIGH_PARTITION_ATTEMPS:
                warnings.warn(
                    f"The numer of partition attempts seems high ({len(ks)}), are you sure?"
                )
        else:
            # When the by-fragment wire and/or gate limits are supplied, derive k and imbalance and
            # return a single partition config.
            ks = [len(fragment_wires)]

        for k in ks:
            imbalance = cls.infer_imbalance(
                k,
                num_tape_wires,
                num_tape_gates,
                max_device_wires if fragment_wires is None else max(fragment_wires),
                max_device_gates if fragment_gates is None else max(fragment_gates),
                imbalance_tolerance,
            )
            cut_configs.append(cls(k, imbalance, fragment_wires, fragment_gates))

        return cut_configs


@dataclass(frozen=True)
class CutSpec:
    """
    Container wrapping partitioning results returned from the automatic cutter.
    NOTE: Incomplete.
    """

    trial_id: Tuple[int, int]
    num_wires: int
    num_gates: int
    config: CutConfig
    fragments: Dict[Any, Any]
    raw_cuts: List[Any]  # i.e. including "hyper-wires"
    raw_cost: Union[int, float]
    # mode: str = "wire"

    @property
    def k(self) -> int:
        """Property: the number of partitions."""
        return self.config.k

    @property
    def cuts(self):
        """Property: the list of cut edges."""
        # removes hyperedges, assuming they are at the end of raw_cuts.
        return self.raw_cuts[: (self.num_gates if self.mode == "gate" else self.num_wires)]

    @property
    def num_cuts(self) -> int:
        """Property: the number of cut edges."""
        return len(self.cuts)

    @property
    def cost(self) -> int:
        """Property: cost of circuit cut"""
        # TODO: should be different for gate cut
        return self.num_cuts / 2

    @property
    def fragment_gates(self):
        """Property: dict of fragment sizes {fragment_id: fragment_size}"""
        return {k: len(gates) for k, gates in self.fragments.items()}

    @staticmethod
    def collect_fragment_wires(gates):
        """Collects wires from a fragment"""
        # TODO: implement this.
        return set([w for g in gates for w in g.wires])

    @property
    def fragment_wires(self):
        """Property: dict of number of wires by fragment {fragment_id: wire_size}"""
        return {k: len(self.collect_wires(gates)) for k, gates in self.fragments.items()}
