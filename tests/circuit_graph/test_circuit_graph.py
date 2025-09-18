# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the :mod:`pennylane.circuit_graph` module.
"""
# pylint: disable=no-self-use,too-many-arguments,protected-access

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.circuit_graph import CircuitGraph
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.resource import Resources, ResourcesOperation
from pennylane.wires import Wires


@pytest.fixture(name="ops")
def ops_fixture():
    """A fixture of a complex example of operations that depend on previous operations."""
    return [
        qml.RX(0.43, wires=0),
        qml.RY(0.35, wires=1),
        qml.RZ(0.35, wires=2),
        qml.CNOT(wires=[0, 1]),
        qml.Hadamard(wires=2),
        qml.CNOT(wires=[2, 0]),
        qml.PauliX(wires=1),
    ]


@pytest.fixture(name="obs")
def obs_fixture():
    """A fixture of observables to go after the queue fixture."""
    return [
        qml.expval(qml.PauliX(wires=0)),
        qml.expval(qml.Hermitian(np.identity(4), wires=[1, 2])),
    ]


@pytest.fixture(name="circuit")
def circuit_fixture(ops, obs):
    """A fixture of a circuit generated based on the ops and obs fixtures above."""
    return CircuitGraph(ops, obs, Wires([0, 1, 2]))


@pytest.fixture(name="parametrized_circuit_gaussian")
def parametrized_circuit_gaussian_fixture(wires):
    def qfunc(a, b, c, d, e, f):
        qml.Rotation(a, wires=wires[0])
        qml.Rotation(b, wires=wires[1])
        qml.Rotation(c, wires=wires[2])
        qml.Beamsplitter(d, 1, wires=[wires[0], wires[1]])
        qml.Rotation(1, wires=wires[0])
        qml.Rotation(e, wires=wires[1])
        qml.Rotation(f, wires=wires[2])

        return qml.expval(qml.ops.NumberOperator(wires=wires[0]))

    return qfunc


def circuit_measure_max_once():
    """A fixture of a circuit that measures wire 0 once."""
    return qml.expval(qml.PauliX(wires=0))


def circuit_measure_max_twice():
    """A fixture of a circuit that measures wire 0 twice."""
    return qml.expval(qml.PauliZ(wires=0)), qml.probs(wires=0)


def circuit_measure_multiple_with_max_twice():
    """A fixture of a circuit that measures wire 0 twice."""
    return (
        qml.expval(qml.PauliZ(wires=0)),
        qml.probs(wires=[0, 1, 2]),
        qml.var(qml.PauliZ(wires=[1]) @ qml.PauliZ([2])),
    )


# pylint: disable=too-few-public-methods
class CustomOpDepth2(ResourcesOperation):
    num_wires = 3

    def resources(self):
        return Resources(num_wires=self.num_wires, depth=2)


# pylint: disable=too-few-public-methods
class CustomOpDepth3(ResourcesOperation):
    num_wires = 2

    def resources(self):
        return Resources(num_wires=self.num_wires, depth=3)


# pylint: disable=too-few-public-methods
class CustomOpDepth4(ResourcesOperation):
    num_wires = 2

    def resources(self):
        return Resources(num_wires=self.num_wires, depth=4)


# pylint: disable=too-many-public-methods
class TestCircuitGraph:
    """Test conversion of queues to DAGs"""

    def test_no_dependence(self):
        """Test case where operations do not depend on each other.
        This should result in a graph with no edges."""

        ops = [qml.RX(0.43, wires=0), qml.RY(0.35, wires=1)]

        res = CircuitGraph(ops, [], Wires([0, 1])).graph
        assert len(res) == 2
        assert not res.edges()

    def test_dependence(self, ops, obs):
        """Test a more complex example containing operations
        that do depend on the result of previous operations"""

        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))
        graph = circuit.graph
        assert len(graph.node_indexes()) == 9
        assert len(graph.edges()) == 9

        a = {(graph.get_node_data(e[0]), graph.get_node_data(e[1])) for e in graph.edge_list()}

        b = {
            (0, 3),
            (1, 3),
            (2, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (5, 7),
            (5, 8),
            (6, 8),
        }
        assert a == b

    def test_ancestors_and_descendants_example(self, ops, obs):
        """
        Test that the ``ancestors`` and ``descendants`` methods return the expected result.
        """
        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))

        queue = ops + obs

        ancestors = circuit.ancestors([queue[6]])
        ancestors_index = circuit.ancestors_of_indexes([6])
        assert len(ancestors) == len(ancestors_index) == 3
        for o_idx in (0, 1, 3):
            assert queue[o_idx] in ancestors
            assert queue[o_idx] in ancestors_index

        descendants = circuit.descendants([queue[6]])
        descendants_index = circuit.descendants_of_indexes([6])
        assert descendants == [queue[8]]
        assert descendants_index == [queue[8]]

    @pytest.mark.parametrize("sort", [True, False])
    def test_ancestors_and_descendents_repeated_op(self, sort):
        """Test ancestors and descendents raises a ValueError is the requested operation occurs more than once."""

        op = qml.X(0)
        ops = [op, qml.Y(0), op, qml.Z(0), op]
        graph = CircuitGraph(ops, [], qml.wires.Wires([0, 1, 2]))

        with pytest.raises(ValueError, match=r"operator that occurs multiple times."):
            graph.ancestors([op], sort=sort)
        with pytest.raises(ValueError, match=r"operator that occurs multiple times."):
            graph.descendants([op], sort=sort)

    @pytest.mark.parametrize("sort", [True, False])
    def test_ancestors_and_descendents_single_op_error(self, sort):
        """Test ancestors and descendents raises a ValueError is the requested operation occurs more than once."""

        op = qml.Z(0)
        graph = CircuitGraph([op], [], [0, 1, 2])

        with pytest.raises(
            ValueError, match=r"CircuitGraph.ancestors accepts an iterable of operators"
        ):
            graph.ancestors(op, sort=sort)
        with pytest.raises(
            ValueError, match=r"CircuitGraph.descendants accepts an iterable of operators"
        ):
            graph.descendants(op, sort=sort)

    def test_update_node(self, ops, obs):
        """Changing nodes in the graph."""

        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))
        new = qml.RX(0.1, wires=0)
        circuit.update_node(ops[0], new)
        assert circuit.operations[0] is new
        new_mp = qml.var(qml.Y(0))
        circuit.update_node(obs[0], new_mp)
        assert circuit.observables[0] is new_mp

    def test_update_node_error(self, ops, obs):
        """Test that changing nodes in the graph may raise an error."""
        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))
        new = qml.RX(0.1, wires=0)
        new = qml.CNOT(wires=[0, 1])
        with pytest.raises(ValueError):
            circuit.update_node(ops[0], new)

    def test_observables(self, circuit, obs):
        """Test that the `observables` property returns the list of observables in the circuit."""
        assert str(circuit.observables) == str(obs)

    def test_operations(self, circuit, ops):
        """Test that the `operations` property returns the list of operations in the circuit."""
        assert str(circuit.operations) == str(ops)

    def test_observables_in_order_deprecation(self, circuit, obs):
        """Test that a deprecation warning is raised."""
        with pytest.warns(
            PennyLaneDeprecationWarning, match="``CircuitGraph.observables_in_order`` is deprecated"
        ):
            assert str(circuit.observables_in_order) == str(obs)

    def test_operations_in_order_deprecation(self, circuit, ops):
        """Test that a deprecation warning is raised."""
        with pytest.warns(
            PennyLaneDeprecationWarning, match="``CircuitGraph.operations_in_order`` is deprecated"
        ):
            assert str(circuit.operations_in_order) == str(ops)

    def test_ancestors_in_order_deprecation(self):
        """Test that a deprecation warning is raised."""
        op = qml.X(0)
        ops = [op, qml.Y(0), qml.Z(0)]
        graph = CircuitGraph(ops, [], qml.wires.Wires([0, 1, 2]))

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``CircuitGraph.ancestors_in_order`` is deprecated"
        ):
            graph.ancestors_in_order([op])

    def test_descendants_in_order_deprecation(self):
        """Test that a deprecation warning is raised."""
        op = qml.X(0)
        ops = [op, qml.Y(0), qml.Z(0)]
        graph = CircuitGraph(ops, [], qml.wires.Wires([0, 1, 2]))

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``CircuitGraph.descendants_in_order`` is deprecated"
        ):
            graph.descendants_in_order([op])

    def test_op_indices(self, circuit):
        """Test that for the given circuit, this method will fetch the correct operation indices for
        a given wire"""
        op_indices_for_wire_0 = [0, 3, 5, 7]
        op_indices_for_wire_1 = [1, 3, 6, 8]
        op_indices_for_wire_2 = [2, 4, 5, 8]

        assert circuit.wire_indices(0) == op_indices_for_wire_0
        assert circuit.wire_indices(1) == op_indices_for_wire_1
        assert circuit.wire_indices(2) == op_indices_for_wire_2

    @pytest.mark.parametrize("wires", [["a", "q1", 3]])
    def test_layers(self, parametrized_circuit_gaussian, wires):
        """A test of a simple circuit with 3 layers and 6 trainable parameters"""

        dev = qml.device("default.gaussian", wires=wires)
        qnode = qml.QNode(parametrized_circuit_gaussian, dev)
        tape = qml.workflow.construct_tape(qnode)(
            *pnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], requires_grad=True)
        )
        circuit = tape.graph
        layers = circuit.parametrized_layers
        ops = circuit.operations

        assert len(layers) == 3
        assert layers[0].ops == [ops[x] for x in [0, 1, 2]]
        assert layers[0].param_inds == [0, 1, 2]
        assert layers[1].ops == [ops[3]]
        assert layers[1].param_inds == [3]
        assert layers[2].ops == [ops[x] for x in [5, 6]]
        assert layers[2].param_inds == [6, 7]

    def test_iterate_layers_repeat_op(self):
        """Test iterate_parametrized_layers can work when the operation is repeated."""
        op = qml.RX(0.5, 0)
        par_info = [{"op": op, "op_idx": 0, "p_idx": 0}, {"op": op, "op_idx": 2, "p_idx": 0}]
        graph = qml.CircuitGraph(
            [op, qml.X(0), op], [], wires=op.wires, trainable_params={0, 1}, par_info=par_info
        )
        layers = list(graph.iterate_parametrized_layers())

        assert len(layers) == 2

        assert layers[0].pre_ops == []
        assert layers[0].ops == [op]
        assert layers[0].param_inds == (0,)
        assert layers[0].post_ops == [qml.X(0), op]

        assert layers[1].ops == [op]
        assert layers[1].param_inds == (1,)
        assert layers[1].pre_ops == [op, qml.X(0)]
        assert layers[1].post_ops == []

    @pytest.mark.parametrize("wires", [["a", "q1", 3]])
    def test_iterate_layers(self, parametrized_circuit_gaussian, wires):
        """A test of the different layers, their successors and ancestors using a simple circuit"""

        dev = qml.device("default.gaussian", wires=wires)
        qnode = qml.QNode(parametrized_circuit_gaussian, dev)
        tape = qml.workflow.construct_tape(qnode)(
            *pnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], requires_grad=True)
        )
        circuit = tape.graph
        result = list(circuit.iterate_parametrized_layers())

        assert len(result) == 3
        assert set(result[0][0]) == set()
        assert set(result[0][1]) == set(circuit.operations[:3])
        assert result[0][2] == (0, 1, 2)
        assert set(result[0][3]) == set(circuit.operations[3:] + circuit.observables)

        assert set(result[1][0]) == set(circuit.operations[:2])
        assert set(result[1][1]) == {circuit.operations[3]}
        assert result[1][2] == (3,)
        assert set(result[1][3]) == set(circuit.operations[4:6] + circuit.observables[:2])

        assert set(result[2][0]) == set(circuit.operations[:4])
        assert set(result[2][1]) == set(circuit.operations[5:])
        assert result[2][2] == (6, 7)
        assert set(result[2][3]) == set(circuit.observables[1:])

    @pytest.mark.parametrize(
        "circ, expected",
        [
            (circuit_measure_max_once, 1),
            (circuit_measure_max_twice, 2),
            (circuit_measure_multiple_with_max_twice, 2),
        ],
    )
    def test_max_simultaneous_measurements(self, circ, expected):
        """A test for getting the maximum number of measurements on any wire in
        the circuit graph."""

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circ, dev)
        tape = qml.workflow.construct_tape(qnode)()
        circuit = tape.graph
        assert circuit.max_simultaneous_measurements == expected

    def test_str_print(self):
        """Tests if the circuit prints correct."""
        ops = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])]
        obs_w_wires = [qml.measurements.sample(op=None, wires=[0, 1, 2])]

        circuit_w_wires = CircuitGraph(ops, obs_w_wires, wires=Wires([0, 1, 2]))
        expected = """Operations\n==========\nH(0)\nCNOT(wires=[0, 1])\n\nObservables\n===========\nsample(wires=[0, 1, 2])\n"""
        assert str(circuit_w_wires) == expected

    def test_print_contents_deprecation(self):
        """Test that a deprecation warning is raised."""
        ops = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])]
        obs_w_wires = [qml.measurements.sample(op=None, wires=[0, 1, 2])]

        circuit_w_wires = CircuitGraph(ops, obs_w_wires, wires=Wires([0, 1, 2]))

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``CircuitGraph.print_contents`` is deprecated"
        ):
            circuit_w_wires.print_contents()

    tape_depth = (
        ([qml.PauliZ(0), qml.CNOT([0, 1]), qml.RX(1.23, 2)], 2),
        ([qml.X(0)] * 4, 4),
        ([qml.Hadamard(0), qml.CNOT([0, 1]), CustomOpDepth3(wires=[1, 0])], 5),
        (
            [
                qml.RX(1.23, 0),
                qml.RZ(-0.45, 0),
                CustomOpDepth3(wires=[3, 4]),
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.Hadamard(2),
                qml.Hadamard(3),
                qml.Hadamard(4),
                CustomOpDepth2(wires=[1, 2, 3]),
                qml.RZ(-1, 4),
                qml.RX(0.5, 4),
                qml.RX(0.5, 3),
                CustomOpDepth4(wires=[0, 1]),
                qml.CNOT(wires=[3, 4]),
            ],
            10,
        ),
    )

    @pytest.mark.parametrize("ops, true_depth", tape_depth)
    def test_get_depth(self, ops, true_depth):
        """Test that depth is computed correctly for operations that define a custom depth > 1"""
        cg = CircuitGraph(ops, [], wires=[0, 1, 2, 3, 4])
        assert cg.get_depth() == true_depth


def test_has_path():
    """Test has_path and has_path_idx."""

    ops = [qml.X(0), qml.X(3), qml.CNOT((0, 1)), qml.X(1), qml.X(3)]
    graph = CircuitGraph(ops, [], wires=[0, 1, 2, 3, 4, 5])

    assert graph.has_path(ops[0], ops[2])
    assert graph.has_path_idx(0, 2)
    assert not graph.has_path(ops[0], ops[4])
    assert not graph.has_path_idx(0, 4)


def test_has_path_repeated_ops():
    """Test has_path and has_path_idx when an operation is repeated."""

    op = qml.X(0)
    ops = [op, qml.CNOT((0, 1)), op, qml.Y(1)]

    graph = CircuitGraph(ops, [], [0, 1, 2, 3])

    assert graph.has_path_idx(0, 3)
    assert graph.has_path_idx(1, 2)
    with pytest.raises(ValueError, match="does not work with operations that have been repeated. "):
        graph.has_path(op, ops[3])
    with pytest.raises(ValueError, match="does not work with operations that have been repeated. "):
        graph.has_path(ops[1], op)

    # still works if they are the same operation.
    assert graph.has_path(op, op)
