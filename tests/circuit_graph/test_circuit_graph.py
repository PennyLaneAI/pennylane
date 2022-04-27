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

import pytest
import numpy as np

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.circuit_graph import CircuitGraph
from pennylane.wires import Wires


@pytest.fixture
def ops():
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


@pytest.fixture
def obs():
    """A fixture of observables to go after the queue fixture."""
    return [
        qml.expval(qml.PauliX(wires=0)),
        qml.expval(qml.Hermitian(np.identity(4), wires=[1, 2])),
    ]


@pytest.fixture
def circuit(ops, obs):
    """A fixture of a circuit generated based on the queue and obs fixtures above."""
    circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))
    return circuit


@pytest.fixture
def parameterized_circuit(wires):
    def qfunc(a, b, c, d, e, f):
        qml.Rotation(a, wires=wires[0]),
        qml.Rotation(b, wires=wires[1]),
        qml.Rotation(c, wires=wires[2]),
        qml.Beamsplitter(d, 1, wires=[wires[0], wires[1]])
        qml.Rotation(1, wires=wires[0]),
        qml.Rotation(e, wires=wires[1]),
        qml.Rotation(f, wires=wires[2]),

        return [
            qml.expval(qml.ops.NumberOperator(wires=wires[0])),
            qml.expval(qml.ops.NumberOperator(wires=wires[1])),
            qml.expval(qml.ops.NumberOperator(wires=wires[2])),
        ]

    return qfunc


def circuit_measure_max_once():
    """A fixture of a circuit that measures wire 0 once."""
    return qml.expval(qml.PauliX(wires=0))


def circuit_measure_max_twice():
    """A fixture of a circuit that measures wire 0 twice."""
    return qml.expval(qml.PauliX(wires=0)), qml.probs(wires=0)


def circuit_measure_multiple_with_max_twice():
    """A fixture of a circuit that measures wire 0 twice."""
    return (
        qml.expval(qml.PauliX(wires=0)),
        qml.probs(wires=[0, 1, 2]),
        qml.var(qml.PauliX(wires=[1]) @ qml.PauliZ([2])),
    )


def measurement_process_equality(mp1, mp2):
    """Helper function to check for equality between two MeasurementProcess objects."""
    same_return_type = mp1.return_type == mp2.return_type
    same_obs = mp1.obs == mp2.obs
    same_wires = mp1._wires == mp2._wires
    same_eigs = mp1._eigvals == mp2._eigvals
    same_name = mp1.name == mp2.name
    same_data = mp1.data == mp2.data

    if same_return_type and same_obs and same_wires and same_eigs and same_name and same_data:
        return True

    return False


def lst_mp_equality(lst1, lst2):
    """Wrapper function to check if two lists are identical when the
    lists contain MeasurementProcess objects."""
    assert len(lst1) == len(lst2)
    lsts_same = True

    for index in range(len(lst1)):
        if not isinstance(lst1[index], type(lst2[index])):  # if they are different types
            return False  # then return false

        elif isinstance(lst1[index], qml.measurements.MeasurementProcess):
            lsts_same = lsts_same and measurement_process_equality(lst1[index], lst2[index])

        else:
            lsts_same = lsts_same and (lst1[index] == lst2[index])

    return lsts_same


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

        queue = ops + obs

        # all ops should be nodes in the graph
        for k in queue:
            assert k in graph.nodes()

        # all nodes in the graph should be ops
        # for k in graph.nodes:
        for k in graph.nodes():
            assert k is queue[k.queue_idx]
        a = set((graph.get_node_data(e[0]), graph.get_node_data(e[1])) for e in graph.edge_list())
        b = set(
            (queue[a], queue[b])
            for a, b in [
                (0, 3),
                (1, 3),
                (2, 4),
                (3, 5),
                (3, 6),
                (4, 5),
                (5, 7),
                (5, 8),
                (6, 8),
            ]
        )
        assert a == b

    def test_ancestors_and_descendants_example(self, ops, obs):
        """
        Test that the ``ancestors`` and ``descendants`` methods return the expected result.
        """
        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))

        queue = ops + obs

        ancestors = circuit.ancestors([queue[6]])
        assert len(ancestors) == 3
        for o_idx in (0, 1, 3):
            assert queue[o_idx] in ancestors

        descendants = circuit.descendants([queue[6]])
        assert descendants == set([queue[8]])

    def test_in_topological_order_example(self, ops, obs):
        """
        Test ``_in_topological_order`` method returns the expected result.
        """
        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))

        to = circuit._in_topological_order(ops)

        to_expected = [
            qml.RZ(0.35, wires=[2]),
            qml.Hadamard(wires=[2]),
            qml.RY(0.35, wires=[1]),
            qml.RX(0.43, wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.PauliX(wires=[1]),
            qml.CNOT(wires=[2, 0]),
        ]

        assert str(to) == str(to_expected)

    def test_update_node(self, ops, obs):
        """Changing nodes in the graph."""

        circuit = CircuitGraph(ops, obs, Wires([0, 1, 2]))
        new = qml.RX(0.1, wires=0)
        circuit.update_node(ops[0], new)
        assert circuit.operations[0] is new

    def test_observables(self, circuit, obs):
        """Test that the `observables` property returns the list of observables in the circuit."""
        assert str(circuit.observables) == str(obs)

    def test_operations(self, circuit, ops):
        """Test that the `operations` property returns the list of operations in the circuit."""
        assert str(circuit.operations) == str(ops)

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
    def test_layers(self, parameterized_circuit, wires):
        """A test of a simple circuit with 3 layers and 6 trainable parameters"""

        dev = qml.device("default.gaussian", wires=wires)
        qnode = qml.QNode(parameterized_circuit, dev)
        qnode(*pnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], requires_grad=True))
        circuit = qnode.qtape.graph
        layers = circuit.parametrized_layers
        ops = circuit.operations

        assert len(layers) == 3
        assert layers[0].ops == [ops[x] for x in [0, 1, 2]]
        assert layers[0].param_inds == [0, 1, 2]
        assert layers[1].ops == [ops[3]]
        assert layers[1].param_inds == [3]
        assert layers[2].ops == [ops[x] for x in [5, 6]]
        assert layers[2].param_inds == [6, 7]

    @pytest.mark.parametrize("wires", [["a", "q1", 3]])
    def test_iterate_layers(self, parameterized_circuit, wires):
        """A test of the different layers, their successors and ancestors using a simple circuit"""

        dev = qml.device("default.gaussian", wires=wires)
        qnode = qml.QNode(parameterized_circuit, dev)
        qnode(*pnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], requires_grad=True))
        circuit = qnode.qtape.graph
        result = list(circuit.iterate_parametrized_layers())

        assert len(result) == 3
        assert set(result[0][0]) == set([])
        assert set(result[0][1]) == set(circuit.operations[:3])
        assert result[0][2] == (0, 1, 2)
        assert set(result[0][3]) == set(circuit.operations[3:] + circuit.observables)

        assert set(result[1][0]) == set(circuit.operations[:2])
        assert set(result[1][1]) == set([circuit.operations[3]])
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
        qnode()
        circuit = qnode.qtape.graph
        assert circuit.max_simultaneous_measurements == expected

    def test_grid_when_sample_no_wires(self):
        """A test to ensure the sample operation applies to all wires when
        none are explicitly provided."""

        ops = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])]
        obs_no_wires = [qml.measurements.sample(op=None, wires=None)]
        obs_w_wires = [qml.measurements.sample(op=None, wires=[0, 1, 2])]

        circuit_no_wires = CircuitGraph(ops, obs_no_wires, wires=Wires([0, 1, 2]))
        circuit_w_wires = CircuitGraph(ops, obs_w_wires, wires=Wires([0, 1, 2]))

        sample_w_wires_op = qml.measurements.sample(op=None, wires=[0, 1, 2])
        expected_grid = {
            0: [ops[0], ops[1], sample_w_wires_op],
            1: [ops[1], sample_w_wires_op],
            2: [sample_w_wires_op],
        }

        for key in range(3):
            lst_w_wires = circuit_w_wires._grid[key]
            lst_no_wires = circuit_no_wires._grid[key]
            lst_expected = expected_grid[key]

            assert lst_mp_equality(lst_no_wires, lst_w_wires)
            assert lst_mp_equality(lst_no_wires, lst_expected)
