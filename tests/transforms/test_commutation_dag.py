# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Unit tests for CommutationDAG
"""
import pytest
from collections import OrderedDict
from pennylane.wires import Wires
import pennylane.numpy as np
import pennylane as qml


class TestCommutationDAG:
    """Commutation DAG tests."""

    def test_return_dag(self):
        def circuit():
            qml.PauliZ(wires=0)

        dag_object = qml.transforms.commutation_dag(circuit)()
        dag = dag_object.graph

        assert len(dag) != 0

    def test_dag_invalid_argument(self):
        """Assert error raised when input is neither a tape, QNode, nor quantum function"""

        with pytest.raises(ValueError, match="Input is not a tape, QNode, or quantum function"):
            qml.transforms.commutation_dag(qml.PauliZ(0))()

    def test_dag_wrong_function(self):
        """Assert error raised when input function is not a quantum function"""

        def test_function(x):
            return x

        with pytest.raises(ValueError, match="Function contains no quantum operation"):
            qml.transforms.commutation_dag(test_function)(1)

    def test_dag_transform_simple_dag_function(self):
        """Test a simple DAG on 1 wire with a quantum function."""

        def circuit():
            qml.PauliZ(wires=0)
            qml.PauliX(wires=0)

        dag = qml.transforms.commutation_dag(circuit)()

        a = qml.PauliZ(wires=0)
        b = qml.PauliX(wires=0)

        nodes = [a, b]
        edges = [(0, 1, {"commute": False})]

        assert dag.get_node(0).op.compare(a)
        assert dag.get_node(1).op.compare(b)
        assert dag.get_edge(0, 1) == {0: {"commute": False}}
        assert dag.get_edge(0, 2) is None
        assert dag.observables == []
        for i, node in enumerate(dag.get_nodes()):
            assert node[1].op.compare(nodes[i])
        for i, edge in enumerate(dag.get_edges()):
            assert edges[i] == edge

    def test_dag_transform_simple_dag_tape(self):
        """Test a simple DAG on 1 wire with a quantum tape."""
        with qml.tape.QuantumTape() as tape:
            qml.PauliZ(wires=0)
            qml.PauliX(wires=0)

        dag = qml.transforms.commutation_dag(tape)()

        a = qml.PauliZ(wires=0)
        b = qml.PauliX(wires=0)

        nodes = [a, b]
        edges = [(0, 1, {"commute": False})]

        assert dag.get_node(0).op.compare(a)
        assert dag.get_node(1).op.compare(b)
        assert dag.get_edge(0, 1) == {0: {"commute": False}}
        assert dag.get_edge(0, 2) is None
        assert dag.observables == []
        for i, node in enumerate(dag.get_nodes()):
            assert node[1].op.compare(nodes[i])
        for i, edge in enumerate(dag.get_edges()):
            assert edges[i] == edge

    def test_dag_transform_simple_dag_function_custom_wire(self):
        """Test a simple DAG on 2 wires with a quantum function and custom wires."""

        def circuit():
            qml.PauliZ(wires="a")
            qml.PauliX(wires="c")

        dag = qml.transforms.commutation_dag(circuit)()

        a = qml.PauliZ(wires=0)
        b = qml.PauliX(wires=1)

        nodes = [a, b]
        edges = [(0, 1, {"commute": False})]

        assert dag.get_node(0).op.compare(a)
        assert dag.get_node(1).op.compare(b)
        assert dag.get_edge(0, 1) is None
        assert dag.get_edge(0, 2) is None
        assert dag.observables == []
        for i, node in enumerate(dag.get_nodes()):
            assert node[1].op.compare(nodes[i])
        for i, edge in enumerate(dag.get_edges()):
            assert edges[i] == edge

    def test_dag_transform_simple_dag_qnode(self):
        """Test a simple DAG on 1 wire with a qnode."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(wires=0)
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliX(wires=0))

        dag = qml.transforms.commutation_dag(circuit)()

        a = qml.PauliZ(wires=0)
        b = qml.PauliX(wires=0)

        nodes = [a, b]
        edges = [(0, 1, {"commute": False})]

        assert dag.get_node(0).op.compare(a)
        assert dag.get_node(1).op.compare(b)
        assert dag.get_edge(0, 1) == {0: {"commute": False}}
        assert dag.get_edge(0, 2) is None
        assert dag.observables[0].return_type.__repr__() == "expval"
        assert dag.observables[0].name == "PauliX"
        assert dag.observables[0].wires.tolist() == [0]
        for i, node in enumerate(dag.get_nodes()):
            assert node[1].op.compare(nodes[i])
        for i, edge in enumerate(dag.get_edges()):
            assert edges[i] == edge

    def test_dag_pattern(self):
        "Test a the DAG and its attributes for a more complicated circuit."

        def circuit():
            qml.CNOT(wires=[3, 0])
            qml.PauliX(wires=4)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[4, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[1, 2])
            qml.PauliX(wires=1)
            qml.CNOT(wires=[1, 0])
            qml.PauliX(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 3])

        dag = qml.transforms.commutation_dag(circuit)()

        wires = [3, 0, 4, 2, 1]
        consecutive_wires = Wires(range(len(wires)))
        wires_map = OrderedDict(zip(wires, consecutive_wires))

        nodes = [
            qml.CNOT(wires=[3, 0]),
            qml.PauliX(wires=4),
            qml.PauliZ(wires=0),
            qml.CNOT(wires=[4, 2]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[3, 4]),
            qml.CNOT(wires=[1, 2]),
            qml.PauliX(wires=1),
            qml.CNOT(wires=[1, 0]),
            qml.PauliX(wires=1),
            qml.CNOT(wires=[1, 2]),
            qml.CNOT(wires=[0, 3]),
        ]

        for node in nodes:
            node._wires = Wires([wires_map[wire] for wire in node.wires.tolist()])

        edges = [
            (0, 2, {"commute": False}),
            (0, 4, {"commute": False}),
            (1, 3, {"commute": False}),
            (2, 8, {"commute": False}),
            (3, 5, {"commute": False}),
            (4, 6, {"commute": False}),
            (5, 11, {"commute": False}),
            (6, 7, {"commute": False}),
            (7, 8, {"commute": False}),
            (8, 9, {"commute": False}),
            (8, 11, {"commute": False}),
            (9, 10, {"commute": False}),
        ]

        direct_successors = [[2, 4], [3], [8], [5], [6], [11], [7], [8], [9, 11], [10], [], []]
        successors = [
            [2, 4, 6, 7, 8, 9, 10, 11],
            [3, 5, 11],
            [8, 9, 10, 11],
            [5, 11],
            [6, 7, 8, 9, 10, 11],
            [11],
            [7, 8, 9, 10, 11],
            [8, 9, 10, 11],
            [9, 10, 11],
            [10],
            [],
            [],
        ]
        direct_predecessors = [[], [], [0], [1], [0], [3], [4], [6], [2, 7], [8], [9], [5, 8]]
        predecessors = [
            [],
            [],
            [0],
            [1],
            [0],
            [1, 3],
            [0, 4],
            [0, 4, 6],
            [0, 2, 4, 6, 7],
            [0, 2, 4, 6, 7, 8],
            [0, 2, 4, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ]

        assert dag.observables == []

        for i in range(0, 12):
            assert dag.get_node(i).op.name == nodes[i].name
            assert dag.get_node(i).op.wires == nodes[i].wires
            assert dag.direct_successors(i) == direct_successors[i]
            assert dag.get_node(i).successors == successors[i] == dag.successors(i)
            assert dag.direct_predecessors(i) == direct_predecessors[i]
            assert dag.get_node(i).predecessors == predecessors[i] == dag.predecessors(i)

        for i, edge in enumerate(dag.get_edges()):
            assert edges[i] == edge

    @pytest.mark.autograd
    def test_dag_parameters_autograd(self):
        "Test a the DAG and its attributes for autograd parameters."

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.PauliZ(0))

        x = np.array([np.pi / 4, np.pi / 3, np.pi / 2], requires_grad=False)

        get_dag = qml.transforms.commutation_dag(circuit)
        dag = get_dag(x[0], x[1], x[2])

        nodes = [
            qml.RX(x[0], wires=0),
            qml.RX(x[1], wires=0),
            qml.CNOT(wires=[1, 2]),
            qml.RY(x[1], wires=1),
            qml.Hadamard(wires=2),
            qml.CRZ(x[2], wires=[2, 0]),
            qml.RY(-x[1], wires=1),
        ]

        edges = [
            (0, 5, {"commute": False}),
            (1, 5, {"commute": False}),
            (2, 3, {"commute": False}),
            (2, 4, {"commute": False}),
            (2, 6, {"commute": False}),
            (4, 5, {"commute": False}),
        ]

        direct_successors = [[5], [5], [3, 4, 6], [], [5], [], []]
        successors = [[5], [5], [3, 4, 5, 6], [], [5], [], []]
        direct_predecessors = [[], [], [], [2], [2], [0, 1, 4], [2]]
        predecessors = [[], [], [], [2], [2], [0, 1, 2, 4], [2]]

        for i in range(0, 7):
            assert dag.get_node(i).op.name == nodes[i].name
            assert dag.get_node(i).op.data == nodes[i].data
            assert dag.get_node(i).op.wires == nodes[i].wires
            assert dag.direct_successors(i) == direct_successors[i]
            assert dag.get_node(i).successors == successors[i] == dag.successors(i)
            assert dag.direct_predecessors(i) == direct_predecessors[i]
            assert dag.get_node(i).predecessors == predecessors[i] == dag.predecessors(i)

        for i, edge in enumerate(dag.get_edges()):
            assert edges[i] == edge

    @pytest.mark.tf
    def test_dag_parameters_tf(self):
        "Test a the DAG and its attributes for tensorflow parameters."
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.PauliZ(0))

        x = tf.Variable([np.pi / 4, np.pi / 3, np.pi / 2], dtype=tf.float64)

        get_dag = qml.transforms.commutation_dag(circuit)
        dag = get_dag(x[0], x[1], x[2])

        nodes = [
            qml.RX(x[0], wires=0),
            qml.RX(x[1], wires=0),
            qml.CNOT(wires=[1, 2]),
            qml.RY(x[1], wires=1),
            qml.Hadamard(wires=2),
            qml.CRZ(x[2], wires=[2, 0]),
            qml.RY(-x[1], wires=1),
        ]

        edges = [
            (0, 5, {"commute": False}),
            (1, 5, {"commute": False}),
            (2, 3, {"commute": False}),
            (2, 4, {"commute": False}),
            (2, 6, {"commute": False}),
            (4, 5, {"commute": False}),
        ]

        direct_successors = [[5], [5], [3, 4, 6], [], [5], [], []]
        successors = [[5], [5], [3, 4, 5, 6], [], [5], [], []]
        direct_predecessors = [[], [], [], [2], [2], [0, 1, 4], [2]]
        predecessors = [[], [], [], [2], [2], [0, 1, 2, 4], [2]]

        for i in range(0, 7):
            assert dag.get_node(i).op.name == nodes[i].name
            assert dag.get_node(i).op.data == nodes[i].data
            assert dag.get_node(i).op.wires == nodes[i].wires
            assert dag.direct_successors(i) == direct_successors[i]
            assert dag.get_node(i).successors == successors[i] == dag.successors(i)
            assert dag.direct_predecessors(i) == direct_predecessors[i]
            assert dag.get_node(i).predecessors == predecessors[i] == dag.predecessors(i)

        for i, edge in enumerate(dag.get_edges()):
            assert edges[i] == edge

    @pytest.mark.torch
    def test_dag_parameters_torch(self):
        "Test a the DAG and its attributes for torch parameters."
        import torch

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.PauliZ(0))

        x = torch.tensor([np.pi / 4, np.pi / 3, np.pi / 2], requires_grad=False)

        get_dag = qml.transforms.commutation_dag(circuit)
        dag = get_dag(x[0], x[1], x[2])

        nodes = [
            qml.RX(x[0], wires=0),
            qml.RX(x[1], wires=0),
            qml.CNOT(wires=[1, 2]),
            qml.RY(x[1], wires=1),
            qml.Hadamard(wires=2),
            qml.CRZ(x[2], wires=[2, 0]),
            qml.RY(-x[1], wires=1),
        ]

        edges = [
            (0, 5, {"commute": False}),
            (1, 5, {"commute": False}),
            (2, 3, {"commute": False}),
            (2, 4, {"commute": False}),
            (2, 6, {"commute": False}),
            (4, 5, {"commute": False}),
        ]

        direct_successors = [[5], [5], [3, 4, 6], [], [5], [], []]
        successors = [[5], [5], [3, 4, 5, 6], [], [5], [], []]
        direct_predecessors = [[], [], [], [2], [2], [0, 1, 4], [2]]
        predecessors = [[], [], [], [2], [2], [0, 1, 2, 4], [2]]

        for i in range(0, 7):
            assert dag.get_node(i).op.name == nodes[i].name
            assert dag.get_node(i).op.data == nodes[i].data
            assert dag.get_node(i).op.wires == nodes[i].wires
            assert dag.direct_successors(i) == direct_successors[i]
            assert dag.get_node(i).successors == successors[i] == dag.successors(i)
            assert dag.direct_predecessors(i) == direct_predecessors[i]
            assert dag.get_node(i).predecessors == predecessors[i] == dag.predecessors(i)

        for i, edge in enumerate(dag.get_edges()):
            assert edges[i] == edge

    @pytest.mark.jax
    def test_dag_parameters_jax(self):
        "Test a the DAG and its attributes for jax parameters."

        from jax import numpy as jnp

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.PauliZ(0))

        x = jnp.array([np.pi / 4, np.pi / 3, np.pi / 2], dtype=jnp.float64)
        get_dag = qml.transforms.commutation_dag(circuit)
        dag = get_dag(x[0], x[1], x[2])

        nodes = [
            qml.RX(x[0], wires=0),
            qml.RX(x[1], wires=0),
            qml.CNOT(wires=[1, 2]),
            qml.RY(x[1], wires=1),
            qml.Hadamard(wires=2),
            qml.CRZ(x[2], wires=[2, 0]),
            qml.RY(-x[1], wires=1),
        ]

        edges = [
            (0, 5, {"commute": False}),
            (1, 5, {"commute": False}),
            (2, 3, {"commute": False}),
            (2, 4, {"commute": False}),
            (2, 6, {"commute": False}),
            (4, 5, {"commute": False}),
        ]

        direct_successors = [[5], [5], [3, 4, 6], [], [5], [], []]
        successors = [[5], [5], [3, 4, 5, 6], [], [5], [], []]
        direct_predecessors = [[], [], [], [2], [2], [0, 1, 4], [2]]
        predecessors = [[], [], [], [2], [2], [0, 1, 2, 4], [2]]

        for i in range(0, 7):
            assert dag.get_node(i).op.name == nodes[i].name
            assert dag.get_node(i).op.data == nodes[i].data
            assert dag.get_node(i).op.wires == nodes[i].wires
            assert dag.direct_successors(i) == direct_successors[i]
            assert dag.get_node(i).successors == successors[i] == dag.successors(i)
            assert dag.direct_predecessors(i) == direct_predecessors[i]
            assert dag.get_node(i).predecessors == predecessors[i] == dag.predecessors(i)

        for i, edge in enumerate(dag.get_edges()):
            assert edges[i] == edge
