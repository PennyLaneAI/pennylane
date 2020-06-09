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
Unit and integration tests for creating the :mod:`pennylane` :attr:`QNode.circuit.hash` attribute.
"""
import pytest
import numpy as np

import pennylane as qml
from pennylane.operation import Tensor
from pennylane.circuit_graph import CircuitGraph
from pennylane.qnodes import BaseQNode
from pennylane.variable import Variable
from pennylane.wires import Wires


class TestCircuitGraphHash:
    """Test the creation of a hash on a CircuitGraph"""

    numeric_queues = [
                        ([
                            qml.RX(0.3, wires=[0])
                        ],
                         [],
                        'RX!0.3![0]|||'
                        ),
                        ([
                            qml.RX(0.3, wires=[0]),
                            qml.RX(0.4, wires=[1]),
                            qml.RX(0.5, wires=[2]),
                        ],
                         [],
                        'RX!0.3![0]RX!0.4![1]RX!0.5![2]|||'
                        )
                     ]

    @pytest.mark.parametrize("queue, observable_queue, expected_string", numeric_queues)
    def test_serialize_numeric_arguments(self, queue, observable_queue, expected_string):
        """Tests that the same hash is created for two circuitgraphs that have numeric arguments."""
        circuit_graph_1 = CircuitGraph(queue + observable_queue, {}, Wires([0, 1, 2]))
        circuit_graph_2 = CircuitGraph(queue + observable_queue, {}, Wires([0, 1, 2]))

        assert circuit_graph_1.serialize() == circuit_graph_2.serialize()
        assert expected_string == circuit_graph_1.serialize()


    variable = Variable(1)

    symbolic_queue = [
                        ([qml.RX(variable, wires=[0])],
                         [],
                        'RX!V1![0]|||'
                        ),

                    ]


    @pytest.mark.parametrize("queue, observable_queue, expected_string", symbolic_queue)
    def test_serialize_symbolic_argument(self, queue, observable_queue, expected_string):
        """Tests that the same hash is created for two circuitgraphs that have symbolic arguments."""
        circuit_graph_1 = CircuitGraph(queue + observable_queue, {}, Wires([0]))
        circuit_graph_2 = CircuitGraph(queue + observable_queue, {}, Wires([0]))

        assert circuit_graph_1.serialize() == circuit_graph_2.serialize()
        assert expected_string == circuit_graph_1.serialize()


    variable = Variable(1)

    symbolic_queue = [
                        ([
                            qml.RX(variable, wires=[0]),
                            qml.RX(0.3, wires=[1]),
                            qml.RX(variable, wires=[2])
                        ],
                         [],
                        'RX!V1![0]RX!0.3![1]RX!V1![2]|||'
                        ),

                        ]


    @pytest.mark.parametrize("queue, observable_queue, expected_string", symbolic_queue)
    def test_serialize_numeric_and_symbolic_argument(self, queue, observable_queue, expected_string):
        """Tests that the same hash is created for two circuitgraphs that have both numeric and symbolic arguments."""

        circuit_graph_1 = CircuitGraph(queue + observable_queue, {}, Wires([0, 1, 2]))
        circuit_graph_2 = CircuitGraph(queue + observable_queue, {}, Wires([0, 1, 2]))

        assert circuit_graph_1.serialize() == circuit_graph_2.serialize()
        assert expected_string == circuit_graph_1.serialize()

    variable = Variable(1)

    many_symbolic_queue = [
                        ([
                            qml.RX(variable, wires=[0]),
                            qml.RX(variable, wires=[1])
                            ],
                         [],
                        'RX!V1![0]' +
                        'RX!V1![1]' +
                        '|||'
                        ),

                        ]

    @pytest.mark.parametrize("queue, observable_queue, expected_string", many_symbolic_queue)
    def test_serialize_symbolic_argument_multiple_times(self, queue, observable_queue, expected_string):
        """Tests that the same hash is created for two circuitgraphs that have the same symbolic argument
        used multiple times."""
        circuit_graph_1 = CircuitGraph(queue + observable_queue, {}, Wires([0, 1]))
        circuit_graph_2 = CircuitGraph(queue + observable_queue, {}, Wires([0, 1]))

        assert circuit_graph_1.serialize() == circuit_graph_2.serialize()
        assert expected_string == circuit_graph_1.serialize()

    variable1 = Variable(1)
    variable2 = Variable(2)

    multiple_symbolic_queue = [
                        ([
                            qml.RX(variable1, wires=[0]),
                            qml.RX(variable2, wires=[1])
                            ],
                         [],
                        'RX!V1![0]' +
                        'RX!V2![1]' +
                        '|||'
                        ),
                        ]

    @pytest.mark.parametrize("queue, observable_queue, expected_string", multiple_symbolic_queue)
    def test_serialize_multiple_symbolic_arguments(self, queue, observable_queue, expected_string):
        """Tests that the same hash is created for two circuitgraphs that have multiple symbolic arguments."""

        circuit_graph_1 = CircuitGraph(queue + observable_queue, {}, Wires([0, 1]))
        circuit_graph_2 = CircuitGraph(queue + observable_queue, {}, Wires([0, 1]))

        assert circuit_graph_1.serialize() == circuit_graph_2.serialize()
        assert expected_string == circuit_graph_1.serialize()


    observable1 = qml.PauliZ(0)
    observable1.return_type = not None

    observable2 = qml.Hermitian(np.array([[1, 0],[0, -1]]), wires=[0])
    observable2.return_type = not None

    observable3 = Tensor(qml.PauliZ(0) @ qml.PauliZ(1))
    observable3.return_type = not None

    numeric_observable_queue = [
                        ([],
                         [observable1],
                        '|||PauliZ[0]'
                        ),
                        (
                         [],
                         [observable2],
                        '|||Hermitian![[ 1  0]\n [ 0 -1]]![0]'
                        ),
                        (
                         [],
                         [observable3],
                        '|||[\'PauliZ\', \'PauliZ\'][0, 1]'
                        )

                     ]

    @pytest.mark.parametrize("queue, observable_queue, expected_string", numeric_observable_queue)
    def test_serialize_numeric_arguments_observables(self, queue, observable_queue, expected_string):
        """Tests that the same hash is created for two circuitgraphs that have identical queues and empty variable_deps."""

        circuit_graph_1 = CircuitGraph(queue + observable_queue, {}, Wires([0, 1]))
        circuit_graph_2 = CircuitGraph(queue + observable_queue, {}, Wires([0, 1]))

        assert circuit_graph_1.serialize() == circuit_graph_2.serialize()
        assert expected_string == circuit_graph_1.serialize()


class TestQNodeCircuitHashIntegration:
    """Test for the circuit hash that is being created for a QNode during evaluation (inside of _construct)"""

    def test_evaluate_circuit_hash_numeric(self):
        """Tests that the circuit hash of identical circuits containing only numeric parameters are equal"""
        dev = qml.device("default.qubit", wires=2)

        a = 0.3
        b = 0.2

        def circuit1():
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2():
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 == circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_symbolic(self, x, y):
        """Tests that the circuit hash of identical circuits containing only symbolic parameters are equal"""
        dev = qml.device("default.qubit", wires=2)

        def circuit1(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 == circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_numeric_and_symbolic(self, x, y):
        """Tests that the circuit hash of identical circuits containing numeric and symbolic parameters are equal"""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.RZ(0.3, wires=[2])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.RZ(0.3, wires=[2])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 == circuit_hash_2

    @pytest.mark.parametrize(
        "a,b",
        zip(np.linspace(0.1, 2 * np.pi, 3), np.linspace(0, 2 * np.pi, 3)),
    )
    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 0, 3), np.linspace(-2 * np.pi, 0, 3)),
    )
    def test_evaluate_circuit_hash_symbolic_assigned_arguments_do_not_matter(self, a, b, x, y):
        """Tests that the circuit hashes of identical circuits where different values are assigned to symbolic parameters are equal"""
        dev = qml.device("default.qubit", wires=2)

        def circuit1(a, b):
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([a, b], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 == circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_numeric_and_symbolic_tensor_return(self, x, y):
        """Tests that the circuit hashes of identical circuits having a tensor product in the return
        statement are equal"""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.RZ(0.3, wires=[2])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.RZ(0.3, wires=[2])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 == circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_same_operation_has_numeric_and_symbolic(self, x, y):
        """Tests that the circuit hashes of identical circuits where one operation has both numeric
        and symbolic arguments are equal"""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 == circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_numeric_and_symbolic_return_type_does_not_matter(self, x, y):
        """Tests that the circuit hashes of identical circuits only differing on their return types are equal"""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        def circuit3(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0) @ qml.PauliX(1))

        node3 = BaseQNode(circuit1, dev)
        node3.evaluate([x, y], {})
        circuit_hash_3 = node3.circuit.hash

        assert circuit_hash_1 == circuit_hash_2 == circuit_hash_3

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_hermitian(self, x, y):
        """Tests that the circuit hashes of identical circuits containing a Hermitian observable are equal"""
        dev = qml.device("default.qubit", wires=3)

        matrix = np.array([[1, 0], [0, 1]])

        def circuit1(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(matrix, wires=[0]) @ qml.PauliX(1))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(matrix, wires=[0]) @ qml.PauliX(1))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 == circuit_hash_2

class TestQNodeCircuitHashDifferentHashIntegration:
    """Tests for checking that different circuit graph hashes are being created for different circuits in a QNode during evaluation (inside of _construct)"""

    def test_evaluate_circuit_hash_numeric_different(self):
        """Tests that the circuit hashes of identical circuits except for one numeric value are different"""
        dev = qml.device("default.qubit", wires=2)

        a = 0.3
        b = 0.2

        def circuit1():
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([], {})
        circuit_hash_1 = node1.circuit.hash

        c = 0.6

        def circuit2():
            qml.RX(c, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 != circuit_hash_2

    def test_evaluate_circuit_hash_numeric_different_operation(self):
        """Tests that the circuit hashes of identical circuits except for one of the operations are different"""
        dev = qml.device("default.qubit", wires=2)

        a = 0.3

        def circuit1():
            qml.RX(a, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2():
            qml.RY(a, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 != circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_numeric_and_symbolic_operation_differs(self, x, y):
        """Tests that the circuit hashes of identical circuits that have numeric and symbolic arguments
        except for one of the operations are different"""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.RX(x, wires=[0])
            qml.RZ(y, wires=[1])          # <-------------------------------------- RZ
            qml.RZ(0.3, wires=[2])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])          # <-------------------------------------- RY
            qml.RZ(0.3, wires=[2])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 != circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_different_return_observable_vs_tensor(self, x, y):
        """Tests that the circuit hashes of identical circuits except for the return statement are different"""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.RZ(0.3, wires=[2])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))                        # <------------- qml.PauliZ(0)

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.RZ(0.3, wires=[2])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))         # <------------- qml.PauliZ(0) @ qml.PauliX(1)

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 != circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_same_operation_has_numeric_and_symbolic_different_order(self, x, y):
        """Tests that the circuit hashes of identical circuits except for the order of numeric and symbolic arguments
        in one of the operations are different."""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.Rot(x, 0.3, y, wires=[0])         # <------------- x, 0.3, y
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.Rot(x, y, 0.3, wires=[0])         # <------------- x, y, 0.3
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 != circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_same_operation_has_numeric_and_symbolic_different_argument(self, x, y):
        """Tests that the circuit hashes of identical circuits except for the numeric value 
        in one of the operations are different."""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.Rot(x, y, 0.3, wires=[0])         # <------------- 0.3
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.Rot(x, y, 0.5, wires=[0])         # <------------- 0.5
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 != circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 2), np.linspace(-2 * np.pi, 2 * np.pi, 2) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_same_operation_has_numeric_and_symbolic_different_wires(self, x, y):
        """Tests that the circuit hashes of identical circuits except for the wires
        in one of the operations are different."""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])                               #<------ wires = [0, 1]
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[1, 0])                               #<------ wires = [1, 0]
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 != circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 2), np.linspace(-2 * np.pi, 2 * np.pi, 2) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_same_operation_has_numeric_and_symbolic_different_wires_in_return(self, x, y):
        """Tests that the circuit hashes of identical circuits except for the wires
        in the return statement are different."""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))         # <----- (0) @ (1)

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(2))         # <----- (0) @ (2)

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 != circuit_hash_2

    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 7), np.linspace(-2 * np.pi, 2 * np.pi, 7) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_numeric_and_symbolic_different_parameter(self, x, y):
        """Tests that the circuit hashes of identical circuits except for the numeric argument of a signle operation
        in the circuits are different"""
        dev = qml.device("default.qubit", wires=3)

        def circuit1(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.RZ(0.3, wires=[2])         # <------------- 0.3
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.RZ(0.5, wires=[2])         # <------------- 0.5
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 != circuit_hash_2


    @pytest.mark.parametrize(
        "x,y",
        zip(np.linspace(-2 * np.pi, 2 * np.pi, 2), np.linspace(-2 * np.pi, 2 * np.pi, 2) ** 2 / 11),
    )
    def test_evaluate_circuit_hash_hermitian_different_matrices(self, x, y):
        """Tests that the circuit hashes of identical circuits except for the matrix argument of the Hermitian observable
        in the return statement are different."""
        dev = qml.device("default.qubit", wires=3)

        matrix_1 = np.array([[1, 0], [0, 1]])
        matrix_2 = np.array([[1, 0], [0, -1]])

        def circuit1(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(matrix_1, wires=[0]) @ qml.PauliX(1))

        node1 = BaseQNode(circuit1, dev)
        node1.evaluate([x, y], {})
        circuit_hash_1 = node1.circuit.hash

        def circuit2(x, y):
            qml.Rot(x, y, 0.3, wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(matrix_2, wires=[0]) @ qml.PauliX(1))

        node2 = BaseQNode(circuit2, dev)
        node2.evaluate([x, y], {})
        circuit_hash_2 = node2.circuit.hash

        assert circuit_hash_1 != circuit_hash_2

    def test_compiled_program_was_stored(self):
        """Test that QVM device stores the compiled program correctly"""
        dev = qml.device("default.qubit", wires=3)

        def circuit(params, wires):
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])

        obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
        obs_list = obs * 6

        qnodes = qml.map(circuit, obs_list, dev)
        qnodes([], parallel=True)

        hashes = set()
        for qnode in qnodes:
            hashes.add(qnode.circuit.hash)

        assert len(hashes) == 1

