# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests for conversion and helper methods used in `pennylane.transforms.zx`
"""
import sys
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.exceptions import QuantumFunctionError
from pennylane.tape import QuantumScript
from pennylane.transforms import TransformError

pyzx = pytest.importorskip("pyzx")
pytestmark = pytest.mark.external
supported_operations = [
    qml.X(wires=0),
    qml.Y(wires=0),
    qml.Z(wires=0),
    qml.S(wires=0),
    qml.T(wires=0),
    qml.Hadamard(wires=0),
    qml.SWAP(wires=[0, 1]),
    qml.CNOT(wires=[0, 1]),
    qml.CY(wires=[0, 1]),
    qml.CZ(wires=[0, 1]),
    qml.CH(wires=[0, 1]),
    qml.RX(0.3, wires=0),
    qml.RY(0.3, wires=0),
    qml.RZ(0.3, wires=0),
    qml.PhaseShift(0.3, wires=0),
    qml.CRX(0.3, wires=[0, 1]),
    qml.CRY(0.3, wires=[0, 1]),
    qml.CRZ(0.3, wires=[0, 1]),
    qml.Toffoli(wires=[0, 1, 2]),
    qml.CCZ(wires=[0, 1, 2]),
]

decompose_phases = [True, False]
qscript = [True, False]


def test_import_pyzx_error(monkeypatch):
    """Test that a ModuleNotFoundError is raised by the to_zx function
    when the pyzx external package is not installed."""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "pyzx", None)

        with pytest.raises(ModuleNotFoundError, match="The `pyzx` package is required."):
            qml.transforms.to_zx(qml.PauliX(wires=0))

        with pytest.raises(ModuleNotFoundError, match="The `pyzx` package is required."):
            qml.transforms.to_zx(QuantumScript([qml.PauliX(wires=0), qml.PauliZ(wires=1)]))


class TestConvertersZX:
    """Test converters to_zx and from_zx."""

    def test_invalid_argument(self):
        """Assert error raised when input is neither a tape, QNode, nor quantum function"""
        with pytest.raises(
            TransformError,
            match="Input is not an Operator, tape, QNode, or quantum function",
        ):
            _ = qml.transforms.to_zx(None)

    @pytest.mark.parametrize("script", qscript)
    @pytest.mark.parametrize("operation", supported_operations)
    def test_supported_operations(self, operation, script):
        """Test to convert the script to a ZX graph and back for supported operations."""

        I = qml.math.eye(2 ** len(operation.wires))

        if script:
            qs = QuantumScript([operation])
        else:
            qs = operation

        matrix_qscript = qml.matrix(qs, wire_order=qs.wires)

        zx_g = qml.transforms.to_zx(qs)
        matrix_zx = zx_g.to_matrix()

        assert isinstance(zx_g, pyzx.graph.graph_s.GraphS)
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_qscript, qml.math.conj(matrix_zx.T))
        # Remove global phase
        if not np.allclose(mat_product[0, 0], 1.0):
            mat_product /= mat_product[0, 0]

        assert qml.math.allclose(mat_product, I)

        qscript_back = qml.transforms.from_zx(zx_g)
        assert isinstance(qscript_back, qml.tape.QuantumScript)

        matrix_qscript_back = qml.matrix(qscript_back, wire_order=list(range(len(qs.wires))))

        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_qscript, qml.math.conj(matrix_qscript_back.T))
        # Remove global phase
        if not np.allclose(mat_product[0, 0], 1.0):
            mat_product /= mat_product[0, 0]

        assert qml.math.allclose(mat_product, I)

    @pytest.mark.parametrize("decompose", decompose_phases)
    def test_circuit(self, decompose):
        """Test a simple circuit."""

        I = qml.math.eye(2**3)

        operations = [
            qml.RZ(5 / 4 * np.pi, wires=0),
            qml.RZ(3 / 4 * np.pi, wires=1),
            qml.PauliY(wires=1),
            qml.RX(0.1, wires=0),
            qml.PauliZ(wires=0),
            qml.RY(0.2, wires=1),
            qml.RZ(0.3, wires=1),
            qml.PauliX(wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.SWAP(wires=[0, 1]),
            qml.Toffoli(wires=[0, 1, 2]),
            qml.CCZ(wires=[0, 1, 2]),
        ]

        qs = QuantumScript(operations, [])
        zx_g = qml.transforms.to_zx(qs)

        assert isinstance(zx_g, pyzx.graph.graph_s.GraphS)

        matrix_qscript = qml.matrix(qs, wire_order=qs.wires)
        matrix_zx = zx_g.to_matrix()
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_qscript, qml.math.conj(matrix_zx.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]
        assert qml.math.allclose(mat_product, I)

        qscript_back = qml.transforms.from_zx(zx_g, decompose_phases=decompose)
        assert isinstance(qscript_back, qml.tape.QuantumScript)

        matrix_qscript_back = qml.matrix(qscript_back, wire_order=list(range(len(qs.wires))))

        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_qscript, qml.math.conj(matrix_qscript_back.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]

        assert qml.math.allclose(mat_product, I)

    def test_circuit_mod_5_4(self):
        """Test the circuit mod 5 4."""
        operations = [
            qml.PauliX(wires=4),
            qml.Hadamard(wires=4),
            qml.CNOT(wires=[3, 4]),
            qml.CNOT(wires=[0, 4]),
            qml.T(wires=4),
            qml.CNOT(wires=[3, 4]),
            qml.adjoint(qml.T)(wires=4),
            qml.CNOT(wires=[0, 4]),
            qml.CNOT(wires=[0, 3]),
            qml.adjoint(qml.T)(wires=3),
            qml.CNOT(wires=[0, 3]),
            qml.CNOT(wires=[3, 4]),
            qml.CNOT(wires=[2, 4]),
            qml.adjoint(qml.T)(wires=4),
            qml.CNOT(wires=[3, 4]),
            qml.T(wires=4),
            qml.CNOT(wires=[2, 4]),
            qml.CNOT(wires=[2, 3]),
            qml.T(wires=3),
            qml.CNOT(wires=[2, 3]),
            qml.Hadamard(wires=4),
            qml.CNOT(wires=[3, 4]),
            qml.Hadamard(wires=4),
            qml.CNOT(wires=[2, 4]),
            qml.adjoint(qml.T)(wires=4),
            qml.CNOT(wires=[1, 4]),
            qml.T(wires=4),
            qml.CNOT(wires=[2, 4]),
            qml.adjoint(qml.T)(wires=4),
            qml.CNOT(wires=[1, 4]),
            qml.T(wires=4),
            qml.CNOT(wires=[1, 2]),
            qml.adjoint(qml.T)(wires=2),
            qml.CNOT(wires=[1, 2]),
            qml.Hadamard(wires=4),
            qml.CNOT(wires=[2, 4]),
            qml.Hadamard(wires=4),
            qml.CNOT(wires=[1, 4]),
            qml.T(wires=4),
            qml.CNOT(wires=[0, 4]),
            qml.adjoint(qml.T)(wires=4),
            qml.CNOT(wires=[1, 4]),
            qml.T(wires=4),
            qml.CNOT(wires=[0, 4]),
            qml.adjoint(qml.T)(wires=4),
            qml.CNOT(wires=[0, 1]),
            qml.T(wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=4),
            qml.CNOT(wires=[1, 4]),
            qml.CNOT(wires=[0, 4]),
        ]

        qs = QuantumScript(operations, [])
        zx_g = qml.transforms.to_zx(qs)

        assert isinstance(zx_g, pyzx.graph.graph_s.GraphS)

        matrix_qscript = qml.matrix(qs, wire_order=qs.wires)
        matrix_zx = zx_g.to_matrix()
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_qscript, qml.math.conj(matrix_zx.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]
        I = qml.math.eye(2**5)
        assert qml.math.allclose(mat_product, I)

        qscript_back = qml.transforms.from_zx(zx_g)
        assert isinstance(qscript_back, qml.tape.QuantumScript)

        matrix_qscript_back = qml.matrix(qscript_back, wire_order=list(range(len(qs.wires))))

        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_qscript, qml.math.conj(matrix_qscript_back.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]
        assert qml.math.allclose(mat_product, I)

    def test_expand_measurements(self):
        """Test with expansion of measurements."""
        I = qml.math.eye(2**2)

        operations = [
            qml.RX(0.1, wires=0),
            qml.PauliZ(wires=0),
            qml.RZ(0.3, wires=1),
            qml.PauliX(wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.SWAP(wires=[0, 1]),
        ]
        measurements = [qml.expval(qml.PauliZ(0) @ qml.PauliX(1))]

        qs = QuantumScript(operations, measurements)
        zx_g = qml.transforms.to_zx(qs, expand_measurements=True)
        assert isinstance(zx_g, pyzx.graph.graph_s.GraphS)

        # Add rotation Hadamard because of PauliX
        operations.append(qml.Hadamard(wires=[1]))
        operations_with_rotations = operations
        qscript_with_rot = QuantumScript(operations_with_rotations, [])
        matrix_qscript = qml.matrix(qscript_with_rot, wire_order=[0, 1])

        matrix_zx = zx_g.to_matrix()
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_qscript, qml.math.conj(matrix_zx.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]
        assert qml.math.allclose(mat_product, I)

        qscript_back = qml.transforms.from_zx(zx_g)
        assert isinstance(qscript_back, qml.tape.QuantumScript)

        matrix_qscript_back = qml.matrix(qscript_back, wire_order=list(range(len(qs.wires))))
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_qscript, qml.math.conj(matrix_qscript_back.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]
        assert qml.math.allclose(mat_product, I)

    def test_embeddings(self):
        """Test with expansion of prep."""
        I = qml.math.eye(2**2)

        prep = [qml.AngleEmbedding(features=[1, 2], wires=range(2), rotation="Z")]

        operations = [
            qml.RX(0.1, wires=0),
            qml.PauliZ(wires=0),
            qml.RZ(0.3, wires=1),
            qml.PauliX(wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.SWAP(wires=[0, 1]),
        ]

        qs = QuantumScript(prep + operations, [])
        zx_g = qml.transforms.to_zx(qs)

        assert isinstance(zx_g, pyzx.graph.graph_s.GraphS)

        matrix_qscript = qml.matrix(qs, wire_order=qs.wires)
        matrix_zx = zx_g.to_matrix()
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_qscript, qml.math.conj(matrix_zx.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]
        assert qml.math.allclose(mat_product, I)

        qscript_back = qml.transforms.from_zx(zx_g)
        assert isinstance(qscript_back, qml.tape.QuantumScript)

        matrix_qscript_back = qml.matrix(qscript_back, wire_order=list(range(len(qs.wires))))
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_qscript, qml.math.conj(matrix_qscript_back.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]
        assert qml.math.allclose(mat_product, I)

    def test_no_decomposition(self):
        """Cross qubit connections is not diagram-like."""
        graph = pyzx.Graph(None)
        q_mapper = pyzx.circuit.gates.TargetMapper()

        inputs = []

        # Create the qubits in the graph and the qubit mapper
        vertex = graph.add_vertex(pyzx.VertexType.BOUNDARY, 0, 0)
        inputs.append(vertex)
        q_mapper.set_prev_vertex(0, vertex)
        q_mapper.set_next_row(0, 1)
        q_mapper.set_qubit(0, 0)

        # Cross qubit connection
        r = q_mapper.next_row(0)

        v1 = graph.add_vertex(pyzx.VertexType.Z, q_mapper.to_qubit(0), r)
        graph.add_edge(graph.edge(q_mapper.prev_vertex(0), v1), pyzx.EdgeType.SIMPLE)
        q_mapper.set_prev_vertex(0, v1)

        q_mapper.set_qubit(1, 1)
        q_mapper.set_next_row(1, r + 1)
        q_mapper.set_next_row(0, r + 1)

        r = max(q_mapper.next_row(1), q_mapper.next_row(0))

        v2 = graph.add_vertex(pyzx.VertexType.Z, q_mapper.to_qubit(1), r)
        graph.add_edge(graph.edge(q_mapper.prev_vertex(0), v2), pyzx.EdgeType.SIMPLE)
        q_mapper.set_prev_vertex(0, v2)

        q_mapper.set_next_row(1, r + 1)
        q_mapper.set_next_row(0, r + 1)
        r = max(q_mapper.next_row(1), q_mapper.next_row(0))

        graph.add_edge((v1, v2), edgetype=pyzx.EdgeType.SIMPLE)

        q_mapper.set_next_row(1, r + 1)
        q_mapper.set_next_row(0, r + 1)
        graph.scalar.add_power(1)

        outputs = []

        graph.set_inputs(tuple(inputs))
        graph.set_outputs(tuple(outputs))

        with pytest.raises(
            QuantumFunctionError,
            match="Cross qubit connections, the graph is not circuit-like.",
        ):
            qml.transforms.from_zx(graph)

    def test_no_suitable_decomposition(self):
        """Test that an error is raised when no suitable decomposition is found."""

        operations = [qml.sum(qml.PauliX(0), qml.PauliZ(0))]

        qs = QuantumScript(operations, [])
        with pytest.raises(
            QuantumFunctionError,
            match="The expansion of the quantum tape failed, PyZX does not support",
        ):
            qml.transforms.to_zx(qs)

    def test_same_type_nodes_simple_edge(self):
        """Test that a Green-Green nodes with simple edge has no corresponding circuit."""
        graph = pyzx.Graph(None)
        q_mapper = pyzx.circuit.gates.TargetMapper()
        c_mapper = pyzx.circuit.gates.TargetMapper()

        inputs = []

        # Create the qubits in the graph and the qubit mapper
        for i in range(2):
            vertex = graph.add_vertex(pyzx.VertexType.BOUNDARY, i, 0)
            inputs.append(vertex)
            q_mapper.set_prev_vertex(i, vertex)
            q_mapper.set_next_row(i, 1)
            q_mapper.set_qubit(i, i)

        # Create Green Green with simple Edge
        r = max(q_mapper.next_row(1), q_mapper.next_row(0))

        v1 = graph.add_vertex(pyzx.VertexType.Z, q_mapper.to_qubit(1), r)
        graph.add_edge(graph.edge(q_mapper.prev_vertex(1), v1), pyzx.EdgeType.SIMPLE)
        q_mapper.set_prev_vertex(1, v1)

        v2 = graph.add_vertex(pyzx.VertexType.Z, q_mapper.to_qubit(0), r)
        graph.add_edge(graph.edge(q_mapper.prev_vertex(0), v2), pyzx.EdgeType.SIMPLE)
        q_mapper.set_prev_vertex(0, v2)

        graph.add_edge((v1, v2), edgetype=pyzx.EdgeType.SIMPLE)

        q_mapper.set_next_row(1, r + 1)
        q_mapper.set_next_row(0, r + 1)
        graph.scalar.add_power(1)

        row = max(q_mapper.max_row(), c_mapper.max_row())

        outputs = []
        for mapper in (q_mapper, c_mapper):
            for label in mapper.labels():
                qubit = mapper.to_qubit(label)
                vertex = graph.add_vertex(pyzx.VertexType.BOUNDARY, qubit, row)
                outputs.append(vertex)
                pre_vertex = mapper.prev_vertex(label)
                graph.add_edge(graph.edge(pre_vertex, vertex))

        graph.set_inputs(tuple(inputs))
        graph.set_outputs(tuple(outputs))

        with pytest.raises(
            QuantumFunctionError,
            match="Two green or respectively two red nodes connected by a ",
        ):
            qml.transforms.from_zx(graph)

    def test_different_type_node_hadamard_edge(self):
        """Test that a Green-Red nodes with Hadamard edge has no corresponding circuit."""
        graph = pyzx.Graph(None)
        q_mapper = pyzx.circuit.gates.TargetMapper()
        c_mapper = pyzx.circuit.gates.TargetMapper()

        inputs = []

        # Create the qubits in the graph and the qubit mapper
        for i in range(2):
            vertex = graph.add_vertex(pyzx.VertexType.BOUNDARY, i, 0)
            inputs.append(vertex)
            q_mapper.set_prev_vertex(i, vertex)
            q_mapper.set_next_row(i, 1)
            q_mapper.set_qubit(i, i)

        # Create Green Red with Hadamard Edge
        r = max(q_mapper.next_row(1), q_mapper.next_row(0))

        v1 = graph.add_vertex(pyzx.VertexType.Z, q_mapper.to_qubit(1), r)
        graph.add_edge(graph.edge(q_mapper.prev_vertex(1), v1), pyzx.EdgeType.SIMPLE)
        q_mapper.set_prev_vertex(1, v1)

        v2 = graph.add_vertex(pyzx.VertexType.X, q_mapper.to_qubit(0), r)
        graph.add_edge(graph.edge(q_mapper.prev_vertex(0), v2), pyzx.EdgeType.SIMPLE)
        q_mapper.set_prev_vertex(0, v2)

        graph.add_edge((v1, v2), edgetype=pyzx.EdgeType.HADAMARD)

        q_mapper.set_next_row(1, r + 1)
        q_mapper.set_next_row(0, r + 1)
        graph.scalar.add_power(1)

        row = max(q_mapper.max_row(), c_mapper.max_row())

        outputs = []
        for mapper in (q_mapper, c_mapper):
            for label in mapper.labels():
                qubit = mapper.to_qubit(label)
                vertex = graph.add_vertex(pyzx.VertexType.BOUNDARY, qubit, row)
                outputs.append(vertex)
                pre_vertex = mapper.prev_vertex(label)
                graph.add_edge(graph.edge(pre_vertex, vertex))

        graph.set_inputs(tuple(inputs))
        graph.set_outputs(tuple(outputs))

        with pytest.raises(
            QuantumFunctionError,
            match="A green and red node connected by a Hadamard edge ",
        ):
            qml.transforms.from_zx(graph)

    def test_cx_gate(self):
        """Test that CX node is converted to the right tape"""
        graph = pyzx.Graph(None)
        q_mapper = pyzx.circuit.gates.TargetMapper()
        c_mapper = pyzx.circuit.gates.TargetMapper()

        inputs = []

        # Create the qubits in the graph and the qubit mapper
        for i in range(2):
            vertex = graph.add_vertex(pyzx.VertexType.BOUNDARY, i, 0)
            inputs.append(vertex)
            q_mapper.set_prev_vertex(i, vertex)
            q_mapper.set_next_row(i, 1)
            q_mapper.set_qubit(i, i)

        # Create Green Red with Hadamard Edge
        r = max(q_mapper.next_row(1), q_mapper.next_row(0))

        v1 = graph.add_vertex(pyzx.VertexType.X, q_mapper.to_qubit(1), r)
        graph.add_edge(graph.edge(q_mapper.prev_vertex(1), v1), pyzx.EdgeType.SIMPLE)
        q_mapper.set_prev_vertex(1, v1)

        v2 = graph.add_vertex(pyzx.VertexType.X, q_mapper.to_qubit(0), r)
        graph.add_edge(graph.edge(q_mapper.prev_vertex(0), v2), pyzx.EdgeType.SIMPLE)
        q_mapper.set_prev_vertex(0, v2)

        graph.add_edge((v1, v2), edgetype=pyzx.EdgeType.HADAMARD)

        q_mapper.set_next_row(1, r + 1)
        q_mapper.set_next_row(0, r + 1)
        graph.scalar.add_power(1)

        row = max(q_mapper.max_row(), c_mapper.max_row())

        outputs = []
        for mapper in (q_mapper, c_mapper):
            for label in mapper.labels():
                qubit = mapper.to_qubit(label)
                vertex = graph.add_vertex(pyzx.VertexType.BOUNDARY, qubit, row)
                outputs.append(vertex)
                pre_vertex = mapper.prev_vertex(label)
                graph.add_edge(graph.edge(pre_vertex, vertex))

        graph.set_inputs(tuple(inputs))
        graph.set_outputs(tuple(outputs))

        tape = qml.transforms.from_zx(graph)
        expected_op = [qml.Hadamard(wires=[1]), qml.CNOT(wires=[1, 0]), qml.Hadamard(wires=[1])]
        for op, op_ex in zip(tape.operations, expected_op):
            qml.assert_equal(op, op_ex)

    def test_qnode_decorator(self):
        """Test the QNode decorator."""
        dev = qml.device("default.qubit", wires=2)

        @partial(qml.transforms.to_zx, expand_measurements=True)
        @qml.qnode(device=dev)
        def circuit(p):
            qml.RZ(p[0], wires=1)
            qml.RZ(p[1], wires=1)
            qml.RX(p[2], wires=0)
            qml.PauliZ(wires=0)
            qml.RZ(p[3], wires=1)
            qml.PauliX(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 0])
            qml.SWAP(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        params = [5 / 4 * np.pi, 3 / 4 * np.pi, 0.1, 0.3]
        g = circuit(params)

        assert isinstance(g, pyzx.graph.graph_s.GraphS)

    def test_qnode_decorator_no_params(self):
        """Test the QNode decorator."""
        dev = qml.device("default.qubit", wires=2)

        @partial(qml.transforms.to_zx, expand_measurements=True)
        @qml.qnode(device=dev)
        def circuit():
            qml.PauliZ(wires=0)
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        g = circuit()

        assert isinstance(g, pyzx.graph.graph_s.GraphS)
