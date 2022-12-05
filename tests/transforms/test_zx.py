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
Unit tests for the `pennylane.transforms.zx` folder.
"""
import numpy as np
import pytest
import pennylane as qml
from pennylane.tape import QuantumTape

# pyzx = pytest.importorskip("pyzx")
import pyzx

import pennylane as qml

pytestmark = pytest.mark.zx

I = qml.math.eye(2)

supported_operations = [
    qml.PauliX(wires=0),
    qml.PauliZ(wires=0),
    qml.Hadamard(wires=0),
    qml.S(wires=0),
    qml.T(wires=0),
    qml.SWAP(wires=[0, 1]),
    qml.CNOT(wires=[0, 1]),
    qml.CZ(wires=[0, 1]),
    qml.CH(wires=[0, 1]),
]

supported_operations_params = [
    qml.RX(0.3, wires=0),
    qml.RZ(0.3, wires=0),
    qml.CRZ(0.3, wires=[0, 1]),
]

expanded_operations = [qml.PauliY(wires=0), qml.PhaseShift(0.3, wires=0), qml.RY(0.3, wires=0)]

non_diagram_like_operations = [qml.CCZ(wires=[0, 1, 2]), qml.Toffoli(wires=[0, 1, 2])]

circuits = []


class TestConvertersZX:
    """Test converters tape_to_graph_zx and graph_zx_to_tape."""

    @pytest.mark.parametrize("operation", supported_operations)
    def test_supported_operation_no_params(self, operation):
        """Test the tape to graph zx tape."""

        I = qml.math.eye(2 ** len(operation.wires))

        tape = QuantumTape([operation], [], [])
        matrix_tape = qml.matrix(tape)

        zx_g = qml.transforms.to_zx(tape)
        matrix_zx = zx_g.to_matrix()

        assert isinstance(zx_g, pyzx.graph.graph_s.GraphS)
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_tape, qml.math.conj(matrix_zx.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]

        assert qml.math.allclose(mat_product, I)

        tape_back = qml.transforms.from_zx(zx_g)
        assert isinstance(tape_back, qml.tape.QuantumTape)

        matrix_tape_back = qml.matrix(tape_back, wire_order=[i for i in range(0, len(tape.wires))])

        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_tape, qml.math.conj(matrix_tape_back.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]

        assert qml.math.allclose(mat_product, I)

    @pytest.mark.parametrize("operation", supported_operations_params)
    def test_supported_operation_params(self, operation):
        """Test the tape to graph zx tape."""

        I = qml.math.eye(2 ** len(operation.wires))

        tape = QuantumTape([operation], [], [])
        matrix_tape = qml.matrix(tape)

        zx_g = qml.transforms.to_zx(tape)

        matrix_zx = zx_g.to_matrix()

        assert isinstance(zx_g, pyzx.graph.graph_s.GraphS)
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_tape, qml.math.conj(matrix_zx.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]

        assert qml.math.allclose(mat_product, I)

        tape_back = qml.transforms.from_zx(zx_g)
        assert isinstance(tape_back, qml.tape.QuantumTape)

        matrix_tape_back = qml.matrix(tape_back, wire_order=[i for i in range(0, len(tape.wires))])

        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_tape, qml.math.conj(matrix_tape_back.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]

        assert qml.math.allclose(mat_product, I)

    @pytest.mark.parametrize("operation", expanded_operations)
    def test_expanded_operation_params(self, operation):
        """Test the tape to graph zx tape."""

        I = qml.math.eye(2 ** len(operation.wires))

        tape = QuantumTape([operation], [], [])
        matrix_tape = qml.matrix(tape)

        zx_g = qml.transforms.to_zx(tape)

        matrix_zx = zx_g.to_matrix()

        assert isinstance(zx_g, pyzx.graph.graph_s.GraphS)
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_tape, qml.math.conj(matrix_zx.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]

        assert qml.math.allclose(mat_product, I)

        tape_back = qml.transforms.from_zx(zx_g)
        assert isinstance(tape_back, qml.tape.QuantumTape)

        matrix_tape_back = qml.matrix(tape_back, wire_order=[i for i in range(0, len(tape.wires))])

        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_tape, qml.math.conj(matrix_tape_back.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]

        assert qml.math.allclose(mat_product, I)

    @pytest.mark.parametrize("operation", non_diagram_like_operations)
    def test_non_diagram_like_op(self, operation):
        """Test the tape to graph zx tape."""

        I = qml.math.eye(2 ** len(operation.wires))

        tape = QuantumTape([operation], [], [])
        matrix_tape = qml.matrix(tape)

        zx_g = qml.transforms.to_zx(tape)
        assert isinstance(zx_g, pyzx.graph.graph_s.GraphS)

        matrix_zx = zx_g.to_matrix()
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_tape, qml.math.conj(matrix_zx.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]
        assert qml.math.allclose(mat_product, I)

        with pytest.raises(qml.QuantumFunctionError, match="Graph doesn't seem circuit like"):
            qml.transforms.from_zx(zx_g)

    @pytest.mark.parametrize("operation", non_diagram_like_operations)
    def test_circuit(self, operation):
        """Test the tape to graph zx tape."""

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
        # measurements = [qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))]

        tape = QuantumTape(operations, [], [])
        zx_g = qml.transforms.to_zx(tape)

        assert isinstance(zx_g, pyzx.graph.graph_s.GraphS)

        matrix_tape = qml.matrix(tape)
        matrix_zx = zx_g.to_matrix()
        # Check whether the two matrices are each others conjugate transposes
        mat_product = qml.math.dot(matrix_tape, qml.math.conj(matrix_zx.T))
        # Remove global phase
        mat_product /= mat_product[0, 0]
        assert qml.math.allclose(mat_product, I)
