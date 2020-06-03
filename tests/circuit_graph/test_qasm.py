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
Unit tests for the :mod:`pennylane.circuit_graph.to_openqasm()` method.
"""
# pylint: disable=no-self-use,too-many-arguments,protected-access
from textwrap import dedent

import numpy as np
import pytest

import pennylane as qml
from pennylane import CircuitGraph
from pennylane.wires import Wires


class TestToQasmUnitTests:
    """Unit tests for the to_openqasm() method"""

    def test_empty_circuit(self):
        """Test that an empty circuit graph is properly
        serialized into an empty QASM program."""
        circuit = CircuitGraph([], {})
        res = circuit.to_openqasm()
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        assert res == expected

    def test_native_qasm_gates(self):
        """Test that a circuit containing solely native QASM
        gates is properly serialized."""
        ops = [
            qml.RX(0.43, wires=0),
            qml.RY(0.35, wires=1),
            qml.RZ(0.35, wires=2),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=2),
            qml.CNOT(wires=[2, 0]),
            qml.PauliX(wires=1),
        ]

        circuit = CircuitGraph(ops, {})
        res = circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            rx(0.43) q[0];
            ry(0.35) q[1];
            rz(0.35) q[2];
            cx q[0],q[1];
            h q[2];
            cx q[2],q[0];
            x q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            """
        )

        assert res == expected

    def test_native_inverse_gates(self):
        """Test that a circuit containing inverse gates that are supported
        natively by QASM, such as sdg, are correctly serialized."""
        ops = [
            qml.S(wires=0),
            qml.S(wires=0).inv(),
            qml.T(wires=0),
            qml.T(wires=0).inv(),
        ]

        circuit = CircuitGraph(ops, {})
        res = circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            s q[0];
            sdg q[0];
            t q[0];
            tdg q[0];
            measure q[0] -> c[0];
            """
        )

        assert res == expected

    def test_unused_wires(self):
        """Test that unused wires are correctly taken into account"""
        ops = [
            qml.Hadamard(wires=4),
            qml.CNOT(wires=[1, 0]),
        ]

        circuit = CircuitGraph(ops, {})
        res = circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[5];
            creg c[5];
            h q[4];
            cx q[1],q[0];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            measure q[3] -> c[3];
            measure q[4] -> c[4];
            """
        )

        assert res == expected

    def test_rotation_gate_decomposition(self):
        """Test that gates not natively supported by QASM, such as the
        rotation gate, are correctly decomposed and serialized."""
        ops1 = [qml.Rot(0.3, 0.1, 0.2, wires=1)]
        circuit1 = CircuitGraph(ops1, {})
        qasm1 = circuit1.to_openqasm()

        ops2 = qml.Rot.decomposition(0.3, 0.1, 0.2, wires=1)
        circuit2 = CircuitGraph(ops2, {})
        qasm2 = circuit2.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            rz(0.3) q[1];
            ry(0.1) q[1];
            rz(0.2) q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            """
        )

        assert qasm1 == expected
        assert qasm1 == qasm2

    def test_state_initialization_decomposition(self):
        """Test that the Mottonen state prepration decomposition
        is correctly applied."""
        psi = np.array([1, -1, -1, 1]) / np.sqrt(4)

        ops1 = [qml.QubitStateVector(psi, wires=[0, 1])]
        circuit1 = CircuitGraph(ops1, {})
        qasm1 = circuit1.to_openqasm()

        ops2 = qml.QubitStateVector.decomposition(psi, wires=[0, 1])
        circuit2 = CircuitGraph(ops2, {})
        qasm2 = circuit2.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            ry(1.5707963267948968) q[1];
            ry(1.5707963267948963) q[0];
            cx q[1],q[0];
            ry(0.0) q[0];
            cx q[1],q[0];
            rz(0.0) q[0];
            cx q[1],q[0];
            rz(3.141592653589793) q[0];
            cx q[1],q[0];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            """
        )

        assert qasm1 == expected
        assert qasm1 == qasm2

    def test_basis_state_initialization_decomposition(self):
        """Test that the basis state preparation decomposition

        is correctly applied."""
        basis_state = np.array([1, 0, 1, 1])

        ops1 = [qml.BasisState(basis_state, wires=[0, 1, 2, 3])]
        circuit1 = CircuitGraph(ops1, {})
        qasm1 = circuit1.to_openqasm()

        ops2 = qml.BasisState.decomposition(basis_state, wires=[0, 1, 2, 3])
        circuit2 = CircuitGraph(ops2, {})
        qasm2 = circuit2.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[4];
            creg c[4];
            x q[0];
            x q[2];
            x q[3];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            measure q[3] -> c[3];
            """
        )

        assert qasm1 == expected
        assert qasm1 == qasm2

    def test_unsupported_gate(self):
        """Test an exception is raised if an unsupported operation is
        applied."""
        U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        ops = [qml.S(wires=0), qml.QubitUnitary(U, wires=[0, 1])]

        circuit = CircuitGraph(ops, {})

        with pytest.raises(
            qml.DeviceError, match="Gate QubitUnitary not supported on device QASM serializer"
        ):
            res = circuit.to_openqasm()

    def test_rotations(self):
        """Test that observable rotations are correctly applied."""
        ops = [
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.Hadamard(2)),
        ]

        circuit = CircuitGraph(ops, {})
        res = circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cx q[0],q[1];
            h q[0];
            ry(-0.7853981633974483) q[2];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            """
        )

        assert res == expected

        ops2 = circuit.operations + circuit.diagonalizing_gates
        circuit2 = CircuitGraph(ops2, {})
        qasm2 = circuit2.to_openqasm()

        assert res == qasm2


class TestQNodeQasmIntegrationTests:
    """Test that the QASM serialization works correctly
    when circuits are created via QNodes."""

    def test_empty_circuit(self):
        """Test that an empty QNode is properly
        serialized into an empty QASM program."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def qnode():
            return qml.expval(qml.PauliZ(0))

        # construct the qnode circuit
        qnode()

        res = qnode.circuit.to_openqasm()
        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            measure q[0] -> c[0];
            """
        )
        assert res == expected

    def test_native_qasm_gates(self):
        """Test that a QNode containing solely native QASM
        gates is properly serialized."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode():
            qml.RX(0.43, wires=0)
            qml.RY(0.35, wires=1)
            qml.RZ(0.35, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 0])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliZ(0))

        # construct the qnode circuit
        qnode()
        res = qnode.circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            rx(0.43) q[0];
            ry(0.35) q[1];
            rz(0.35) q[2];
            cx q[0],q[1];
            h q[2];
            cx q[2],q[0];
            x q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            """
        )

        assert res == expected

    def test_parametrized_native_qasm_gates(self):
        """Test that a QNode containing solely native QASM
        gates, as well as input parameters, is properly serialized.
        In addition, double check the serialization changes as parameters
        are changed."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode(x, y):
            qml.RX(x, wires=0)
            qml.RY(y[0], wires=1)
            qml.RZ(y[1], wires=2)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 0])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliZ(0))

        # execute the QNode with parameters, and serialize
        params = np.array([0.5, [0.2, 0.1]])
        qnode(*params)

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            rx(0.5) q[0];
            ry(0.2) q[1];
            rz(0.1) q[2];
            cx q[0],q[1];
            h q[2];
            cx q[2],q[0];
            x q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            """
        )

        res = qnode.circuit.to_openqasm()
        assert res == expected

        # execute the QNode with new parameters, and serialize again
        params = np.array([0.1, [0.3, 0.2]])
        qnode(*params)

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            rx(0.1) q[0];
            ry(0.3) q[1];
            rz(0.2) q[2];
            cx q[0],q[1];
            h q[2];
            cx q[2],q[0];
            x q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            """
        )

        res = qnode.circuit.to_openqasm()
        assert res == expected

    def test_native_inverse_gates(self):
        """Test that a QNode containing inverse gates that are supported
        natively by QASM, such as sdg, are correctly serialized."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def qnode():
            qml.S(wires=0)
            qml.S(wires=0).inv()
            qml.T(wires=0)
            qml.T(wires=0).inv()
            return qml.expval(qml.PauliZ(0))

        # construct the qnode circuit
        qnode()
        res = qnode.circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            s q[0];
            sdg q[0];
            t q[0];
            tdg q[0];
            measure q[0] -> c[0];
            """
        )

        assert res == expected

    def test_unused_wires(self):
        """Test that unused wires are correctly taken into account"""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def qnode():
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[1, 0])
            return qml.expval(qml.PauliZ(0))

        # construct the qnode circuit
        qnode()
        res = qnode.circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[5];
            creg c[5];
            h q[4];
            cx q[1],q[0];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            measure q[3] -> c[3];
            measure q[4] -> c[4];
            """
        )

        assert res == expected

    def test_rotation_gate_decomposition(self):
        """Test that gates not natively supported by QASM, such as the
        rotation gate, are correctly decomposed and serialized."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qnode():
            qml.Rot(0.3, 0.1, 0.2, wires=1)
            return qml.expval(qml.PauliZ(0))

        # construct the qnode circuit
        qnode()
        res = qnode.circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            rz(0.3) q[1];
            ry(0.1) q[1];
            rz(0.2) q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            """
        )

        assert res == expected

    def test_state_initialization_decomposition(self):
        """Test that the Mottonen state prepration decomposition
        is correctly applied."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qnode(state=None):
            qml.QubitStateVector(state, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # construct the qnode circuit
        qnode(state=np.array([1, -1, -1, 1]) / np.sqrt(4))
        res = qnode.circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            ry(1.5707963267948968) q[1];
            ry(1.5707963267948963) q[0];
            cx q[1],q[0];
            ry(0.0) q[0];
            cx q[1],q[0];
            rz(0.0) q[0];
            cx q[1],q[0];
            rz(3.141592653589793) q[0];
            cx q[1],q[0];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            """
        )

        assert res == expected

    def test_basis_state_initialization_decomposition(self):
        """Test that the basis state prepration decomposition
        is correctly applied."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def qnode(state=None):
            qml.BasisState(state, wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        # construct the qnode circuit
        qnode(state=np.array([1, 0, 1, 1]))
        res = qnode.circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[4];
            creg c[4];
            x q[0];
            x q[2];
            x q[3];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            measure q[3] -> c[3];
            """
        )

        assert res == expected

    def test_unsupported_gate(self):
        """Test an exception is raised if an unsupported operation is
        applied."""
        U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def qnode():
            qml.S(wires=0)
            qml.QubitUnitary(U, wires=0)
            return qml.expval(qml.PauliZ(0))

        qnode()

        with pytest.raises(
            qml.DeviceError, match="Gate QubitUnitary not supported on device QASM serializer"
        ):
            qnode.circuit.to_openqasm()

    def test_rotations(self):
        """Test that observable rotations are correctly applied."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return [
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.Hadamard(2)),
            ]

        qnode()
        res = qnode.circuit.to_openqasm()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cx q[0],q[1];
            h q[0];
            ry(-0.7853981633974483) q[2];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            """
        )

        assert res == expected


class TestQASMConformanceTests:
    """Conformance tests to ensure that the CircuitGraph
    serialized QASM conforms to the QASM standard as implemented
    by Qiskit. Note that this test class requires Qiskit and
    PennyLane-Qiskit as a dependency."""


    @pytest.fixture
    def check_dependencies(self):
        self.qiskit = pytest.importorskip("qiskit", minversion="0.14.1")
        pl_qiskit = pytest.importorskip("pennylane_qiskit")

    def test_agrees_qiskit_plugin(self, check_dependencies):
        """Test that the QASM generated by the CircuitGraph agrees
        with the QASM generated by the PennyLane-Qiskit plugin."""
        dev = qml.device("qiskit.basicaer", wires=3)

        @qml.qnode(dev)
        def qnode(x):
            qml.Hadamard(wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x[0], wires=1)
            return [
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.Hadamard(2)),
            ]

        qnode([0.1, 0.2])
        res = qnode.circuit.to_openqasm()

        # Note: Qiskit hardcodes in pi as a QASM constant.
        # Here, we replace it with its numerical value.
        expected = dev._circuit.qasm().replace("pi/4", str(np.pi / 4))

        assert res == expected

    def test_basis_state_agrees_qiskit_plugin(self, check_dependencies):
        """Test that the basis state prepration QASM agrees
        with that generated by the PennyLane-Qiskit plugin. This is
        a useful test to ensure that we are using the correct qubit
        ordering convention."""
        dev = qml.device("qiskit.basicaer", wires=4)

        @qml.qnode(dev)
        def qnode(state=None):
            qml.BasisState(state, wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        # construct the qnode circuit
        qnode(state=np.array([1, 0, 1, 1]))
        res = qnode.circuit.to_openqasm()
        expected = dev._circuit.qasm()

        assert res == expected

    def test_qiskit_load_generated_qasm(self, check_dependencies):
        """Test that the QASM generated by the CircuitGraph
        corresponds to valid QASM, that can be loaded by Qiskit."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode(x):
            qml.Hadamard(wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x[0], wires=1)
            return [
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.Hadamard(2)),
            ]

        params = [0.1, 0.2]
        qnode(params)
        qasm = qnode.circuit.to_openqasm()
        qc = self.qiskit.QuantumCircuit.from_qasm_str(qasm)

        gates = [g for g, _, _ in qc.data]

        for idx, g in enumerate(gates):
            # attach a wires attribute to each gate, containing
            # a list of wire integers it acts on, so we can assert
            # correctness below.
            g.wires = [q.index for q in qc.data[idx][1]]

        # operations
        assert gates[0].name == "h"
        assert gates[0].wires == Wires([0])

        assert gates[1].name == "ry"
        assert gates[1].wires == Wires([0])
        assert gates[1].params == [params[1]]

        assert gates[2].name == "cx"
        assert gates[2].wires == Wires([0, 1])

        assert gates[4].name == "rx"
        assert gates[4].wires == Wires([1])
        assert gates[4].params == [params[0]]

        # rotations
        assert gates[3].name == "h"
        assert gates[3].wires == Wires([0])

        assert gates[5].name == "ry"
        assert gates[5].wires == Wires([2])
        assert gates[5].params == [-np.pi / 4]
