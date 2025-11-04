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
from pennylane.wires import Wires


class TestToQasmUnitTests:
    """Unit tests for the to_openqasm() method"""

    def test_empty_circuit(self):
        """Test that an empty circuit graph is properly
        serialized into an empty QASM program."""
        circuit = qml.tape.QuantumScript()
        res = qml.to_openqasm(circuit)
        expected = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        assert res == expected

    def test_native_qasm_gates(self):
        """Test that a circuit containing solely native QASM
        gates is properly serialized."""
        with qml.queuing.AnnotatedQueue() as q_circuit:
            qml.RX(0.43, wires=0)
            qml.RY(0.35, wires=1)
            qml.RZ(0.35, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, 0])
            qml.PauliX(wires=1)

        circuit = qml.tape.QuantumScript.from_queue(q_circuit)
        res = qml.to_openqasm(circuit)

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

    def test_unused_wires(self):
        """Test that unused wires are correctly taken into account"""
        with qml.queuing.AnnotatedQueue() as q_circuit:
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[1, 0])

        circuit = qml.tape.QuantumScript.from_queue(q_circuit)
        res = qml.to_openqasm(circuit, wires=Wires([0, 1, 2, 3, 4]))

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

    def test_to_ApproxTimeEvolution(self):
        """Test for case that requires a decomposition depth of 3 to successfully convert to a MutliRZ"""
        H = qml.Hamiltonian([1], [qml.PauliZ(0) @ qml.PauliZ(1)])

        circuit = qml.tape.QuantumScript([qml.ApproxTimeEvolution(H, 1, n=1)])
        res = qml.to_openqasm(circuit, wires=Wires([0, 1]))

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            cx q[1],q[0];
            rz(2.0) q[0];
            cx q[1],q[0];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            """
        )

        assert res == expected

    def test_rotation_gate_decomposition(self):
        """Test that gates not natively supported by QASM, such as the
        rotation gate, are correctly decomposed and serialized."""
        with qml.queuing.AnnotatedQueue() as q1:
            qml.Rot(0.3, 0.1, 0.2, wires=1)

        circuit1 = qml.tape.QuantumScript.from_queue(q1)
        qasm1 = qml.to_openqasm(circuit1, wires=Wires([0, 1]))

        with qml.queuing.AnnotatedQueue() as q2:
            qml.Rot.compute_decomposition(0.3, 0.1, 0.2, wires=1)

        circuit2 = qml.tape.QuantumScript.from_queue(q2)
        qasm2 = qml.to_openqasm(circuit2, wires=Wires([0, 1]))

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

        with qml.queuing.AnnotatedQueue() as q1:
            qml.StatePrep(psi, wires=[0, 1])

        circuit1 = qml.tape.QuantumScript.from_queue(q1)
        qasm1 = qml.to_openqasm(circuit1, precision=11)

        with qml.queuing.AnnotatedQueue() as q2:
            qml.StatePrep.compute_decomposition(psi, wires=[0, 1])

        circuit2 = qml.tape.QuantumScript.from_queue(q2)
        qasm2 = qml.to_openqasm(circuit2, wires=Wires([0, 1]), precision=11)

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            ry(1.5707963268) q[0];
            ry(1.5707963268) q[1];
            cx q[0],q[1];
            cx q[0],q[1];
            cx q[0],q[1];
            rz(3.1415926536) q[1];
            cx q[0],q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            """
        )

        # different
        assert qasm1 == expected
        assert qasm1 == qasm2

    def test_basis_state_initialization_decomposition(self):
        """Test that the basis state preparation decomposition
        is correctly applied."""
        basis_state = np.array([1, 0, 1, 1])

        with qml.queuing.AnnotatedQueue() as q1:
            qml.BasisState(basis_state, wires=[0, 1, 2, 3])

        circuit1 = qml.tape.QuantumScript.from_queue(q1)
        qasm1 = qml.to_openqasm(circuit1)

        with qml.queuing.AnnotatedQueue() as q2:
            qml.BasisState.compute_decomposition(basis_state, wires=[0, 1, 2, 3])

        circuit2 = qml.tape.QuantumScript.from_queue(q2)
        qasm2 = qml.to_openqasm(circuit2, wires=[0, 1, 2, 3])

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

    def test_rotations(self):
        """Test that observable rotations are correctly applied."""

        with qml.queuing.AnnotatedQueue() as q_circuit:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliX(0))
            qml.expval(qml.PauliZ(1))
            qml.expval(qml.Hadamard(2))

        circuit = qml.tape.QuantumScript.from_queue(q_circuit)
        res = qml.to_openqasm(circuit)

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

        with qml.queuing.AnnotatedQueue() as q2:
            _ = [o.queue() for o in ops2]

        circuit2 = qml.tape.QuantumScript.from_queue(q2)
        qasm2 = qml.to_openqasm(circuit2)

        assert res == qasm2

    def test_only_tape_measurements(self):
        """Test that no computational basis measurements are added other
        than those already in the tape when ``measure_all=False``."""
        with qml.queuing.AnnotatedQueue() as q_circuit:
            qml.RX(0.43, wires="a")
            qml.RY(0.35, wires="b")
            qml.RZ(0.35, wires=2)
            qml.CNOT(wires=["a", "b"])
            qml.Hadamard(wires=2)
            qml.CNOT(wires=[2, "a"])
            qml.PauliX(wires="b")
            qml.expval(qml.PauliZ("a"))
            qml.expval(qml.PauliZ(2))

        circuit = qml.tape.QuantumScript.from_queue(q_circuit)
        res = qml.to_openqasm(circuit, measure_all=False)

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
            measure q[2] -> c[2];
            """
        )

        assert res == expected


class TestQNodeQasmIntegrationTests:
    """Test that the QASM serialization works correctly
    when circuits are created via QNodes."""

    def test_empty_circuit(self):
        """Test that an empty tape is properly
        serialized into an empty QASM program."""
        tape = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])
        res = qml.to_openqasm(tape)

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
        """Test that a tape containing solely native QASM
        gates is properly serialized."""
        tape = qml.tape.QuantumScript(
            [
                qml.RX(0.43, wires=0),
                qml.RY(0.35, wires=1),
                qml.RZ(0.35, wires=2),
                qml.CNOT(wires=[0, 1]),
                qml.Hadamard(wires=2),
                qml.CNOT(wires=[2, 0]),
                qml.PauliX(wires=1),
            ],
            [qml.expval(qml.PauliZ(0))],
        )
        res = qml.to_openqasm(tape)

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
        x = np.array(0.5)
        y = np.array([0.2, 0.1])
        res = qml.to_openqasm(qnode)(x, y)

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

        assert res == expected

        # execute the QNode with new parameters, and serialize again
        x2 = np.array(0.1)
        y2 = np.array([0.3, 0.2])
        res = qml.to_openqasm(qnode)(x2, y2)

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

        assert res == expected

    def test_unsupported_gate(self):
        """Test an exception is raised if an unsupported operation is applied."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def qnode():
            qml.DoubleExcitationPlus(0.5, wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="not supported with to_openqasm"):
            qml.to_openqasm(qnode)()

    def test_unused_wires(self):
        """Test that unused wires are correctly taken into account"""
        dev = qml.device("default.qubit", wires=5)

        tape = qml.tape.QuantumScript(
            [qml.Hadamard(wires=4), qml.CNOT(wires=[1, 0])], [qml.expval(qml.PauliZ(0))]
        )
        res = qml.to_openqasm(tape, wires=dev.wires)

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

        tape = qml.tape.QuantumScript(
            [qml.Rot(0.3, 0.1, 0.2, wires=1)], [qml.expval(qml.PauliZ(0))]
        )
        res = qml.to_openqasm(tape, wires=dev.wires)

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
            qml.StatePrep(state, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # construct the qnode circuit
        state = np.array([1, -1, -1, 1]) / np.sqrt(4)
        res = qml.to_openqasm(qnode, precision=11)(state=state)

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            ry(1.5707963268) q[0];
            ry(1.5707963268) q[1];
            cx q[0],q[1];
            cx q[0],q[1];
            cx q[0],q[1];
            rz(3.1415926536) q[1];
            cx q[0],q[1];
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
        state = np.array([1, 0, 1, 1])
        res = qml.to_openqasm(qnode)(state=state)

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

    def test_rotations(self):
        """Test that observable rotations are correctly applied."""
        tape = qml.tape.QuantumScript(
            [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])],
            [
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.Hadamard(2)),
            ],
        )
        res = qml.to_openqasm(tape)

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

    def test_wires(self):
        """Test that the QASM serializer correctly integrates with the new wires class."""
        dev = qml.device("default.qubit", wires=["a", "b", "c"])

        @qml.qnode(dev)
        def qnode():
            qml.Hadamard(wires="a")
            qml.CNOT(wires=["b", "a"])
            return [
                qml.expval(qml.PauliX("c")),
                qml.expval(qml.PauliZ("a")),
                qml.expval(qml.Hadamard("b")),
            ]

        res = qml.to_openqasm(qnode)()

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cx q[1],q[0];
            h q[2];
            ry(-0.7853981633974483) q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
            """
        )

        assert res == expected

    def test_precision(self):
        """Test that the QASM serializer takes into account the desired precision."""
        tape = qml.tape.QuantumScript([qml.RX(np.pi, 0)], [qml.expval(qml.PauliZ(0))])
        res = qml.to_openqasm(tape, precision=4)

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            rx(3.142) q[0];
            measure q[0] -> c[0];
            """
        )

        assert res == expected

    @pytest.mark.tf
    def test_tf_interface_information_removed(self):
        """Test that interface information from tensorflow is not included in the
        parameter string for parametrized operators"""
        import tensorflow as tf

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def qnode(param):
            qml.RX(param, wires="a")
            return qml.expval(qml.PauliZ("a"))

        res = qml.to_openqasm(qnode)(tf.Variable(1.2))

        expected = dedent(
            """\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            rx(1.2) q[0];
            measure q[0] -> c[0];
            """
        )

        assert res == expected

    def test_error_reset_True(self):
        """Test an error is raised if the mcm has reset"""
        m0 = qml.measure(0, reset=True)
        tape = qml.tape.QuantumScript(m0.measurements)
        with pytest.raises(NotImplementedError):
            qml.to_openqasm(tape)

    def test_error_postselection(self):
        """Test that an error is raised if postselection exists."""
        m0 = qml.measure(0, postselect=1)
        tape = qml.tape.QuantumScript(m0.measurements)
        with pytest.raises(NotImplementedError):
            qml.to_openqasm(tape)

    def test_error_if_mcm_processed(self):
        """Test a NotImplementedError is raised if the mcm is processed in the conditional."""
        m0 = qml.measure(0)
        tape = qml.tape.QuantumScript([m0.measurements[0], qml.ops.Conditional(2 * m0, qml.X(0))])
        with pytest.raises(NotImplementedError):
            qml.to_openqasm(tape)

    def test_multiple_mcms(self):
        """Test that multiple mcms can be translated."""

        m0 = qml.measure(0)
        m1 = qml.measure(1)
        m2 = qml.measure(2)

        tape = qml.tape.QuantumScript(
            [qml.X(0), *m0.measurements, *m1.measurements, *m2.measurements]
        )

        expected = dedent(
            """\
                            OPENQASM 2.0;
                            include "qelib1.inc";
                            qreg q[3];
                            creg c[3];
                            creg mcms[3];
                            x q[0];
                            measure q[0] -> mcms[0];
                            measure q[1] -> mcms[1];
                            measure q[2] -> mcms[2];
                            measure q[0] -> c[0];
                            measure q[1] -> c[1];
                            measure q[2] -> c[2];
                            """
        )
        assert expected == qml.to_openqasm(tape)

    @pytest.mark.parametrize("precision", (None, 2))
    def test_conditional(self, precision):
        """Test that a conditional can be translated."""

        m0 = qml.measure(0)
        tape = qml.tape.QuantumScript(
            [m0.measurements[0], qml.ops.Conditional(m0, qml.RX(0.123456, 0))]
        )
        res = qml.to_openqasm(tape, precision=precision)

        p = f"{0.123456:.{precision}}" if precision else str(0.123456)
        expected = dedent(
            f"""\
                    OPENQASM 2.0;
                    include "qelib1.inc";
                    qreg q[1];
                    creg c[1];
                    creg mcms[1];
                    measure q[0] -> mcms[0];
                    if(mcms[0]==1) rx({p}) q[0];
                    measure q[0] -> c[0];
                    """
        )
        assert res == expected


# pylint: disable=unused-argument
@pytest.mark.slow
class TestQASMConformanceTests:
    """Conformance tests to ensure that the CircuitGraph
    serialized QASM conforms to the QASM standard as implemented
    by Qiskit. Note that this test class requires Qiskit and
    PennyLane-Qiskit as a dependency."""

    # pylint: disable=attribute-defined-outside-init
    @pytest.fixture(name="check_dependencies")
    def check_dependencies_fixture(self):
        self.qiskit = pytest.importorskip("qiskit", minversion="0.14.1")
        pytest.importorskip("pennylane_qiskit")

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

        res = qml.to_openqasm(qnode)([0.1, 0.2])

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

        state = np.array([1, 0, 1, 1])
        res = qml.to_openqasm(qnode)(state=state)
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
        qasm = qml.to_openqasm(qnode)(params)
        qc = self.qiskit.QuantumCircuit.from_qasm_str(qasm)

        gates = [g for g, _, _ in qc.data]

        qreg = qc.qregs[0]
        for idx, g in enumerate(gates):
            # attach a wires attribute to each gate, containing
            # a list of wire integers it acts on, so we can assert
            # correctness below.
            g.wires = [qreg.index(q) for q in qc.data[idx][1]]

        # operations
        assert gates[0].name == "h"
        assert gates[0].wires == [0]

        assert gates[1].name == "ry"
        assert gates[1].wires == [0]
        assert gates[1].params == [params[1]]

        assert gates[2].name == "cx"
        assert gates[2].wires == [0, 1]

        assert gates[4].name == "rx"
        assert gates[4].wires == [1]
        assert gates[4].params == [params[0]]

        # rotations
        assert gates[3].name == "h"
        assert gates[3].wires == [0]

        assert gates[5].name == "ry"
        assert gates[5].wires == [2]
        assert gates[5].params == [-np.pi / 4]
