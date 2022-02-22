# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Tests for the transform implementing the deferred measurement principle.
"""
import pytest
import math

import pennylane as qml
import pennylane.numpy as np


class TestQNode:
    """Test that the transform integrates well with QNodes."""

    def test_only_mcm(self):
        """Test that a quantum function that only contains one mid-circuit
        measurement yields the correct results and is transformed correctly."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode1():
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode2():
            m = qml.mid_measure(1)
            return qml.expval(qml.PauliZ(0))

        res1 = qnode1()
        res2 = qnode2()
        assert res1 == res2
        assert isinstance(res1, type(res2))
        assert res1.shape == res2.shape

        for op1, op2 in zip(qnode1.qtape.queue, qnode2.qtape.queue):
            assert type(op1) == type(op2)
            assert op1.data == op2.data

    def test_mid_measure_between_ops(self):
        """Test that a quantum function that contains one operation before and
        after a mid-circuit measurement yields the correct results and is
        transformed correctly."""
        dev = qml.device("default.qubit", wires=3)

        def func1():
            qml.RY(0.123, wires=0)
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        def func2():
            qml.RY(0.123, wires=0)
            qml.mid_measure(1)
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        tape_deferred_func = qml.defer_measurements(func2)
        qnode1 = qml.QNode(func1, dev)
        qnode2 = qml.QNode(tape_deferred_func, dev)

        res1 = qnode1()
        res2 = qnode2()
        assert res1 == res2
        assert isinstance(res1, type(res2))
        assert res1.shape == res2.shape

        for op1, op2 in zip(qnode1.qtape.queue, qnode2.qtape.queue):
            assert type(op1) == type(op2)
            assert op1.data == op2.data

    def test_already_measured_error_operation(self):
        """Test that attempting to apply an operation on a wires that has been
        measured raises an error."""
        dev = qml.device("default.qubit", wires=3)

        def qfunc():
            qml.mid_measure(1)
            qml.PauliX(1)
            return qml.expval(qml.PauliZ(0))

        tape_deferred_func = qml.defer_measurements(qfunc)
        qnode = qml.QNode(tape_deferred_func, dev)

        with pytest.raises(ValueError, match="Cannot apply operations"):
            qnode()

    def test_already_measured_error_terminal_measurement(self):
        """Test that attempting to measure a wire at the end of the circuit
        that has been measured in the middle of the circuit raises an error."""
        dev = qml.device("default.qubit", wires=3)

        def qfunc():
            qml.mid_measure(1)
            return qml.expval(qml.PauliZ(1))

        tape_deferred_func = qml.defer_measurements(qfunc)
        qnode = qml.QNode(tape_deferred_func, dev)

        with pytest.raises(ValueError, match="Cannot apply operations"):
            qnode()


class TestMidCircuitMeasurements:
    """Tests mid circuit measurements"""

    @pytest.mark.parametrize("r", np.linspace(0.0, 1.6, 10))
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
    def test_quantum_teleportation(self, device, r):
        dev = qml.device(device, wires=3)

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.RY(rads, wires=0)

            return qml.probs(wires=0)

        @qml.qnode(dev)
        @qml.defer_measurements
        def teleportation_circuit(rads):

            # Create Alice's secret qubit state
            qml.RY(rads, wires=0)

            # create an EPR pair with wires 1 and 2. 1 is held by Alice and 2 held by Bob
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 2])

            # Alice sends her qubits through a CNOT gate.
            qml.CNOT(wires=[0, 1])

            # Alice then sends the first qubit through a Hadamard gate.
            qml.Hadamard(wires=0)

            # Alice measures her qubits, obtaining one of four results, and sends this information to Bob.
            m_0 = qml.mid_measure(0)
            m_1 = qml.mid_measure(1)

            # Given Alice's measurements, Bob performs one of four operations on his half of the EPR pair and
            # recovers the original quantum state.
            qml.if_then(m_1, qml.RX)(math.pi, wires=2)
            qml.if_then(m_0, qml.RZ)(math.pi, wires=2)

            return qml.probs(wires=2)

        normal_probs = normal_circuit(r)
        teleported_probs = teleportation_circuit(r)

        assert np.allclose(normal_probs, teleported_probs)

    @pytest.mark.parametrize("r", np.linspace(0.1, 2 * np.pi - 0.1, 4))
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("ops", [(qml.RX, qml.CRX), (qml.RY, qml.CRY), (qml.RZ, qml.CRZ)])
    def test_conditional_rotations(self, device, r, ops):
        dev = qml.device(device, wires=3)

        op, controlled_op = ops

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.Hadamard(0)
            controlled_op(rads, wires=[0, 1])
            return qml.probs(wires=1)

        @qml.qnode(dev)
        @qml.defer_measurements
        def teleportation_circuit(rads):
            qml.Hadamard(0)
            m_0 = qml.mid_measure(0)
            qml.if_then(m_0, op)(rads, wires=1)
            return qml.probs(wires=1)

        normal_probs = normal_circuit(r)
        teleported_probs = teleportation_circuit(r)

        assert np.allclose(normal_probs, teleported_probs)

    def test_keyword_syntax(self):
        """Test that passing an argument to the conditioned operation using the
        keyword syntax works."""
        op = qml.RY

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qnode1(parameters):
            qml.Hadamard(0)
            qml.ctrl(op, control=0)(phi=par, wires=1)
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode2(parameters):
            qml.Hadamard(0)
            m_0 = qml.mid_measure(0)
            qml.if_then(m_0, op)(phi=par, wires=1)
            return qml.expval(qml.PauliZ(1))

        par = np.array(0.3)

        assert np.allclose(qnode1(par), qnode2(par))


class TestTemplates:
    """Tests templates being conditioned on mid-circuit measurement outcomes."""

    def test_basis_state_prep(self, template):
        """Test the basis state prep template conditioned on mid-circuit
        measurement outcomes."""
        template = qml.BasisStatePreparation

        basis_state = [0, 1, 1, 0]

        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def qnode1():
            qml.Hadamard(0)
            qml.ctrl(template, control=0)(basis_state, wires=range(1, 5))
            return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4))

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode2():
            qml.Hadamard(0)
            m_0 = qml.mid_measure(0)
            qml.if_then(m_0, template)(basis_state, wires=range(1, 5))
            return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4))

        dev = qml.device("default.qubit", wires=2)

        assert np.allclose(qnode1(), qnode2())

    @pytest.mark.parametrize("template", [qml.StronglyEntanglingLayers, qml.BasicEntanglerLayers])
    def test_layers(self, template):
        """Test layers conditioned on mid-circuit measurement outcomes."""
        dev = qml.device("default.qubit", wires=3)

        num_wires = 2

        @qml.qnode(dev)
        def qnode1(parameters):
            qml.Hadamard(0)
            qml.ctrl(template, control=0)(parameters, wires=range(1, 3))
            return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2))

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode2(parameters):
            qml.Hadamard(0)
            m_0 = qml.mid_measure(0)
            qml.if_then(m_0, template)(parameters, wires=range(1, 3))
            return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2))

        shape = template.shape(n_layers=2, n_wires=num_wires)
        weights = np.random.random(size=shape)

        assert np.allclose(qnode1(weights), qnode2(weights))


class TestDrawing:
    """Tests drawing circuits with mid-circuit measurements and conditional
    operations that have been transformed"""

    def test_drawing(self):
        """Test that drawing a func with mid-circuit measurements works and
        that controlled operations are drawn for conditional operations."""

        def qfunc():
            m_0 = qml.mid_measure(0)
            qml.if_then(m_0, qml.RY)(0.312, wires=1)

            m_2 = qml.mid_measure(2)
            qml.if_then(m_2, qml.RY)(0.312, wires=1)
            return qml.expval(qml.PauliZ(1))

        dev = qml.device("default.qubit", wires=4)

        transformed_qfunc = qml.transforms.defer_measurements(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        expected = (
            "0: ─╭C────────────────────────────────────────────────────┤     \n"
            "1: ─╰ControlledOperation(0.31)─╭ControlledOperation(0.31)─┤  <Z>\n"
            "2: ────────────────────────────╰C─────────────────────────┤     "
        )
        assert qml.draw(transformed_qnode)() == expected
