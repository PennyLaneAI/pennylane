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
from pennylane.measurements import MeasurementValue


class TestQNode:
    """Test that the transform integrates well with QNodes."""

    def test_only_mcm(self):
        """Test that a quantum function that only contains one mid-circuit
        measurement yields the correct results and is transformed correctly."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode1():
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode2():
            m = qml.measure(1)
            return qml.expval(qml.PauliZ(0))

        res1 = qnode1()
        res2 = qnode2()
        assert res1 == res2
        assert isinstance(res1, type(res2))
        assert res1.shape == res2.shape

        assert len(qnode1.qtape.operations) == len(qnode2.qtape.operations)
        assert len(qnode1.qtape.measurements) == len(qnode2.qtape.measurements)

        # Check the operations
        for op1, op2 in zip(qnode1.qtape.operations, qnode2.qtape.operations):
            assert type(op1) == type(op2)
            assert op1.data == op2.data

        # Check the measurements
        for op1, op2 in zip(qnode1.qtape.measurements, qnode2.qtape.measurements):
            assert type(op1) == type(op2)
            assert op1.data == op2.data

    def test_measure_between_ops(self):
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
            qml.measure(1)
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

        assert len(qnode1.qtape.operations) == len(qnode2.qtape.operations)
        assert len(qnode1.qtape.measurements) == len(qnode2.qtape.measurements)

        # Check the operations
        for op1, op2 in zip(qnode1.qtape.operations, qnode2.qtape.operations):
            assert type(op1) == type(op2)
            assert op1.data == op2.data

        # Check the measurements
        for op1, op2 in zip(qnode1.qtape.measurements, qnode2.qtape.measurements):
            assert type(op1) == type(op2)
            assert op1.data == op2.data

    @pytest.mark.parametrize(
        "mid_measure_wire, tp_wires", [(0, [1, 2, 3]), (0, [3, 1, 2]), ("a", ["b", "c", "d"])]
    )
    def test_measure_with_tensor_obs(self, mid_measure_wire, tp_wires):
        """Test that the defer_measurements transform works well even with
        tensor observables in the tape."""
        dev = qml.device("default.qubit", wires=[mid_measure_wire] + tp_wires)

        with qml.tape.QuantumTape() as tape:
            qml.measure(mid_measure_wire)
            qml.expval(qml.operation.Tensor(*[qml.PauliZ(w) for w in tp_wires]))

        tape = qml.defer_measurements(tape)

        # Check the operations and measurements in the tape
        assert tape._ops == []
        assert len(tape.measurements) == 1

        measurement = tape.measurements[0]
        assert isinstance(measurement, qml.measurements.MeasurementProcess)

        tensor = measurement.obs
        assert len(tensor.obs) == 3

        for idx, ob in enumerate(tensor.obs):
            assert isinstance(ob, qml.PauliZ)
            assert ob.wires == qml.wires.Wires(tp_wires[idx])

    def test_already_measured_error_operation(self):
        """Test that attempting to apply an operation on a wires that has been
        measured raises an error."""
        dev = qml.device("default.qubit", wires=3)

        def qfunc():
            qml.measure(1)
            qml.PauliX(1)
            return qml.expval(qml.PauliZ(0))

        tape_deferred_func = qml.defer_measurements(qfunc)
        qnode = qml.QNode(tape_deferred_func, dev)

        with pytest.raises(ValueError, match="wires have been measured already: {1}"):
            qnode()

    def test_already_measured_error_terminal_measurement(self):
        """Test that attempting to measure a wire at the end of the circuit
        that has been measured in the middle of the circuit raises an error."""
        dev = qml.device("default.qubit", wires=3)

        def qfunc():
            qml.measure(1)
            return qml.expval(qml.PauliZ(1))

        tape_deferred_func = qml.defer_measurements(qfunc)
        qnode = qml.QNode(tape_deferred_func, dev)

        with pytest.raises(ValueError, match="Cannot apply operations"):
            qnode()

    def test_cv_op_error(self):
        """Test that CV operations are not supported."""
        dev = qml.device("default.gaussian", wires=3)

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode():
            qml.Rotation(0.123, wires=[0])
            return qml.expval(qml.NumberOperator(1))

        with pytest.raises(
            ValueError, match="Continuous variable operations and observables are not supported"
        ):
            qnode()

    def test_cv_obs_error(self):
        """Test that CV observables are not supported."""
        dev = qml.device("default.gaussian", wires=3)

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode():
            return qml.expval(qml.NumberOperator(1))

        with pytest.raises(
            ValueError, match="Continuous variable operations and observables are not supported"
        ):
            qnode()


class TestConditionalOperations:
    """Tests conditional operations"""

    @pytest.mark.parametrize(
        "terminal_measurement",
        [
            qml.expval(qml.PauliZ(1)),
            qml.var(qml.PauliZ(2) @ qml.PauliZ(0)),
            qml.probs(wires=[1, 0]),
        ],
    )
    def test_correct_ops_in_tape(self, terminal_measurement):
        """Test that the underlying tape contains the correct operations."""
        dev = qml.device("default.qubit", wires=5)

        first_par = 0.1
        sec_par = 0.3

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(4)
            qml.cond(m_0, qml.RY)(first_par, wires=1)

            m_1 = qml.measure(3)
            qml.cond(m_0, qml.RZ)(sec_par, wires=1)
            qml.apply(terminal_measurement)

        tape = qml.defer_measurements(tape)

        assert len(tape.operations) == 2
        assert len(tape.measurements) == 1

        # Check the two underlying Controlled instances
        first_ctrl_op = tape.operations[0]
        assert isinstance(first_ctrl_op, qml.ops.op_math.Controlled)
        assert qml.equal(first_ctrl_op.base, qml.RY(first_par, 1))

        sec_ctrl_op = tape.operations[1]
        assert isinstance(sec_ctrl_op, qml.ops.op_math.Controlled)
        assert qml.equal(sec_ctrl_op.base, qml.RZ(sec_par, 1))

        assert tape.measurements[0] is terminal_measurement

    def test_correct_ops_in_tape_inversion(self):
        """Test that the underlying tape contains the correct operations if a
        measurement value was inverted."""
        dev = qml.device("default.qubit", wires=3)

        first_par = 0.1
        sec_par = 0.3

        terminal_measurement = qml.expval(qml.PauliZ(1))

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(~m_0, qml.RY)(first_par, wires=1)
            qml.apply(terminal_measurement)

        tape = qml.defer_measurements(tape)

        # Conditioned on 0 as the control value, PauliX is applied before and after
        assert len(tape.operations) == 3
        assert len(tape.measurements) == 1

        # We flip the control qubit
        first_x = tape.operations[0]
        assert isinstance(first_x, qml.PauliX)
        assert first_x.wires == qml.wires.Wires(0)

        # Check the two underlying Controlled instance
        ctrl_op = tape.operations[1]
        assert isinstance(ctrl_op, qml.ops.op_math.Controlled)
        assert qml.equal(ctrl_op.base, qml.RY(first_par, 1))

        assert ctrl_op.wires == qml.wires.Wires([0, 1])

        # We flip the control qubit back
        sec_x = tape.operations[2]
        assert isinstance(sec_x, qml.PauliX)
        assert sec_x.wires == qml.wires.Wires(0)

    def test_correct_ops_in_tape_assert_zero_state(self):
        """Test that the underlying tape contains the correct operations if a
        conditional operation was applied in the zero state case.

        Note: this case is the same as inverting right after obtaining a
        measurement value."""
        dev = qml.device("default.qubit", wires=3)

        first_par = 0.1
        sec_par = 0.3

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0 == 0, qml.RY)(first_par, wires=1)
            qml.expval(qml.PauliZ(1))

        tape = qml.defer_measurements(tape)

        # Conditioned on 0 as the control value, PauliX is applied before and after
        assert len(tape.operations) == 3
        assert len(tape.measurements) == 1

        # We flip the control qubit
        first_x = tape.operations[0]
        assert isinstance(first_x, qml.PauliX)
        assert first_x.wires == qml.wires.Wires(0)

        # Check the underlying Controlled instance
        ctrl_op = tape.operations[1]
        assert isinstance(ctrl_op, qml.ops.op_math.Controlled)
        assert qml.equal(ctrl_op.base, qml.RY(first_par, 1))

        # We flip the control qubit back
        sec_x = tape.operations[2]
        assert isinstance(sec_x, qml.PauliX)
        assert sec_x.wires == qml.wires.Wires(0)

    @pytest.mark.parametrize("rads", np.linspace(0.0, np.pi, 3))
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    def test_quantum_teleportation(self, device, rads):
        """Test quantum teleportation."""
        dev = qml.device(device, wires=3)

        terminal_measurement = qml.probs(wires=2)

        with qml.tape.QuantumTape() as tape:

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
            m_0 = qml.measure(0)
            m_1 = qml.measure(1)

            # Given Alice's measurements, Bob performs one of four operations on his half of the EPR pair and
            # recovers the original quantum state.
            qml.cond(m_1, qml.RX)(math.pi, wires=2)
            qml.cond(m_0, qml.RZ)(math.pi, wires=2)

            qml.apply(terminal_measurement)

        tape = qml.defer_measurements(tape)
        assert len(tape.operations) == 5 + 2  # 5 regular ops + 2 conditional ops
        assert len(tape.measurements) == 1

        # Check the each operation
        op1 = tape.operations[0]
        assert isinstance(op1, qml.RY)
        assert op1.wires == qml.wires.Wires(0)
        assert op1.data == [rads]

        op2 = tape.operations[1]
        assert isinstance(op2, qml.Hadamard)
        assert op2.wires == qml.wires.Wires(1)

        op3 = tape.operations[2]
        assert isinstance(op3, qml.CNOT)
        assert op3.wires == qml.wires.Wires([1, 2])

        op4 = tape.operations[3]
        assert isinstance(op4, qml.CNOT)
        assert op4.wires == qml.wires.Wires([0, 1])

        op5 = tape.operations[4]
        assert isinstance(op5, qml.Hadamard)
        assert op5.wires == qml.wires.Wires([0])

        # Check the two underlying  Controlled instances
        ctrl_op1 = tape.operations[5]
        assert isinstance(ctrl_op1, qml.ops.op_math.Controlled)
        assert qml.equal(ctrl_op1.base, qml.RX(math.pi, 2))

        ctrl_op2 = tape.operations[6]
        assert isinstance(ctrl_op2, qml.ops.op_math.Controlled)
        assert qml.equal(ctrl_op2.base, qml.RZ(math.pi, 2))
        assert ctrl_op2.wires == qml.wires.Wires([0, 2])

        # Check the measurement
        assert tape.measurements[0] == terminal_measurement

    @pytest.mark.parametrize("r", np.linspace(0.1, 2 * np.pi - 0.1, 4))
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("ops", [(qml.RX, qml.CRX), (qml.RY, qml.CRY), (qml.RZ, qml.CRZ)])
    def test_conditional_rotations(self, device, r, ops):
        """Test that the quantum conditional operations match the output of
        controlled rotations."""
        dev = qml.device(device, wires=3)

        op, controlled_op = ops

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.Hadamard(0)
            controlled_op(rads, wires=[0, 1])
            return qml.probs(wires=1)

        @qml.qnode(dev)
        @qml.defer_measurements
        def quantum_control_circuit(rads):
            qml.Hadamard(0)
            m_0 = qml.measure(0)
            qml.cond(m_0, op)(rads, wires=1)
            return qml.probs(wires=1)

        normal_probs = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(normal_probs, cond_probs)

    def test_hermitian_queued(self):
        """Test that the defer_measurements transform works with
        qml.Hermitian."""
        rads = 0.3

        mat = np.eye(8)
        measurement = qml.expval(qml.Hermitian(mat, wires=[3, 1, 2]))

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(rads, wires=4)
            qml.apply(measurement)

        tape = qml.defer_measurements(tape)

        assert len(tape.operations) == 1
        assert len(tape.measurements) == 1

        # Check the underlying Controlled instances
        first_ctrl_op = tape.operations[0]
        assert isinstance(first_ctrl_op, qml.ops.op_math.Controlled)
        assert qml.equal(first_ctrl_op.base, qml.RY(rads, 4))

        assert len(tape.measurements) == 1
        assert tape.measurements[0] == measurement

    def test_hamiltonian_queued(self):
        """Test that the defer_measurements transform works with
        qml.Hamiltonian."""
        rads = 0.3
        a = qml.PauliX(3)
        b = qml.PauliX(1)
        c = qml.PauliZ(2)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        H = qml.Hamiltonian(coeffs, obs, grouping_type="qwc")

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(rads, wires=4)
            qml.expval(H)

        tape = qml.defer_measurements(tape)

        assert len(tape.operations) == 1
        assert len(tape.measurements) == 1

        # Check the underlying Controlled instance
        first_ctrl_op = tape.operations[0]
        assert isinstance(first_ctrl_op, qml.ops.op_math.Controlled)
        assert qml.equal(first_ctrl_op.base, qml.RY(rads, 4))
        assert len(tape.measurements) == 1
        assert isinstance(tape.measurements[0], qml.measurements.MeasurementProcess)
        assert tape.measurements[0].obs == H

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("ops", [(qml.RX, qml.CRX), (qml.RY, qml.CRY), (qml.RZ, qml.CRZ)])
    def test_conditional_rotations_assert_zero_state(self, device, ops):
        """Test that the quantum conditional operations applied by controlling
        on the zero outcome match the output of controlled rotations."""
        dev = qml.device(device, wires=3)
        r = 2.345

        op, controlled_op = ops

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.Hadamard(0)
            controlled_op(rads, wires=[0, 1])
            return qml.probs(wires=1)

        @qml.qnode(dev)
        @qml.defer_measurements
        def quantum_control_circuit(rads):
            qml.Hadamard(0)
            qml.PauliX(0)
            m_0 = qml.measure(0)
            qml.cond(m_0 == 0, op)(rads, wires=1)
            return qml.probs(wires=1)

        normal_probs = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(normal_probs, cond_probs)

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    def test_conditional_rotations_with_else(self, device):
        """Test that an else operation can also defined using qml.cond."""
        dev = qml.device(device, wires=2)
        r = 2.345

        op1, controlled_op1 = qml.RY, qml.CRY
        op2, controlled_op2 = qml.RX, qml.CRX

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.Hadamard(0)
            controlled_op1(rads, wires=[0, 1])

            qml.PauliX(0)
            controlled_op2(rads, wires=[0, 1])
            qml.PauliX(0)
            return qml.probs(wires=1)

        @qml.qnode(dev)
        @qml.defer_measurements
        def quantum_control_circuit(rads):
            qml.Hadamard(0)
            m_0 = qml.measure(0)
            qml.cond(m_0, op1, op2)(rads, wires=1)
            return qml.probs(wires=1)

        exp = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(exp, cond_probs)

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
            m_0 = qml.measure(0)
            qml.cond(m_0, op)(phi=par, wires=1)
            return qml.expval(qml.PauliZ(1))

        par = np.array(0.3)

        assert np.allclose(qnode1(par), qnode2(par))

    @pytest.mark.parametrize("control_val, expected", [(0, -1), (1, 1)])
    def test_condition_using_measurement_outcome(self, control_val, expected):
        """Apply a conditional bitflip by selecting the measurement
        outcome."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qnode():
            m_0 = qml.measure(0)
            qml.cond(m_0 == control_val, qml.PauliX)(wires=1)
            return qml.expval(qml.PauliZ(1))

        assert qnode() == expected

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    def test_cond_qfunc(self, device):
        """Test that a qfunc can also used with qml.cond."""
        dev = qml.device(device, wires=2)

        r = 2.324

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.Hadamard(0)

            qml.CNOT(wires=[0, 1])
            qml.CRY(rads, wires=[0, 1])
            qml.CZ(wires=[0, 1])
            return qml.probs(wires=1)

        def f(x):
            qml.PauliX(1)
            qml.RY(x, wires=1)
            qml.PauliZ(1)

        @qml.qnode(dev)
        @qml.defer_measurements
        def quantum_control_circuit(rads):
            qml.Hadamard(0)
            m_0 = qml.measure(0)
            qml.cond(m_0, f)(r)
            return qml.probs(wires=1)

        exp = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(exp, cond_probs)

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    def test_cond_qfunc_with_else(self, device):
        """Test that a qfunc can also used with qml.cond even when an else
        qfunc is provided."""
        dev = qml.device(device, wires=2)

        x = 0.3
        y = 3.123

        @qml.qnode(dev)
        def normal_circuit(x, y):
            qml.RY(x, wires=1)

            qml.ctrl(f, 1)(y)

            # Flip the qubit before/after to control on 0
            qml.PauliX(1)
            qml.ctrl(g, 1)(y)
            qml.PauliX(1)
            return qml.probs(wires=[0])

        def f(a):
            qml.PauliX(0)
            qml.RY(a, wires=0)
            qml.PauliZ(0)

        def g(a):
            qml.RX(a, wires=0)
            qml.PhaseShift(a, wires=0)

        @qml.qnode(dev)
        def cond_qnode(x, y):
            qml.RY(x, wires=1)
            m_0 = qml.measure(1)
            qml.cond(m_0, f, g)(y)
            return qml.probs(wires=[0])

        assert np.allclose(normal_circuit(x, y), cond_qnode(x, y))
        assert np.allclose(qml.matrix(normal_circuit)(x, y), qml.matrix(cond_qnode)(x, y))


class TestTemplates:
    """Tests templates being conditioned on mid-circuit measurement outcomes."""

    def test_basis_state_prep(self):
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
            m_0 = qml.measure(0)
            qml.cond(m_0, template)(basis_state, wires=range(1, 5))
            return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4))

        dev = qml.device("default.qubit", wires=2)

        assert np.allclose(qnode1(), qnode2())

        assert len(qnode1.qtape.operations) == len(qnode2.qtape.operations)
        assert len(qnode1.qtape.measurements) == len(qnode2.qtape.measurements)

        # Check the operations
        for op1, op2 in zip(qnode1.qtape.operations, qnode2.qtape.operations):
            assert type(op1) == type(op2)
            assert np.allclose(op1.data, op2.data)

        # Check the measurements
        for op1, op2 in zip(qnode1.qtape.measurements, qnode2.qtape.measurements):
            assert type(op1) == type(op2)
            assert np.allclose(op1.data, op2.data)

    def test_angle_embedding(self):
        """Test the angle embedding template conditioned on mid-circuit
        measurement outcomes."""
        template = qml.AngleEmbedding
        feature_vector = [1, 2, 3]

        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def qnode1():
            qml.Hadamard(0)
            qml.ctrl(template, control=0)(features=feature_vector, wires=range(1, 5), rotation="Z")
            return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4))

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode2():
            qml.Hadamard(0)
            m_0 = qml.measure(0)
            qml.cond(m_0, template)(features=feature_vector, wires=range(1, 5), rotation="Z")
            return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4))

        dev = qml.device("default.qubit", wires=2)
        res1 = qnode1()
        res2 = qnode2()

        assert np.allclose(res1, res2)

        assert len(qnode1.qtape.operations) == len(qnode2.qtape.operations)
        assert len(qnode1.qtape.measurements) == len(qnode2.qtape.measurements)

        # Check the operations
        for op1, op2 in zip(qnode1.qtape.operations, qnode2.qtape.operations):
            assert type(op1) == type(op2)
            assert np.allclose(op1.data, op2.data)

        # Check the measurements
        for op1, op2 in zip(qnode1.qtape.measurements, qnode2.qtape.measurements):
            assert type(op1) == type(op2)
            assert np.allclose(op1.data, op2.data)

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
            m_0 = qml.measure(0)
            qml.cond(m_0, template)(parameters, wires=range(1, 3))
            return qml.expval(qml.PauliZ(1) @ qml.PauliZ(2))

        shape = template.shape(n_layers=2, n_wires=num_wires)
        weights = np.random.random(size=shape)

        assert np.allclose(qnode1(weights), qnode2(weights))

        assert len(qnode1.qtape.operations) == len(qnode2.qtape.operations)
        assert len(qnode1.qtape.measurements) == len(qnode2.qtape.measurements)

        # Check the operations
        for op1, op2 in zip(qnode1.qtape.operations, qnode2.qtape.operations):
            assert type(op1) == type(op2)
            assert np.allclose(op1.data, op2.data)

        # Check the measurements
        for op1, op2 in zip(qnode1.qtape.measurements, qnode2.qtape.measurements):
            assert type(op1) == type(op2)
            assert np.allclose(op1.data, op2.data)


class TestDrawing:
    """Tests drawing circuits with mid-circuit measurements and conditional
    operations that have been transformed"""

    def test_drawing(self):
        """Test that drawing a func with mid-circuit measurements works and
        that controlled operations are drawn for conditional operations."""

        def qfunc():
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(0.312, wires=1)

            m_2 = qml.measure(2)
            qml.cond(m_2, qml.RY)(0.312, wires=1)
            return qml.expval(qml.PauliZ(1))

        dev = qml.device("default.qubit", wires=4)

        transformed_qfunc = qml.transforms.defer_measurements(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        expected = (
            "0: ─╭●──────────────────┤     \n"
            "1: ─╰RY(0.31)─╭RY(0.31)─┤  <Z>\n"
            "2: ───────────╰●────────┤     "
        )
        assert qml.draw(transformed_qnode)() == expected


class TestMeasurementValueManipulation:
    def test_apply_function_to_measurement(self):

        m = MeasurementValue("m", fn=lambda v: v)

        sin_of_m = m.apply(np.sin)
        assert sin_of_m[0] == 0.0
        assert sin_of_m[1] == np.sin(1)

    def test_add_to_measurements(self):
        m0 = MeasurementValue("m0", fn=lambda v: v)
        m1 = MeasurementValue("m1", fn=lambda v: v)
        sum_of_measurements = m0 + m1
        assert sum_of_measurements[0] == 0
        assert sum_of_measurements[1] == 1
        assert sum_of_measurements[2] == 1
        assert sum_of_measurements[3] == 2

    def test_equality_with_scalar(self):
        m = MeasurementValue("m", fn=lambda v: v)
        m_eq = m == 0
        assert m_eq[0] == True  # confirming value is actually eq to True, not just truthy
        assert m_eq[1] == False

    def test_inversion(self):
        m = MeasurementValue("m", fn=lambda v: v)
        m_inversion = ~m
        assert m_inversion[0] == True
        assert m_inversion[1] == False

    def test_lt(self):
        m = MeasurementValue("m", fn=lambda v: v)
        m_inversion = m < 0.5
        assert m_inversion[0] == True
        assert m_inversion[1] == False

    def test_gt(self):
        m = MeasurementValue("m", fn=lambda v: v)
        m_inversion = m > 0.5
        assert m_inversion[0] == False
        assert m_inversion[1] == True

    def test_merge_measurements_values_dependant_on_same_measurement(self):
        m0 = MeasurementValue("m", fn=lambda v: v)
        m1 = MeasurementValue("m", fn=lambda v: v)
        combined = m0 + m1
        assert combined[0] == 0
        assert combined[1] == 2

    def test_combine_measurement_value_with_non_measurement(self):
        m0 = MeasurementValue("m", fn=lambda v: v)
        out = m0 + 10
        assert out[0] == 10
        assert out[1] == 11

    def test_str(self):
        m = MeasurementValue("m", fn=lambda v: v)
        assert str(m) == "if m=0 => 0\nif m=1 => 1"

    def test_complex_str(self):
        a = MeasurementValue("a", fn=lambda v: v)
        b = MeasurementValue("b", fn=lambda v: v)
        assert str(a + b) == "if a=0,b=0 => 0\nif a=0,b=1 => 1\nif a=1,b=0 => 1\nif a=1,b=1 => 2"
