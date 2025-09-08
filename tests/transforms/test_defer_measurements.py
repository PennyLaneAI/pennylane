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
import math
import re

# pylint: disable=too-few-public-methods, too-many-arguments
from functools import partial

import pytest
from device_shots_to_analytic import shots_to_analytic

import pennylane as qml
import pennylane.numpy as np
from pennylane.devices import DefaultQubit
from pennylane.exceptions import DeviceError
from pennylane.measurements import MeasurementValue, MidMeasureMP
from pennylane.ops import Controlled


def test_broadcasted_postselection(mocker):
    """Test that broadcast_expand is used iff broadcasting with postselection."""
    spy = mocker.spy(qml.transforms, "broadcast_expand")

    # Broadcasting with postselection
    tape1 = qml.tape.QuantumScript(
        [qml.RX([0.1, 0.2], 0), MidMeasureMP(0, postselect=1), qml.CNOT([0, 1])],
        [qml.probs(wires=[0])],
    )
    _, _ = qml.defer_measurements(tape1)

    assert spy.call_count == 1

    # Broadcasting without postselection
    tape2 = qml.tape.QuantumScript(
        [qml.RX([0.1, 0.2], 0), MidMeasureMP(0), qml.CNOT([0, 1])],
        [qml.probs(wires=[0])],
    )
    _, _ = qml.defer_measurements(tape2)

    assert spy.call_count == 1

    # Postselection without broadcasting
    tape3 = qml.tape.QuantumScript(
        [qml.RX(0.1, 0), MidMeasureMP(0, postselect=1), qml.CNOT([0, 1])],
        [qml.probs(wires=[0])],
    )
    _, _ = qml.defer_measurements(tape3)

    assert spy.call_count == 1

    # No postselection, no broadcasting
    tape4 = qml.tape.QuantumScript(
        [qml.RX(0.1, 0), MidMeasureMP(0), qml.CNOT([0, 1])],
        [qml.probs(wires=[0])],
    )
    _, _ = qml.defer_measurements(tape4)

    assert spy.call_count == 1


def test_broadcasted_postselection_with_sample_error():
    """Test that an error is raised if returning qml.sample if postselecting with broadcasting"""
    tape = qml.tape.QuantumScript(
        [qml.RX([0.1, 0.2], 0), MidMeasureMP(0, postselect=1)], [qml.sample(wires=0)], shots=10
    )
    dev = qml.device("default.qubit")

    with pytest.raises(ValueError, match="Returning qml.sample is not supported when"):
        qml.defer_measurements(tape)

    @qml.defer_measurements
    @qml.set_shots(10)
    @qml.qnode(dev)
    def circuit():
        qml.RX([0.1, 0.2], 0)
        qml.measure(0, postselect=1)
        return qml.sample(wires=0)

    with pytest.raises(ValueError, match="Returning qml.sample is not supported when"):
        _ = circuit()


def test_allow_postselect():
    """Tests that allow_postselect=False forbids postselection on mid-circuit measurements."""

    circuit = qml.tape.QuantumScript([MidMeasureMP(wires=0, postselect=0)], [qml.expval(qml.Z(0))])
    with pytest.raises(ValueError, match="Postselection is not allowed"):
        _, __ = qml.defer_measurements(circuit, allow_postselect=False)


def test_postselection_error_with_wrong_device():
    """Test that an error is raised when postselection is used with a device
    other than `default.qubit`."""
    dev = qml.device("default.mixed", wires=2)

    @qml.defer_measurements
    @qml.qnode(dev)
    def circ():
        qml.measure(0, postselect=1)
        return qml.probs(wires=[0])

    with pytest.raises(
        DeviceError,
        match=re.escape(
            "Operator Projector(array([1]), wires=[0]) not supported with default.mixed and does not provide a decomposition."
        ),
    ):
        _ = circ()


@pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
def test_postselect_mode(postselect_mode, mocker):
    """Test that invalid shots are discarded if requested"""
    shots = 100
    postselect_value = 1
    dev = qml.device("default.qubit")
    spy = mocker.spy(qml.defer_measurements, "_transform")

    @qml.set_shots(shots)
    @qml.qnode(dev, postselect_mode=postselect_mode, mcm_method="deferred")
    def f(x):
        qml.RX(x, 0)
        _ = qml.measure(0, postselect=postselect_value)
        return qml.sample(wires=[0])

    res = f(np.pi / 4)
    spy.assert_called_once()

    if postselect_mode == "hw-like":
        assert len(res) < shots
    else:
        assert len(res) == shots
    assert np.allclose(res, postselect_value)


@pytest.mark.parametrize(
    "mp, err_msg",
    [
        (qml.state(), "Cannot use StateMP as a measurement when"),
        (qml.probs(), "Cannot use ProbabilityMP as a measurement without"),
        (qml.sample(), "Cannot use SampleMP as a measurement without"),
        (qml.counts(), "Cannot use CountsMP as a measurement without"),
    ],
)
def test_unsupported_measurements(mp, err_msg):
    """Test that using unsupported measurements raises an error."""
    tape = qml.tape.QuantumScript([MidMeasureMP(0)], [mp])

    with pytest.raises(ValueError, match=err_msg):
        _, _ = qml.defer_measurements(tape)


@pytest.mark.parametrize(
    "mp, compose_mv",
    [
        (qml.expval, True),
        (qml.var, True),
        (qml.probs, False),
        (qml.sample, True),
        (qml.sample, False),
        (qml.counts, True),
        (qml.counts, False),
    ],
)
def test_multi_mcm_stats_same_wire(mp, compose_mv):
    """Test that a tape collecting statistics on multiple mid-circuit measurements when
    they measure the same wire is transformed correctly."""
    mp1 = MidMeasureMP(0, id="foo")
    mp2 = MidMeasureMP(0, id="bar")
    mv1 = MeasurementValue([mp1], None)
    mv2 = MeasurementValue([mp2], None)

    mv = mv1 * mv2 if compose_mv else [mv1, mv2]
    tape = qml.tape.QuantumScript([qml.PauliX(0), mp1, mp2], [mp(op=mv)], shots=10)
    [deferred_tape], _ = qml.defer_measurements(tape)

    emp1 = MidMeasureMP(1, id="foo")
    emp2 = MidMeasureMP(2, id="bar")
    emv1 = MeasurementValue([emp1], None)
    emv2 = MeasurementValue([emp2], None)
    emv = emv1 * emv2 if compose_mv else [emv1, emv2]

    assert deferred_tape.operations == [qml.PauliX(0), qml.CNOT([0, 1]), qml.CNOT([0, 2])]
    assert deferred_tape.measurements == [mp(op=emv)]


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
            qml.measure(1)
            return qml.expval(qml.PauliZ(0))

        res1 = qnode1()
        res2 = qnode2()
        assert res1 == res2
        assert isinstance(res1, type(res2))
        assert res1.shape == res2.shape

        tape1 = qml.workflow.construct_tape(qnode1)()
        tape2 = qml.workflow.construct_tape(qnode2)()
        assert len(tape2.operations) == 0
        assert len(tape1.measurements) == len(tape2.measurements)

        # Check the measurements
        for op1, op2 in zip(tape1.measurements, tape2.measurements):
            assert isinstance(op1, type(op2))

    def test_reuse_wire_after_measurement(self):
        """Test that wires can be reused after measurement."""
        dev = qml.device("default.qubit", wires=2)

        @qml.defer_measurements
        @qml.qnode(dev)
        def qnode():
            qml.Hadamard(0)
            qml.measure(0)
            qml.PauliZ(0)
            return qml.expval(qml.PauliX(0))

        _ = qnode()

    def test_no_new_wires_without_reuse(self, mocker):
        """Test that new wires are not added if a measured wire is not reused."""
        dev = qml.device("default.qubit", wires=3)

        # Quantum teleportation
        @qml.qnode(dev)
        def qnode1(phi):
            qml.RX(phi, 0)
            qml.Hadamard(1)
            qml.CNOT([1, 2])
            qml.CNOT([0, 1])
            qml.Hadamard(0)

            m0 = qml.measure(0)
            qml.cond(m0, qml.PauliZ)(2)
            m1 = qml.measure(1)
            qml.cond(m1, qml.PauliX)(2)
            return qml.expval(qml.PauliZ(2))

        # Prepare wire 0 in arbitrary state
        @qml.qnode(dev)
        def qnode2(phi):
            qml.RX(phi, 0)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(qml.defer_measurements, "_transform")

        # Outputs should match
        assert np.isclose(qnode1(np.pi / 4), qnode2(np.pi / 4))
        assert spy.call_count == 2  # once per device preprocessing

        tape1 = qml.workflow.construct_tape(qnode1)(np.pi / 4)
        deferred_tapes, _ = qml.defer_measurements(tape1)
        deferred_tape = deferred_tapes[0]
        assert isinstance(deferred_tape.operations[5], Controlled)
        qml.assert_equal(deferred_tape.operations[5].base, qml.PauliZ(2))
        assert deferred_tape.operations[5].hyperparameters["control_wires"] == qml.wires.Wires(0)

        qml.assert_equal(deferred_tape.operations[6], qml.CNOT([1, 2]))

    def test_new_wires_after_reuse(self, mocker):
        """Test that a new wire is added for every measurement after which
        the wire is reused."""
        dev = qml.device("default.qubit", wires=4)
        spy = mocker.spy(qml.defer_measurements, "_transform")

        @qml.defer_measurements
        @qml.qnode(dev)
        def qnode1(phi, theta):
            qml.RX(phi, 0)
            m0 = qml.measure(0, reset=True)  # Reused measurement, one new wire added
            qml.cond(m0, qml.Hadamard)(1)
            m1 = qml.measure(1)  # No reuse
            qml.RY(theta, 2)
            qml.cond(m1, qml.RY)(-theta, 2)
            return qml.expval(qml.PauliZ(2))

        res1 = qnode1(np.pi / 4, 3 * np.pi / 4)

        @qml.defer_measurements
        @qml.qnode(dev)
        def qnode2(phi, theta):
            qml.RX(phi, 0)
            m0 = qml.measure(0)  # No reuse
            qml.cond(m0, qml.Hadamard)(1)
            m1 = qml.measure(1)  # No reuse
            qml.RY(theta, 2)
            qml.cond(m1, qml.RY)(-theta, 2)
            return qml.expval(qml.PauliZ(2))

        res2 = qnode2(np.pi / 4, 3 * np.pi / 4)

        assert spy.call_count == 4

        tape1 = qml.workflow.construct_tape(qnode1)(np.pi / 4, 3 * np.pi / 4)
        deferred_tapes1, _ = qml.defer_measurements(tape1)
        deferred_tape1 = deferred_tapes1[0]
        assert len(deferred_tape1.wires) == 4
        assert len(deferred_tape1.operations) == 6

        assert np.allclose(res1, res2)

        tape2 = qml.workflow.construct_tape(qnode2)(np.pi / 4, 3 * np.pi / 4)
        deferred_tapes2, _ = qml.defer_measurements(tape2)
        deferred_tape2 = deferred_tapes2[0]
        assert len(deferred_tape2.wires) == 3
        assert len(deferred_tape2.operations) == 4

    @pytest.mark.parametrize("reduce_postselected", [None, True, False])
    @pytest.mark.parametrize("shots", [None, 1000])
    @pytest.mark.parametrize("phi", np.linspace(np.pi / 2, 7 * np.pi / 2, 6))
    def test_single_postselection_qnode(self, phi, shots, reduce_postselected):
        """Test that a qnode with a single mid-circuit measurements with postselection
        is transformed correctly by defer_measurements"""
        dev = DefaultQubit()
        dev = shots_to_analytic(dev)

        dm_transform = qml.defer_measurements
        if reduce_postselected is not None:
            dm_transform = partial(dm_transform, reduce_postselected=reduce_postselected)
        else:
            # overwrite None with the expected default value True to determine expected outputs
            reduce_postselected = True

        @qml.set_shots(shots=shots)
        @qml.qnode(dev)
        @dm_transform
        def circ1(phi):
            qml.RX(phi, wires=0)
            # Postselecting on |1> on wire 0 means that the probability of measuring
            # |1> on wire 0 is 1
            m = qml.measure(0, postselect=1)
            qml.cond(m, qml.PauliX)(wires=1)
            # Probability of measuring |1> on wire 1 should be 1
            return qml.probs(wires=1)

        assert np.allclose(circ1(phi), [0, 1])

        expected_circuit = [
            qml.RX(phi, 0),
            qml.Projector([1], wires=0),
            qml.X(1) if reduce_postselected else qml.CNOT([0, 1]),
            qml.probs(wires=1),
        ]

        tape1 = qml.workflow.construct_tape(circ1)(phi)
        assert len(tape1) == len(expected_circuit)
        for op, expected_op in zip(tape1, expected_circuit):
            qml.assert_equal(op, expected_op)

    @pytest.mark.parametrize("reduce_postselected", [None, True, False])
    @pytest.mark.parametrize("shots", [None, 1000])
    @pytest.mark.parametrize("phi", np.linspace(np.pi / 2, 7 * np.pi / 2, 6))
    def test_some_postselection_qnode(
        self, phi, shots, reduce_postselected, tol, tol_stochastic, seed
    ):
        """Test that a qnode with some mid-circuit measurements with postselection
        is transformed correctly by defer_measurements"""
        dev = DefaultQubit(seed=seed)
        dev = shots_to_analytic(dev)

        dm_transform = qml.defer_measurements
        if reduce_postselected is not None:
            dm_transform = partial(dm_transform, reduce_postselected=reduce_postselected)
        else:
            # overwrite None with the expected default value True to determine expected outputs
            reduce_postselected = True

        @qml.set_shots(shots=shots)
        @qml.qnode(dev)
        @dm_transform
        def circ1(phi):
            qml.RX(phi, wires=0)
            qml.RX(phi, wires=2)
            # Postselecting on |1> on wire 0 means that the probability of measuring
            # |1> on wire 0 is 1
            m0 = qml.measure(0, postselect=1)
            m1 = qml.measure(2)
            qml.cond(m0 & m1, qml.PauliX)(wires=1)
            # Probability of measuring |1> on wire 1 should be 1
            return qml.probs(wires=1)

        atol = tol if shots is None else tol_stochastic
        expected_out = [np.cos(phi / 2) ** 2, np.sin(phi / 2) ** 2]
        assert np.allclose(circ1(phi), expected_out, atol=atol, rtol=0)

        expected_circuit = [
            qml.RX(phi, 0),
            qml.RX(phi, 2),
            qml.Projector([1], wires=0),
            qml.CNOT([2, 1]) if reduce_postselected else qml.Toffoli([0, 2, 1]),
            qml.probs(wires=1),
        ]

        tape1 = qml.workflow.construct_tape(circ1)(phi)
        assert len(tape1) == len(expected_circuit)
        for op, expected_op in zip(tape1, expected_circuit):
            qml.assert_equal(op, expected_op)

    @pytest.mark.parametrize("reduce_postselected", [None, True, False])
    @pytest.mark.parametrize("shots", [None, 1000])
    @pytest.mark.parametrize("phi", np.linspace(np.pi / 4, 4 * np.pi, 4))
    @pytest.mark.parametrize("theta", np.linspace(np.pi / 3, 3 * np.pi, 4))
    def test_all_postselection_qnode(
        self, phi, theta, shots, reduce_postselected, tol, tol_stochastic
    ):
        """Test that a qnode with all mid-circuit measurements with postselection
        is transformed correctly by defer_measurements"""
        dev = DefaultQubit()
        dev = shots_to_analytic(dev)

        # Initializing mid circuit measurements here so that id can be controlled (affects
        # wire ordering for qml.cond)
        mp0 = MidMeasureMP(wires=0, postselect=0, id=0)
        mv0 = MeasurementValue([mp0], lambda v: v)
        mp1 = MidMeasureMP(wires=1, postselect=0, id=1)
        mv1 = MeasurementValue([mp1], lambda v: v)
        mp2 = MidMeasureMP(wires=2, reset=True, postselect=1, id=2)
        mv2 = MeasurementValue([mp2], lambda v: v)

        dm_transform = qml.defer_measurements
        if reduce_postselected is not None:
            dm_transform = partial(dm_transform, reduce_postselected=reduce_postselected)
        else:
            # Override None with the expected default value True to determine expected outputs
            reduce_postselected = True

        @qml.set_shots(shots=shots)
        @qml.qnode(dev)
        @dm_transform
        def circ1(phi, theta):
            qml.RX(phi, 0)
            qml.apply(mp0)
            qml.CNOT([0, 1])
            qml.apply(mp1)
            qml.cond(~(mv0 & mv1), qml.RY)(theta, wires=2)
            qml.apply(mp2)
            qml.cond(mv2, qml.PauliX)(1)
            return qml.probs(wires=[0, 1, 2])

        @qml.qnode(dev)
        def circ2():
            # To add wire 0 to tape
            qml.Identity(0)
            qml.PauliX(1)
            qml.Identity(2)
            return qml.probs(wires=[0, 1, 2])

        atol = tol if shots is None else tol_stochastic
        assert np.allclose(circ1(phi, theta), circ2(), atol=atol, rtol=0)

        expected_first_cond_block = (
            [qml.RY(theta, wires=[2])]
            if reduce_postselected
            else [
                Controlled(qml.RY(theta, wires=[2]), control_wires=[3, 4], control_values=cv)
                for cv in ([False, False], [False, True], [True, False])
            ]
        )
        expected_circuit = (
            [
                qml.RX(phi, wires=0),
                qml.Projector([0], wires=0),
                qml.CNOT([0, 3]),
                qml.CNOT([0, 1]),
                qml.Projector([0], wires=1),
                qml.CNOT([1, 4]),
            ]
            + expected_first_cond_block
            + [
                qml.Projector([1], wires=2),
                qml.CNOT([2, 5]),
                qml.PauliX(2),
                qml.PauliX(1) if reduce_postselected else qml.CNOT([5, 1]),
                qml.probs(wires=[0, 1, 2]),
            ]
        )

        tape1 = qml.workflow.construct_tape(circ1)(phi, theta)
        assert len(tape1) == len(expected_circuit)
        for op, expected_op in zip(tape1, expected_circuit):
            qml.assert_equal(op, expected_op)

    @pytest.mark.parametrize("shots", [None, 1000, [1000, 1000]])
    def test_measurement_statistics_single_wire(self, shots, seed):
        """Test that users can collect measurement statistics on
        a single mid-circuit measurement."""
        dev = DefaultQubit(seed=seed)
        dev = shots_to_analytic(dev)

        @qml.set_shots(shots=shots)
        @qml.defer_measurements
        @qml.qnode(dev)
        def circ1(x):
            qml.RX(x, 0)
            m0 = qml.measure(0)
            return qml.probs(op=m0)

        dev = DefaultQubit(seed=seed)
        dev = shots_to_analytic(dev)

        @qml.set_shots(shots=shots)
        @qml.qnode(dev)
        def circ2(x):
            qml.RX(x, 0)
            return qml.probs(wires=[0])

        param = 1.5
        assert np.allclose(circ1(param), circ2(param))

    @pytest.mark.parametrize("shots", [None, 2000, [2000, 2000]])
    def test_measured_value_wires_mapped(self, shots, tol, tol_stochastic):
        """Test that collecting statistics on a measurement value works correctly
        when the measured wire is reused."""
        dev = DefaultQubit()
        dev = shots_to_analytic(dev)

        @qml.set_shots(shots=shots)
        @qml.qnode(dev)
        @qml.defer_measurements
        def circ1(x):
            qml.RX(x, 0)
            m0 = qml.measure(0)
            qml.PauliX(0)
            return qml.probs(op=m0)

        dev = DefaultQubit()
        dev = shots_to_analytic(dev)

        @qml.set_shots(shots=shots)
        @qml.qnode(dev)
        def circ2(x):
            qml.RX(x, 0)
            return qml.probs(wires=[0])

        param = 1.5
        atol = tol if shots is None else tol_stochastic
        assert np.allclose(circ1(param), circ2(param), atol=atol, rtol=0)

        expected_ops = [qml.RX(param, 0), qml.CNOT([0, 1]), qml.PauliX(0)]
        tape1 = qml.workflow.construct_tape(circ1)(param)
        assert tape1.operations == expected_ops

        assert len(tape1.measurements) == 1
        mp = tape1.measurements[0]
        assert isinstance(mp, qml.measurements.ProbabilityMP)
        assert mp.mv is not None
        assert mp.mv.wires == qml.wires.Wires([1])

    @pytest.mark.parametrize("shots", [None, 1000, [1000, 1000]])
    def test_terminal_measurements(self, shots, seed):
        """Test that mid-circuit measurement statistics and terminal measurements
        can be made together."""
        # Using DefaultQubit to allow non-commuting measurements
        dev = DefaultQubit(seed=seed)
        dev = shots_to_analytic(dev)

        @qml.set_shots(shots=shots)
        @qml.defer_measurements
        @qml.qnode(dev)
        def circ1(x, y):
            qml.RX(x, 0)
            m0 = qml.measure(0)
            qml.RY(y, 1)
            return qml.expval(qml.PauliX(1)), qml.probs(op=m0)

        dev = DefaultQubit(seed=seed)
        dev = shots_to_analytic(dev)

        @qml.set_shots(shots=shots)
        @qml.qnode(dev)
        def circ2(x, y):
            qml.RX(x, 0)
            qml.RY(y, 1)
            return qml.expval(qml.PauliX(1)), qml.probs(wires=[0])

        params = [1.5, 2.5]
        if isinstance(shots, list):
            for out1, out2 in zip(circ1(*params), circ2(*params)):
                for o1, o2 in zip(out1, out2):
                    assert np.allclose(o1, o2)
        else:
            assert all(
                np.allclose(out1, out2) for out1, out2 in zip(circ1(*params), circ2(*params))
            )

    def test_measure_between_ops(self):
        """Test that a quantum function that contains one operation before and
        after a mid-circuit measurement yields the correct results and is
        transformed correctly."""
        dev = qml.device("default.qubit", wires=3)
        dev = shots_to_analytic(dev)

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

        tape1 = qml.workflow.construct_tape(qnode1)()
        tape2 = qml.workflow.construct_tape(qnode2)()
        assert len(tape2.operations) == len(tape1.operations)
        assert len(tape1.measurements) == len(tape2.measurements)

        # Check the operations
        for op1, op2 in zip(tape1.operations, tape2.operations):
            assert isinstance(op1, type(op2))
            assert op1.data == op2.data

        # Check the measurements
        for op1, op2 in zip(tape1.measurements, tape2.measurements):
            assert isinstance(op1, type(op2))

    @pytest.mark.parametrize("mid_measure_wire, tp_wires", [(0, [1, 2, 3]), (0, [3, 1, 2])])
    def test_measure_with_tensor_obs(self, mid_measure_wire, tp_wires):
        """Test that the defer_measurements transform works well even with
        tensor observables in the tape."""
        # pylint: disable=protected-access

        with qml.queuing.AnnotatedQueue() as q:
            qml.measure(mid_measure_wire)
            qml.expval(qml.prod(*[qml.PauliZ(w) for w in tp_wires]))

        tape = qml.tape.QuantumScript.from_queue(q)
        tape, _ = qml.defer_measurements(tape)
        tape = tape[0]
        # Check the operations and measurements in the tape
        assert len(tape.measurements) == 1

        measurement = tape.measurements[0]
        assert isinstance(measurement, qml.measurements.MeasurementProcess)

        tensor = measurement.obs
        assert len(tensor.operands) == 3

        for idx, ob in enumerate(tensor.operands):
            assert isinstance(ob, qml.PauliZ)
            assert ob.wires == qml.wires.Wires(tp_wires[idx])

    def test_cv_op_error(self):
        """Test that CV operations are not supported."""
        dev = qml.device("default.gaussian", wires=3)

        @qml.defer_measurements
        @qml.qnode(dev)
        def qnode():
            qml.measure(0)
            qml.Rotation(0.123, wires=[0])
            return qml.expval(qml.NumberOperator(1))

        with pytest.raises(
            ValueError, match="Continuous variable operations and observables are not supported"
        ):
            qnode()

    def test_cv_obs_error(self):
        """Test that CV observables are not supported."""
        dev = qml.device("default.gaussian", wires=3)

        @qml.defer_measurements
        @qml.qnode(dev)
        def qnode():
            qml.measure(0)
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
        first_par = 0.1
        sec_par = 0.3

        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(4, reset=True)
            qml.cond(m_0, qml.RY)(first_par, wires=1)

            m_1 = qml.measure(3)
            qml.cond(m_1, qml.RZ)(sec_par, wires=1)
            qml.apply(terminal_measurement)

        tape = qml.tape.QuantumScript.from_queue(q)

        tapes, _ = qml.defer_measurements(tape)
        tape = tapes[0]

        assert len(tape.operations) == 4
        assert len(tape.measurements) == 1

        # Check the two underlying Controlled instances
        first_ctrl_op = tape.operations[2]
        assert isinstance(first_ctrl_op, Controlled)
        qml.assert_equal(first_ctrl_op.base, qml.RY(first_par, 1))

        sec_ctrl_op = tape.operations[3]
        assert isinstance(sec_ctrl_op, Controlled)
        qml.assert_equal(sec_ctrl_op.base, qml.RZ(sec_par, 1))

        assert tape.measurements[0] == terminal_measurement

    def test_correct_ops_in_tape_inversion(self):
        """Test that the underlying tape contains the correct operations if a
        measurement value was inverted."""
        first_par = 0.1
        terminal_measurement = qml.expval(qml.PauliZ(1))

        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.cond(~m_0, qml.RY)(first_par, wires=1)
            qml.apply(terminal_measurement)

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.defer_measurements(tape)
        tape = tapes[0]
        # Conditioned on 0 as the control value, PauliX is applied before and after
        assert len(tape.operations) == 1
        assert len(tape.measurements) == 1

        # Check the two underlying Controlled instance
        ctrl_op = tape.operations[0]
        assert isinstance(ctrl_op, Controlled)
        qml.assert_equal(ctrl_op.base, qml.RY(first_par, 1))

        assert ctrl_op.wires == qml.wires.Wires([0, 1])

    def test_correct_ops_in_tape_assert_zero_state(self):
        """Test that the underlying tape contains the correct operations if a
        conditional operation was applied in the zero state case.

        Note: this case is the same as inverting right after obtaining a
        measurement value."""
        first_par = 0.1
        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.cond(m_0 == 0, qml.RY)(first_par, wires=1)
            qml.expval(qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.defer_measurements(tape)
        tape = tapes[0]
        # Conditioned on 0 as the control value, PauliX is applied before and after
        assert len(tape.operations) == 1
        assert len(tape.measurements) == 1

        # Check the underlying Controlled instance
        ctrl_op = tape.operations[0]
        assert isinstance(ctrl_op, Controlled)
        qml.assert_equal(ctrl_op.base, qml.RY(first_par, 1))

    @pytest.mark.parametrize("rads", np.linspace(0.0, np.pi, 3))
    def test_quantum_teleportation(self, rads):
        """Test quantum teleportation."""

        terminal_measurement = qml.probs(wires=2)

        with qml.queuing.AnnotatedQueue() as q:
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
            m_1 = qml.measure(1, reset=True)

            # Given Alice's measurements, Bob performs one of four operations on his half of the EPR pair and
            # recovers the original quantum state.
            qml.cond(m_1, qml.RX)(math.pi, wires=2)
            qml.cond(m_0, qml.RZ)(math.pi, wires=2)

            qml.apply(terminal_measurement)

        tape = qml.tape.QuantumScript.from_queue(q)

        tapes, _ = qml.defer_measurements(tape)
        tape = tapes[0]

        assert (
            len(tape.operations) == 5 + 1 + 1 + 2
        )  # 5 regular ops + 1 measurement op + 1 reset op + 2 conditional ops

        assert len(tape.measurements) == 1

        # Check the each operation
        op1 = tape.operations[0]
        assert isinstance(op1, qml.RY)
        assert op1.wires == qml.wires.Wires(0)
        assert op1.data == (rads,)

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

        # Check the two underlying CNOTs for storing measurement state
        meas_op1 = tape.operations[5]
        assert isinstance(meas_op1, qml.CNOT)
        assert meas_op1.wires == qml.wires.Wires([1, 3])

        meas_op2 = tape.operations[6]
        assert isinstance(meas_op2, qml.CNOT)
        assert meas_op2.wires == qml.wires.Wires([3, 1])

        # Check the two underlying Controlled instances
        ctrl_op1 = tape.operations[7]
        assert isinstance(ctrl_op1, Controlled)
        qml.assert_equal(ctrl_op1.base, qml.RX(math.pi, 2))
        assert ctrl_op1.wires == qml.wires.Wires([3, 2])

        ctrl_op2 = tape.operations[8]
        assert isinstance(ctrl_op2, Controlled)
        qml.assert_equal(ctrl_op2.base, qml.RZ(math.pi, 2))
        assert ctrl_op2.wires == qml.wires.Wires([0, 2])

        # Check the measurement
        qml.assert_equal(tape.measurements[0], terminal_measurement)

    @pytest.mark.parametrize("r", np.linspace(0.1, 2 * np.pi - 0.1, 4))
    @pytest.mark.parametrize(
        "device",
        [
            "default.qubit",
            "default.mixed",
            "lightning.qubit",
        ],
    )
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

        @qml.defer_measurements
        @qml.qnode(dev)
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

        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0, reset=True)
            qml.cond(m_0, qml.RY)(rads, wires=4)
            qml.apply(measurement)

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.defer_measurements(tape)
        tape = tapes[0]

        assert len(tape.operations) == 3
        assert len(tape.measurements) == 1

        # Check the underlying CNOT for storing measurement state
        meas_op1 = tape.operations[0]
        assert isinstance(meas_op1, qml.CNOT)
        assert meas_op1.wires == qml.wires.Wires([0, 5])

        # Check the underlying CNOT for resetting measured wire
        meas_op1 = tape.operations[1]
        assert isinstance(meas_op1, qml.CNOT)
        assert meas_op1.wires == qml.wires.Wires([5, 0])

        # Check the underlying Controlled instances
        first_ctrl_op = tape.operations[2]
        assert isinstance(first_ctrl_op, Controlled)
        qml.assert_equal(first_ctrl_op.base, qml.RY(rads, 4))

        assert len(tape.measurements) == 1
        qml.assert_equal(tape.measurements[0], measurement)

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

        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(rads, wires=4)
            qml.expval(H)

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = qml.defer_measurements(tape)
        tape = tapes[0]
        assert len(tape.operations) == 1
        assert len(tape.measurements) == 1

        # Check the underlying Controlled instance
        first_ctrl_op = tape.operations[0]
        assert isinstance(first_ctrl_op, Controlled)
        qml.assert_equal(first_ctrl_op.base, qml.RY(rads, 4))
        assert len(tape.measurements) == 1
        assert isinstance(tape.measurements[0], qml.measurements.MeasurementProcess)
        qml.assert_equal(tape.measurements[0].obs, H)

    @pytest.mark.parametrize(
        "device",
        [
            "default.qubit",
            "default.mixed",
            "lightning.qubit",
        ],
    )
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

        @qml.defer_measurements
        @qml.qnode(dev)
        def quantum_control_circuit(rads):
            qml.Hadamard(0)
            qml.PauliX(0)
            m_0 = qml.measure(0)
            qml.cond(m_0 == 0, op)(rads, wires=1)
            return qml.probs(wires=1)

        normal_probs = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(normal_probs, cond_probs)

    @pytest.mark.parametrize(
        "device",
        [
            "default.qubit",
            "default.mixed",
            "lightning.qubit",
        ],
    )
    def test_conditional_rotations_with_else(self, device):
        """Test that an else operation can also defined using qml.cond."""
        dev = qml.device(device, wires=3)
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

        @qml.defer_measurements
        @qml.qnode(dev)
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

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode1(par):
            qml.Hadamard(0)
            qml.ctrl(op, control=0)(phi=par, wires=1)
            return qml.expval(qml.PauliZ(1))

        @qml.defer_measurements
        @qml.qnode(dev)
        def qnode2(par):
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
        dev = qml.device("default.qubit", wires=3)

        @qml.defer_measurements
        @qml.qnode(dev)
        def qnode():
            m_0 = qml.measure(0)
            qml.cond(m_0 == control_val, qml.PauliX)(wires=1)
            return qml.expval(qml.PauliZ(1))

        assert qnode() == expected

    @pytest.mark.parametrize(
        "device",
        ["default.qubit", "default.mixed", "lightning.qubit"],
    )
    def test_cond_qfunc(self, device):
        """Test that a qfunc can also used with qml.cond."""
        dev = qml.device(device, wires=4)

        r = 2.324

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.Hadamard(0)

            qml.CNOT(wires=[0, 1])
            qml.CRY(rads, wires=[0, 1])
            qml.CZ(wires=[0, 1])
            qml.ctrl(qml.CRX, control=0, control_values=[1])(0.5, [1, 2])
            return qml.probs(wires=[1, 2])

        def f(x):
            qml.PauliX(1)
            qml.RY(x, wires=1)
            qml.PauliZ(1)
            qml.CRX(0.5, [1, 2])

        @qml.defer_measurements
        @qml.qnode(dev)
        def quantum_control_circuit(r):
            qml.Hadamard(0)
            m_0 = qml.measure(0)
            qml.cond(m_0, f)(r)
            return qml.probs(wires=[1, 2])

        exp = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(exp, cond_probs)

    @pytest.mark.parametrize(
        "device",
        ["default.qubit", "default.mixed", "lightning.qubit"],
    )
    def test_cond_qfunc_with_else(self, device):
        """Test that a qfunc can also used with qml.cond even when an else
        qfunc is provided."""
        dev = qml.device(device, wires=3)

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

        @qml.defer_measurements
        @qml.qnode(dev)
        def cond_qnode(x, y):
            qml.RY(x, wires=1)
            m_0 = qml.measure(1)
            qml.cond(m_0, f, g)(y)
            return qml.probs(wires=[0])

        assert np.allclose(normal_circuit(x, y), cond_qnode(x, y))

    def test_cond_on_measured_wire(self):
        """Test that applying a conditional operation on the same wire
        that is measured works as expected."""
        dev = qml.device("default.qubit", wires=2)

        @qml.defer_measurements
        @qml.qnode(dev)
        def qnode():
            qml.Hadamard(0)
            m = qml.measure(0)
            qml.cond(m, qml.PauliX)(0)
            return qml.density_matrix(0)

        # Above circuit will cause wire 0 to go back to the |0> computational
        # basis state. We can inspect the reduced density matrix to confirm this
        # without inspecting the extra wires
        expected_dmat = np.array([[1, 0], [0, 0]])
        assert np.allclose(qnode(), expected_dmat)


class TestExpressionConditionals:
    """Test Conditionals that rely on expressions of mid-circuit measurements."""

    @pytest.mark.parametrize("r", np.linspace(0.1, 2 * np.pi - 0.1, 4))
    @pytest.mark.parametrize("op", [qml.RX, qml.RY, qml.RZ])
    def test_conditional_rotations(self, r, op):
        """Test that the quantum conditional operations match the output of
        controlled rotations. And additionally that summing measurements works as expected."""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.RX(2.4, wires=0)
            qml.RY(1.3, wires=1)
            qml.ctrl(op, control=(0, 1), control_values=[True, True])(rads, wires=2)
            return qml.probs(wires=2)

        @qml.defer_measurements
        @qml.qnode(dev)
        def quantum_control_circuit(rads):
            qml.RX(2.4, wires=0)
            qml.RY(1.3, wires=1)
            m_0 = qml.measure(0)
            m_1 = qml.measure(1)
            qml.cond(m_0 + m_1 == 2, op)(rads, wires=2)
            return qml.probs(wires=2)

        normal_probs = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(normal_probs, cond_probs)

    @pytest.mark.parametrize("r", np.linspace(0.1, 2 * np.pi - 0.1, 4))
    def test_triple_measurement_condition_expression(self, r):
        """Test that combining the results of three mid-circuit measurements works as expected."""
        dev = qml.device("default.qubit", wires=7)

        @qml.defer_measurements
        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.RX(2.4, wires=0)
            qml.RY(1.3, wires=1)
            qml.RX(1.7, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 2])

            qml.ctrl(qml.RX, (0, 1, 2), [False, True, True])(rads, wires=3)

            return qml.probs(wires=3)

        @qml.defer_measurements
        @qml.qnode(dev)
        def quantum_control_circuit(rads):
            qml.RX(2.4, wires=0)
            qml.RY(1.3, wires=1)
            qml.RX(1.7, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 2])
            m_0 = qml.measure(0)
            m_1 = qml.measure(1)
            m_2 = qml.measure(2)

            expression = 4 * m_0 + 2 * m_1 + m_2
            qml.cond(expression == 3, qml.RX)(rads, wires=3)
            return qml.probs(wires=3)

        normal_probs = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(normal_probs, cond_probs)

    def test_multiple_conditions(self):
        """Test that when multiple "branches" of the mid-circuit measurements all satisfy the criteria then
        this translates to multiple control gates.
        """
        dev = qml.device("default.qubit", wires=7)

        @qml.defer_measurements
        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.RX(2.4, wires=0)
            qml.RY(1.3, wires=1)
            qml.RX(1.7, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 2])

            qml.ctrl(qml.RX, (0, 1, 2), [False, True, True])(rads, wires=3)
            qml.ctrl(qml.RX, (0, 1, 2), [True, False, False])(rads, wires=3)
            qml.ctrl(qml.RX, (0, 1, 2), [True, False, True])(rads, wires=3)
            qml.ctrl(qml.RX, (0, 1, 2), [True, True, False])(rads, wires=3)

            return qml.probs(wires=3)

        @qml.defer_measurements
        @qml.qnode(dev)
        def quantum_control_circuit(rads):
            qml.RX(2.4, wires=0)
            qml.RY(1.3, wires=1)
            qml.RX(1.7, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 2])
            m_0 = qml.measure(0)
            m_1 = qml.measure(1)
            m_2 = qml.measure(2)

            expression = 4 * m_0 + 2 * m_1 + m_2
            qml.cond((expression >= 3) & (expression <= 6), qml.RX)(rads, wires=3)
            return qml.probs(wires=3)

        normal_probs = normal_circuit(1.0)
        cond_probs = quantum_control_circuit(1.0)

        assert np.allclose(normal_probs, cond_probs)

    def test_composed_conditions(self):
        """Test that a complex nested expression gets resolved correctly to the corresponding correct control gates."""
        dev = qml.device("default.qubit", wires=7)

        @qml.defer_measurements
        @qml.qnode(dev)
        def normal_circuit(rads):
            qml.RX(2.4, wires=0)
            qml.RY(1.3, wires=1)
            qml.RX(1.7, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 2])

            qml.ctrl(qml.RX, (0, 1, 2), [False, False, False])(rads, wires=3)
            qml.ctrl(qml.RX, (0, 1, 2), [False, False, True])(rads, wires=3)
            qml.ctrl(qml.RX, (0, 1, 2), [True, False, True])(rads, wires=3)
            qml.ctrl(qml.RX, (0, 1, 2), [False, True, False])(rads, wires=3)
            qml.ctrl(qml.RX, (0, 1, 2), [False, True, True])(rads, wires=3)

            return qml.probs(wires=3)

        @qml.defer_measurements
        @qml.qnode(dev)
        def quantum_control_circuit(rads):
            qml.RX(2.4, wires=0)
            qml.RY(1.3, wires=1)
            qml.RX(1.7, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 2])
            m_0 = qml.measure(0)
            expr1 = 2 * m_0
            m_1 = qml.measure(1)
            expr2 = (3 * m_1 + 2) * (4 * expr1 + 2)
            m_2 = qml.measure(2)
            expr3 = expr2 / (m_2 + 3)
            qml.cond(expr3 <= 6, qml.RX)(rads, wires=3)
            return qml.probs(wires=3)

        normal_probs = normal_circuit(1.0)
        cond_probs = quantum_control_circuit(1.0)

        assert np.allclose(normal_probs, cond_probs)


class TestTemplates:
    """Tests templates being conditioned on mid-circuit measurement outcomes."""

    def test_angle_embedding(self):
        """Test the angle embedding template conditioned on mid-circuit
        measurement outcomes."""
        template = qml.AngleEmbedding
        feature_vector = [1, 2, 3]

        dev = qml.device("default.qubit", wires=6)

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

        res1 = qnode1()
        res2 = qnode2()

        assert np.allclose(res1, res2)

        tape1 = qml.workflow.construct_tape(qnode1)()
        tape2 = qml.workflow.construct_tape(qnode2)()
        assert len(tape2.operations) == len(tape1.operations)
        assert len(tape1.measurements) == len(tape2.measurements)

        # Check the operations
        for op1, op2 in zip(tape1.operations, tape2.operations):
            assert isinstance(op1, type(op2))
            assert np.allclose(op1.data, op2.data)

        # Check the measurements
        for op1, op2 in zip(tape1.measurements, tape2.measurements):
            assert isinstance(op1, type(op2))

    @pytest.mark.parametrize("template", [qml.StronglyEntanglingLayers, qml.BasicEntanglerLayers])
    def test_layers(self, template):
        """Test layers conditioned on mid-circuit measurement outcomes."""
        dev = qml.device("default.qubit", wires=4)

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
        tape1 = qml.workflow.construct_tape(qnode1)(weights)
        tape2 = qml.workflow.construct_tape(qnode2)(weights)
        assert len(tape2.operations) == len(tape1.operations)
        assert len(tape1.measurements) == len(tape2.measurements)

        # Check the operations
        for op1, op2 in zip(tape1.operations, tape2.operations):
            assert isinstance(op1, type(op2))
            assert np.allclose(op1.data, op2.data)

        # Check the measurements
        for op1, op2 in zip(tape1.measurements, tape2.measurements):
            assert isinstance(op1, type(op2))


class TestQubitReuseAndReset:
    """Tests for the qubit reuse/reset functionality of `qml.measure`."""

    def test_new_wire_for_multiple_measurements(self):
        """Test that a new wire is added if there are multiple mid-circuit measurements
        on the same wire."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        @qml.defer_measurements
        def circ(x, y):
            qml.RX(x, 0)
            qml.measure(0)
            qml.RY(y, 1)
            qml.measure(0)
            qml.RZ(x + y, 1)
            qml.measure(0)
            return qml.expval(qml.PauliZ(1))

        expected = [
            qml.RX(1.0, 0),
            qml.CNOT([0, 2]),
            qml.RY(2.0, 1),
            qml.CNOT([0, 3]),
            qml.RZ(3.0, 1),
        ]

        tape = qml.workflow.construct_tape(circ)(1.0, 2.0)
        assert tape.operations == expected

    def test_correct_cnot_for_reset(self):
        """Test that a CNOT is applied from the wire that stores the measurement
        to the measured wire after the measurement."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode1(x):
            qml.Hadamard(0)
            qml.CRX(x, [0, 1])
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(dev)
        @qml.defer_measurements
        def qnode2(x):
            qml.Hadamard(0)
            m0 = qml.measure(0, reset=True)
            qml.cond(m0, qml.RX)(x, 1)
            return qml.expval(qml.PauliZ(1))

        assert np.allclose(qnode1(0.123), qnode2(0.123))

        expected_circuit = [
            qml.Hadamard(0),
            qml.CNOT([0, 2]),
            qml.CNOT([2, 0]),
            qml.CRX(0.123, wires=[2, 1]),
            qml.expval(qml.PauliZ(1)),
        ]

        tape2 = qml.workflow.construct_tape(qnode2)(0.123)
        assert len(tape2.circuit) == len(expected_circuit)
        for actual, expected in zip(tape2.circuit, expected_circuit):
            qml.assert_equal(actual, expected)

    def test_measurements_add_new_qubits(self):
        """Test that qubit reuse related logic is applied if a wire with mid-circuit
        measurements is included in terminal measurements."""
        tape = qml.tape.QuantumScript(
            ops=[qml.Hadamard(0), MidMeasureMP(0)], measurements=[qml.density_matrix(wires=[0])]
        )
        expected = np.eye(2) / 2

        tapes, _ = qml.defer_measurements(tape)

        dev = qml.device("default.qubit")
        res = qml.execute(tapes, dev)

        assert np.allclose(res, expected)

        deferred_tape = tapes[0]
        assert deferred_tape.operations == [qml.Hadamard(0), qml.CNOT([0, 1])]
        assert deferred_tape.measurements == [qml.density_matrix(wires=[0])]

    def test_wire_is_reset(self):
        """Test that a wire is reset to the |0> state without any local phases
        after measurement if reset is requested."""
        dev = qml.device("default.qubit", wires=3)

        @qml.defer_measurements
        @qml.qnode(dev)
        def qnode(x):
            qml.Hadamard(0)
            qml.PhaseShift(np.pi / 4, 0)
            m = qml.measure(0, reset=True)
            qml.cond(m, qml.RX)(x, 1)
            return qml.density_matrix(wires=[0])

        # Expected reduced density matrix on wire 0
        expected_mat = np.array([[1, 0], [0, 0]])
        assert np.allclose(qnode(0.123), expected_mat)

    def test_multiple_measurements_mixed_reset(self, mocker):
        """Test that a QNode with multiple mid-circuit measurements with
        different resets is transformed correctly."""
        dev = qml.device("default.qubit", wires=6)

        @qml.qnode(dev)
        def qnode(p, x, y):
            qml.Hadamard(0)
            qml.PhaseShift(p, 0)
            # Set measurement_ids so that the order of wires in combined
            # measurement values is consistent

            mp0 = qml.measurements.MidMeasureMP(0, reset=True, id=0)
            m0 = qml.measurements.MeasurementValue([mp0], lambda v: v)
            qml.cond(~m0, qml.RX)(x, 1)
            mp1 = qml.measurements.MidMeasureMP(1, reset=True, id=1)
            m1 = qml.measurements.MeasurementValue([mp1], lambda v: v)
            qml.cond(m0 & m1, qml.Hadamard)(0)
            mp2 = qml.measurements.MidMeasureMP(0, id=2)
            m2 = qml.measurements.MeasurementValue([mp2], lambda v: v)
            qml.cond(m1 | m2, qml.RY)(y, 2)
            return qml.expval(qml.PauliZ(2))

        spy = mocker.spy(qml.defer_measurements, "_transform")
        _ = qnode(0.123, 0.456, 0.789)
        assert spy.call_count == 1

        expected_circuit = [
            qml.Hadamard(0),
            qml.PhaseShift(0.123, 0),
            qml.CNOT([0, 3]),
            qml.CNOT([3, 0]),
            Controlled(qml.RX(0.456, 1), 3, [False]),
            qml.CNOT([1, 4]),
            qml.CNOT([4, 1]),
            Controlled(qml.Hadamard(0), [3, 4]),
            qml.CNOT([0, 5]),
            Controlled(qml.RY(0.789, 2), [4, 5], [False, True]),
            Controlled(qml.RY(0.789, 2), [4, 5], [True, False]),
            Controlled(qml.RY(0.789, 2), [4, 5], [True, True]),
            qml.expval(qml.PauliZ(2)),
        ]

        tape = qml.workflow.construct_tape(qnode)(0.123, 0.456, 0.789)
        deferred_tapes, _ = qml.defer_measurements(tape)
        deferred_tape = deferred_tapes[0]
        assert len(deferred_tape.circuit) == len(expected_circuit)
        for actual, expected in zip(deferred_tape.circuit, expected_circuit):
            qml.assert_equal(actual, expected)


class TestDrawing:
    """Tests drawing circuits with mid-circuit measurements and conditional
    operations that have been transformed"""

    def test_drawing_no_reuse(self):
        """Test that drawing a func with mid-circuit measurements works and
        that controlled operations are drawn for conditional operations when
        the measured wires are not reused."""

        # TODO: Update after drawing for mid-circuit measurements is updated.

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

    def test_drawing_with_reuse(self):
        """Test that drawing a func with mid-circuit measurements works and
        that controlled operations are drawn for conditional operations when
        the measured wires are reused."""

        # TODO: Update after drawing for mid-circuit measurements is updated.

        def qfunc():
            m_0 = qml.measure(0, reset=True)
            qml.cond(m_0, qml.RY)(0.312, wires=1)

            m_2 = qml.measure(2)
            qml.cond(m_2, qml.RY)(0.312, wires=1)
            return qml.expval(qml.PauliZ(1))

        dev = qml.device("default.qubit", wires=4)

        transformed_qfunc = qml.transforms.defer_measurements(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        expected = (
            "0: ─╭●─╭X─────────────────────┤     \n"
            "1: ─│──│──╭RY(0.31)─╭RY(0.31)─┤  <Z>\n"
            "2: ─│──│──│─────────╰●────────┤     \n"
            "3: ─╰X─╰●─╰●──────────────────┤     "
        )
        assert qml.draw(transformed_qnode)() == expected

    @pytest.mark.parametrize(
        "mp, label",
        [
            (qml.sample, "Sample"),
            (qml.probs, "Probs"),
            (qml.var, "Var[None]"),
            (qml.counts, "Counts"),
            (qml.expval, "<None>"),
        ],
    )
    def test_drawing_with_mcm_terminal_measure(self, mp, label):
        """Test that drawing a func works correctly when collecting statistics on
        mid-circuit measurements."""

        def qfunc():
            m_0 = qml.measure(0, reset=True)
            qml.cond(m_0, qml.RY)(0.312, wires=1)

            return mp(op=m_0), qml.expval(qml.Z(1))

        dev = qml.device("default.qubit", wires=4)

        transformed_qfunc = qml.transforms.defer_measurements(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)

        spaces = " " * len(label)
        expval = "<Z>".ljust(len(label))
        expected = (
            f"0: ─╭●─╭X───────────┤  {spaces}\n"
            f"1: ─│──│──╭RY(0.31)─┤  {expval}\n"
            f"2: ─╰X─╰●─╰●────────┤  {label}"
        )
        assert qml.draw(transformed_qnode)() == expected

    @pytest.mark.parametrize("mp", [qml.sample, qml.probs, qml.var, qml.counts, qml.expval])
    def test_draw_mpl_with_mcm_terminal_measure(self, mp):
        """Test that no error is raised when drawing a circuit which collects
        statistics on mid-circuit measurements"""

        def qfunc():
            m_0 = qml.measure(0, reset=True)
            qml.cond(m_0, qml.RY)(0.312, wires=1)

            return mp(op=m_0), qml.expval(qml.Z(1))

        dev = qml.device("default.qubit", wires=4)

        transformed_qfunc = qml.transforms.defer_measurements(qfunc)
        transformed_qnode = qml.QNode(transformed_qfunc, dev)
        _ = qml.draw_mpl(transformed_qnode)()


def test_custom_wire_labels_allowed_without_reuse():
    """Test that custom wire labels work if no qubits are re-used."""
    with qml.queuing.AnnotatedQueue() as q:
        qml.Hadamard("a")
        ma = qml.measure("a", reset=False)
        qml.cond(ma, qml.PauliX)("b")
        qml.probs(wires="b")

    tape = qml.tape.QuantumScript.from_queue(q)
    tapes, _ = qml.defer_measurements(tape)
    tape = tapes[0]

    assert len(tape) == 3
    qml.assert_equal(tape[0], qml.Hadamard("a"))
    qml.assert_equal(tape[1], qml.CNOT(["a", "b"]))
    qml.assert_equal(tape[2], qml.probs(wires="b"))


def test_integer_wire_labels_with_reset():
    """Tests that integer wire labels work when qubits are re-used."""

    # Reset example
    with qml.queuing.AnnotatedQueue() as q:
        qml.Hadamard(0)
        ma = qml.measure(0, reset=True)
        qml.cond(ma, qml.PauliX)(1)
        qml.probs(wires=0)

    tape = qml.tape.QuantumScript.from_queue(q)
    tapes, _ = qml.defer_measurements(tape)
    tape = tapes[0]

    assert len(tape) == 5
    qml.assert_equal(tape[0], qml.Hadamard(0))
    qml.assert_equal(tape[1], qml.CNOT([0, 2]))
    qml.assert_equal(tape[2], qml.CNOT([2, 0]))
    qml.assert_equal(tape[3], qml.CNOT([2, 1]))
    qml.assert_equal(tape[4], qml.probs(wires=0))

    # Reuse example
    with qml.queuing.AnnotatedQueue() as q:
        qml.Hadamard(0)
        ma = qml.measure(0)
        qml.cond(ma, qml.PauliX)(1)
        qml.Hadamard(0)
        qml.probs(wires=1)

    tape = qml.tape.QuantumScript.from_queue(q)
    tapes, _ = qml.defer_measurements(tape)
    tape = tapes[0]

    assert len(tape) == 5
    qml.assert_equal(tape[0], qml.Hadamard(0))
    qml.assert_equal(tape[1], qml.CNOT([0, 2]))
    qml.assert_equal(tape[2], qml.CNOT([2, 1]))
    qml.assert_equal(tape[3], qml.Hadamard(0))
    qml.assert_equal(tape[4], qml.probs(wires=1))


def test_custom_wire_labels_with_reset():
    """Test that custom wire labels work if any qubits are re-used."""

    # Reset example (should be the same circuit as in previous test but with wire labels)
    with qml.queuing.AnnotatedQueue() as q:
        qml.Hadamard("a")
        ma = qml.measure("a", reset=True)
        qml.cond(ma, qml.PauliX)("b")
        qml.probs(wires="a")

    tape = qml.tape.QuantumScript.from_queue(q)
    tapes, _ = qml.defer_measurements(tape)
    tape = tapes[0]

    assert len(tape) == 5
    qml.assert_equal(tape[0], qml.Hadamard("a"))
    qml.assert_equal(tape[1], qml.CNOT(["a", 0]))
    qml.assert_equal(tape[2], qml.CNOT([0, "a"]))
    qml.assert_equal(tape[3], qml.CNOT([0, "b"]))
    qml.assert_equal(tape[4], qml.probs(wires="a"))

    # Reuse example (should be the same circuit as in previous test but with wire labels)
    with qml.queuing.AnnotatedQueue() as q:
        qml.Hadamard("a")
        ma = qml.measure("a")
        qml.cond(ma, qml.PauliX)("b")
        qml.Hadamard("a")
        qml.probs(wires="b")

    tape = qml.tape.QuantumScript.from_queue(q)
    tapes, _ = qml.defer_measurements(tape)
    tape = tapes[0]

    assert len(tape) == 5
    qml.assert_equal(tape[0], qml.Hadamard("a"))
    qml.assert_equal(tape[1], qml.CNOT(["a", 0]))
    qml.assert_equal(tape[2], qml.CNOT([0, "b"]))
    qml.assert_equal(tape[3], qml.Hadamard("a"))
    qml.assert_equal(tape[4], qml.probs(wires="b"))
