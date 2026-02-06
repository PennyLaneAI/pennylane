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

import pennylane as qp
import pennylane.numpy as np
from pennylane.devices import DefaultQubit
from pennylane.exceptions import DeviceError
from pennylane.ops import Controlled, MeasurementValue, MidMeasure


def test_broadcasted_postselection(mocker):
    """Test that broadcast_expand is used iff broadcasting with postselection."""
    spy = mocker.spy(qp.transforms, "broadcast_expand")

    # Broadcasting with postselection
    tape1 = qp.tape.QuantumScript(
        [qp.RX([0.1, 0.2], 0), MidMeasure(0, postselect=1), qp.CNOT([0, 1])],
        [qp.probs(wires=[0])],
    )
    _, _ = qp.defer_measurements(tape1)

    assert spy.call_count == 1

    # Broadcasting without postselection
    tape2 = qp.tape.QuantumScript(
        [qp.RX([0.1, 0.2], 0), MidMeasure(0), qp.CNOT([0, 1])],
        [qp.probs(wires=[0])],
    )
    _, _ = qp.defer_measurements(tape2)

    assert spy.call_count == 1

    # Postselection without broadcasting
    tape3 = qp.tape.QuantumScript(
        [qp.RX(0.1, 0), MidMeasure(0, postselect=1), qp.CNOT([0, 1])],
        [qp.probs(wires=[0])],
    )
    _, _ = qp.defer_measurements(tape3)

    assert spy.call_count == 1

    # No postselection, no broadcasting
    tape4 = qp.tape.QuantumScript(
        [qp.RX(0.1, 0), MidMeasure(0), qp.CNOT([0, 1])],
        [qp.probs(wires=[0])],
    )
    _, _ = qp.defer_measurements(tape4)

    assert spy.call_count == 1


def test_broadcasted_postselection_with_sample_error():
    """Test that an error is raised if returning qp.sample if postselecting with broadcasting"""
    tape = qp.tape.QuantumScript(
        [qp.RX([0.1, 0.2], 0), MidMeasure(0, postselect=1)], [qp.sample(wires=0)], shots=10
    )
    dev = qp.device("default.qubit")

    with pytest.raises(ValueError, match="Returning qp.sample is not supported when"):
        qp.defer_measurements(tape)

    @qp.defer_measurements
    @qp.set_shots(10)
    @qp.qnode(dev)
    def circuit():
        qp.RX([0.1, 0.2], 0)
        qp.measure(0, postselect=1)
        return qp.sample(wires=0)

    with pytest.raises(ValueError, match="Returning qp.sample is not supported when"):
        _ = circuit()


def test_allow_postselect():
    """Tests that allow_postselect=False forbids postselection on mid-circuit measurements."""

    circuit = qp.tape.QuantumScript([MidMeasure(wires=0, postselect=0)], [qp.expval(qp.Z(0))])
    with pytest.raises(ValueError, match="Postselection is not allowed"):
        _, __ = qp.defer_measurements(circuit, allow_postselect=False)


def test_postselection_error_with_wrong_device():
    """Test that an error is raised when postselection is used with a device
    other than `default.qubit`."""
    dev = qp.device("default.mixed", wires=2)

    @qp.defer_measurements
    @qp.qnode(dev)
    def circ():
        qp.measure(0, postselect=1)
        return qp.probs(wires=[0])

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
    dev = qp.device("default.qubit")
    spy = mocker.spy(qp.defer_measurements, "_tape_transform")

    @qp.set_shots(shots)
    @qp.qnode(dev, postselect_mode=postselect_mode, mcm_method="deferred")
    def f(x):
        qp.RX(x, 0)
        _ = qp.measure(0, postselect=postselect_value)
        return qp.sample(wires=[0])

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
        (qp.state(), "Cannot use StateMP as a measurement when"),
        (qp.probs(), "Cannot use ProbabilityMP as a measurement without"),
        (qp.sample(), "Cannot use SampleMP as a measurement without"),
        (qp.counts(), "Cannot use CountsMP as a measurement without"),
    ],
)
def test_unsupported_measurements(mp, err_msg):
    """Test that using unsupported measurements raises an error."""
    tape = qp.tape.QuantumScript([MidMeasure(0)], [mp])

    with pytest.raises(ValueError, match=err_msg):
        _, _ = qp.defer_measurements(tape)


@pytest.mark.parametrize(
    "mp, compose_mv",
    [
        (qp.expval, True),
        (qp.var, True),
        (qp.probs, False),
        (qp.sample, True),
        (qp.sample, False),
        (qp.counts, True),
        (qp.counts, False),
    ],
)
def test_multi_mcm_stats_same_wire(mp, compose_mv):
    """Test that a tape collecting statistics on multiple mid-circuit measurements when
    they measure the same wire is transformed correctly."""
    mp1 = MidMeasure(0, id="foo")
    mp2 = MidMeasure(0, id="bar")
    mv1 = MeasurementValue([mp1], None)
    mv2 = MeasurementValue([mp2], None)

    mv = mv1 * mv2 if compose_mv else [mv1, mv2]
    tape = qp.tape.QuantumScript([qp.PauliX(0), mp1, mp2], [mp(op=mv)], shots=10)
    [deferred_tape], _ = qp.defer_measurements(tape)

    emp1 = MidMeasure(1, id="foo")
    emp2 = MidMeasure(2, id="bar")
    emv1 = MeasurementValue([emp1], None)
    emv2 = MeasurementValue([emp2], None)
    emv = emv1 * emv2 if compose_mv else [emv1, emv2]

    assert deferred_tape.operations == [qp.PauliX(0), qp.CNOT([0, 1]), qp.CNOT([0, 2])]
    assert deferred_tape.measurements == [mp(op=emv)]


class TestQNode:
    """Test that the transform integrates well with QNodes."""

    def test_only_mcm(self):
        """Test that a quantum function that only contains one mid-circuit
        measurement yields the correct results and is transformed correctly."""
        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def qnode1():
            return qp.expval(qp.PauliZ(0))

        @qp.qnode(dev)
        @qp.defer_measurements
        def qnode2():
            qp.measure(1)
            return qp.expval(qp.PauliZ(0))

        res1 = qnode1()
        res2 = qnode2()
        assert res1 == res2
        assert isinstance(res1, type(res2))
        assert res1.shape == res2.shape

        tape1 = qp.workflow.construct_tape(qnode1)()
        tape2 = qp.workflow.construct_tape(qnode2)()
        assert len(tape2.operations) == 0
        assert len(tape1.measurements) == len(tape2.measurements)

        # Check the measurements
        for op1, op2 in zip(tape1.measurements, tape2.measurements):
            assert isinstance(op1, type(op2))

    def test_reuse_wire_after_measurement(self):
        """Test that wires can be reused after measurement."""
        dev = qp.device("default.qubit", wires=2)

        @qp.defer_measurements
        @qp.qnode(dev)
        def qnode():
            qp.Hadamard(0)
            qp.measure(0)
            qp.PauliZ(0)
            return qp.expval(qp.PauliX(0))

        _ = qnode()

    def test_no_new_wires_without_reuse(self, mocker):
        """Test that new wires are not added if a measured wire is not reused."""
        dev = qp.device("default.qubit", wires=3)

        # Quantum teleportation
        @qp.qnode(dev)
        def qnode1(phi):
            qp.RX(phi, 0)
            qp.Hadamard(1)
            qp.CNOT([1, 2])
            qp.CNOT([0, 1])
            qp.Hadamard(0)

            m0 = qp.measure(0)
            qp.cond(m0, qp.PauliZ)(2)
            m1 = qp.measure(1)
            qp.cond(m1, qp.PauliX)(2)
            return qp.expval(qp.PauliZ(2))

        # Prepare wire 0 in arbitrary state
        @qp.qnode(dev)
        def qnode2(phi):
            qp.RX(phi, 0)
            return qp.expval(qp.PauliZ(0))

        spy = mocker.spy(qp.defer_measurements, "_tape_transform")

        # Outputs should match
        assert np.isclose(qnode1(np.pi / 4), qnode2(np.pi / 4))
        assert spy.call_count == 2  # once per device preprocessing

        tape1 = qp.workflow.construct_tape(qnode1)(np.pi / 4)
        deferred_tapes, _ = qp.defer_measurements(tape1)
        deferred_tape = deferred_tapes[0]
        assert isinstance(deferred_tape.operations[5], Controlled)
        qp.assert_equal(deferred_tape.operations[5].base, qp.PauliZ(2))
        assert deferred_tape.operations[5].hyperparameters["control_wires"] == qp.wires.Wires(0)

        qp.assert_equal(deferred_tape.operations[6], qp.CNOT([1, 2]))

    def test_new_wires_after_reuse(self, mocker):
        """Test that a new wire is added for every measurement after which
        the wire is reused."""
        dev = qp.device("default.qubit", wires=4)
        spy = mocker.spy(qp.defer_measurements, "_tape_transform")

        @qp.defer_measurements
        @qp.qnode(dev)
        def qnode1(phi, theta):
            qp.RX(phi, 0)
            m0 = qp.measure(0, reset=True)  # Reused measurement, one new wire added
            qp.cond(m0, qp.Hadamard)(1)
            m1 = qp.measure(1)  # No reuse
            qp.RY(theta, 2)
            qp.cond(m1, qp.RY)(-theta, 2)
            return qp.expval(qp.PauliZ(2))

        res1 = qnode1(np.pi / 4, 3 * np.pi / 4)

        @qp.defer_measurements
        @qp.qnode(dev)
        def qnode2(phi, theta):
            qp.RX(phi, 0)
            m0 = qp.measure(0)  # No reuse
            qp.cond(m0, qp.Hadamard)(1)
            m1 = qp.measure(1)  # No reuse
            qp.RY(theta, 2)
            qp.cond(m1, qp.RY)(-theta, 2)
            return qp.expval(qp.PauliZ(2))

        res2 = qnode2(np.pi / 4, 3 * np.pi / 4)

        assert spy.call_count == 4

        tape1 = qp.workflow.construct_tape(qnode1)(np.pi / 4, 3 * np.pi / 4)
        deferred_tapes1, _ = qp.defer_measurements(tape1)
        deferred_tape1 = deferred_tapes1[0]
        assert len(deferred_tape1.wires) == 4
        assert len(deferred_tape1.operations) == 6

        assert np.allclose(res1, res2)

        tape2 = qp.workflow.construct_tape(qnode2)(np.pi / 4, 3 * np.pi / 4)
        deferred_tapes2, _ = qp.defer_measurements(tape2)
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

        dm_transform = qp.defer_measurements
        if reduce_postselected is not None:
            dm_transform = partial(dm_transform, reduce_postselected=reduce_postselected)
        else:
            # overwrite None with the expected default value True to determine expected outputs
            reduce_postselected = True

        @qp.set_shots(shots=shots)
        @qp.qnode(dev)
        @dm_transform
        def circ1(phi):
            qp.RX(phi, wires=0)
            # Postselecting on |1> on wire 0 means that the probability of measuring
            # |1> on wire 0 is 1
            m = qp.measure(0, postselect=1)
            qp.cond(m, qp.PauliX)(wires=1)
            # Probability of measuring |1> on wire 1 should be 1
            return qp.probs(wires=1)

        assert np.allclose(circ1(phi), [0, 1])

        expected_circuit = [
            qp.RX(phi, 0),
            qp.Projector([1], wires=0),
            qp.X(1) if reduce_postselected else qp.CNOT([0, 1]),
            qp.probs(wires=1),
        ]

        tape1 = qp.workflow.construct_tape(circ1)(phi)
        assert len(tape1) == len(expected_circuit)
        for op, expected_op in zip(tape1, expected_circuit):
            qp.assert_equal(op, expected_op)

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

        dm_transform = qp.defer_measurements
        if reduce_postselected is not None:
            dm_transform = partial(dm_transform, reduce_postselected=reduce_postselected)
        else:
            # overwrite None with the expected default value True to determine expected outputs
            reduce_postselected = True

        @qp.set_shots(shots=shots)
        @qp.qnode(dev)
        @dm_transform
        def circ1(phi):
            qp.RX(phi, wires=0)
            qp.RX(phi, wires=2)
            # Postselecting on |1> on wire 0 means that the probability of measuring
            # |1> on wire 0 is 1
            m0 = qp.measure(0, postselect=1)
            m1 = qp.measure(2)
            qp.cond(m0 & m1, qp.PauliX)(wires=1)
            # Probability of measuring |1> on wire 1 should be 1
            return qp.probs(wires=1)

        atol = tol if shots is None else tol_stochastic
        expected_out = [np.cos(phi / 2) ** 2, np.sin(phi / 2) ** 2]
        assert np.allclose(circ1(phi), expected_out, atol=atol, rtol=0)

        expected_circuit = [
            qp.RX(phi, 0),
            qp.RX(phi, 2),
            qp.Projector([1], wires=0),
            qp.CNOT([2, 1]) if reduce_postselected else qp.Toffoli([0, 2, 1]),
            qp.probs(wires=1),
        ]

        tape1 = qp.workflow.construct_tape(circ1)(phi)
        assert len(tape1) == len(expected_circuit)
        for op, expected_op in zip(tape1, expected_circuit):
            qp.assert_equal(op, expected_op)

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
        # wire ordering for qp.cond)
        mp0 = MidMeasure(wires=0, postselect=0, id=0)
        mv0 = MeasurementValue([mp0], lambda v: v)
        mp1 = MidMeasure(wires=1, postselect=0, id=1)
        mv1 = MeasurementValue([mp1], lambda v: v)
        mp2 = MidMeasure(wires=2, reset=True, postselect=1, id=2)
        mv2 = MeasurementValue([mp2], lambda v: v)

        dm_transform = qp.defer_measurements
        if reduce_postselected is not None:
            dm_transform = partial(dm_transform, reduce_postselected=reduce_postselected)
        else:
            # Override None with the expected default value True to determine expected outputs
            reduce_postselected = True

        @qp.set_shots(shots=shots)
        @qp.qnode(dev)
        @dm_transform
        def circ1(phi, theta):
            qp.RX(phi, 0)
            qp.apply(mp0)
            qp.CNOT([0, 1])
            qp.apply(mp1)
            qp.cond(~(mv0 & mv1), qp.RY)(theta, wires=2)
            qp.apply(mp2)
            qp.cond(mv2, qp.PauliX)(1)
            return qp.probs(wires=[0, 1, 2])

        @qp.qnode(dev)
        def circ2():
            # To add wire 0 to tape
            qp.Identity(0)
            qp.PauliX(1)
            qp.Identity(2)
            return qp.probs(wires=[0, 1, 2])

        atol = tol if shots is None else tol_stochastic
        assert np.allclose(circ1(phi, theta), circ2(), atol=atol, rtol=0)

        expected_first_cond_block = (
            [qp.RY(theta, wires=[2])]
            if reduce_postselected
            else [
                Controlled(qp.RY(theta, wires=[2]), control_wires=[3, 4], control_values=cv)
                for cv in ([False, False], [False, True], [True, False])
            ]
        )
        expected_circuit = (
            [
                qp.RX(phi, wires=0),
                qp.Projector([0], wires=0),
                qp.CNOT([0, 3]),
                qp.CNOT([0, 1]),
                qp.Projector([0], wires=1),
                qp.CNOT([1, 4]),
            ]
            + expected_first_cond_block
            + [
                qp.Projector([1], wires=2),
                qp.CNOT([2, 5]),
                qp.PauliX(2),
                qp.PauliX(1) if reduce_postselected else qp.CNOT([5, 1]),
                qp.probs(wires=[0, 1, 2]),
            ]
        )

        tape1 = qp.workflow.construct_tape(circ1)(phi, theta)
        assert len(tape1) == len(expected_circuit)
        for op, expected_op in zip(tape1, expected_circuit):
            qp.assert_equal(op, expected_op)

    @pytest.mark.parametrize("shots", [None, 1000, [1000, 1000]])
    def test_measurement_statistics_single_wire(self, shots, seed):
        """Test that users can collect measurement statistics on
        a single mid-circuit measurement."""
        dev = DefaultQubit(seed=seed)
        dev = shots_to_analytic(dev)

        @qp.set_shots(shots=shots)
        @qp.defer_measurements
        @qp.qnode(dev)
        def circ1(x):
            qp.RX(x, 0)
            m0 = qp.measure(0)
            return qp.probs(op=m0)

        dev = DefaultQubit(seed=seed)
        dev = shots_to_analytic(dev)

        @qp.set_shots(shots=shots)
        @qp.qnode(dev)
        def circ2(x):
            qp.RX(x, 0)
            return qp.probs(wires=[0])

        param = 1.5
        assert np.allclose(circ1(param), circ2(param))

    @pytest.mark.parametrize("shots", [None, 2000, [2000, 2000]])
    def test_measured_value_wires_mapped(self, shots, tol, tol_stochastic):
        """Test that collecting statistics on a measurement value works correctly
        when the measured wire is reused."""
        dev = DefaultQubit()
        dev = shots_to_analytic(dev)

        @qp.set_shots(shots=shots)
        @qp.qnode(dev)
        @qp.defer_measurements
        def circ1(x):
            qp.RX(x, 0)
            m0 = qp.measure(0)
            qp.PauliX(0)
            return qp.probs(op=m0)

        dev = DefaultQubit()
        dev = shots_to_analytic(dev)

        @qp.set_shots(shots=shots)
        @qp.qnode(dev)
        def circ2(x):
            qp.RX(x, 0)
            return qp.probs(wires=[0])

        param = 1.5
        atol = tol if shots is None else tol_stochastic
        assert np.allclose(circ1(param), circ2(param), atol=atol, rtol=0)

        expected_ops = [qp.RX(param, 0), qp.CNOT([0, 1]), qp.PauliX(0)]
        tape1 = qp.workflow.construct_tape(circ1)(param)
        assert tape1.operations == expected_ops

        assert len(tape1.measurements) == 1
        mp = tape1.measurements[0]
        assert isinstance(mp, qp.measurements.ProbabilityMP)
        assert mp.mv is not None
        assert mp.mv.wires == qp.wires.Wires([1])

    @pytest.mark.parametrize("shots", [None, 1000, [1000, 1000]])
    def test_terminal_measurements(self, shots, seed):
        """Test that mid-circuit measurement statistics and terminal measurements
        can be made together."""
        # Using DefaultQubit to allow non-commuting measurements
        dev = DefaultQubit(seed=seed)
        dev = shots_to_analytic(dev)

        @qp.set_shots(shots=shots)
        @qp.defer_measurements
        @qp.qnode(dev)
        def circ1(x, y):
            qp.RX(x, 0)
            m0 = qp.measure(0)
            qp.RY(y, 1)
            return qp.expval(qp.PauliX(1)), qp.probs(op=m0)

        dev = DefaultQubit(seed=seed)
        dev = shots_to_analytic(dev)

        @qp.set_shots(shots=shots)
        @qp.qnode(dev)
        def circ2(x, y):
            qp.RX(x, 0)
            qp.RY(y, 1)
            return qp.expval(qp.PauliX(1)), qp.probs(wires=[0])

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
        dev = qp.device("default.qubit", wires=3)
        dev = shots_to_analytic(dev)

        def func1():
            qp.RY(0.123, wires=0)
            qp.PauliX(0)
            return qp.expval(qp.PauliZ(0))

        def func2():
            qp.RY(0.123, wires=0)
            qp.measure(1)
            qp.PauliX(0)
            return qp.expval(qp.PauliZ(0))

        tape_deferred_func = qp.defer_measurements(func2)
        qnode1 = qp.QNode(func1, dev)
        qnode2 = qp.QNode(tape_deferred_func, dev)

        res1 = qnode1()
        res2 = qnode2()
        assert res1 == res2
        assert isinstance(res1, type(res2))
        assert res1.shape == res2.shape

        tape1 = qp.workflow.construct_tape(qnode1)()
        tape2 = qp.workflow.construct_tape(qnode2)()
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

        with qp.queuing.AnnotatedQueue() as q:
            qp.measure(mid_measure_wire)
            qp.expval(qp.prod(*[qp.PauliZ(w) for w in tp_wires]))

        tape = qp.tape.QuantumScript.from_queue(q)
        tape, _ = qp.defer_measurements(tape)
        tape = tape[0]
        # Check the operations and measurements in the tape
        assert len(tape.measurements) == 1

        measurement = tape.measurements[0]
        assert isinstance(measurement, qp.measurements.MeasurementProcess)

        tensor = measurement.obs
        assert len(tensor.operands) == 3

        for idx, ob in enumerate(tensor.operands):
            assert isinstance(ob, qp.PauliZ)
            assert ob.wires == qp.wires.Wires(tp_wires[idx])

    def test_cv_op_error(self):
        """Test that CV operations are not supported."""
        dev = qp.device("default.gaussian", wires=3)

        @qp.defer_measurements
        @qp.qnode(dev)
        def qnode():
            qp.measure(0)
            qp.Rotation(0.123, wires=[0])
            return qp.expval(qp.NumberOperator(1))

        with pytest.raises(
            ValueError, match="Continuous variable operations and observables are not supported"
        ):
            qnode()

    def test_cv_obs_error(self):
        """Test that CV observables are not supported."""
        dev = qp.device("default.gaussian", wires=3)

        @qp.defer_measurements
        @qp.qnode(dev)
        def qnode():
            qp.measure(0)
            return qp.expval(qp.NumberOperator(1))

        with pytest.raises(
            ValueError, match="Continuous variable operations and observables are not supported"
        ):
            qnode()


class TestConditionalOperations:
    """Tests conditional operations"""

    @pytest.mark.parametrize(
        "terminal_measurement",
        [
            qp.expval(qp.PauliZ(1)),
            qp.var(qp.PauliZ(2) @ qp.PauliZ(0)),
            qp.probs(wires=[1, 0]),
        ],
    )
    def test_correct_ops_in_tape(self, terminal_measurement):
        """Test that the underlying tape contains the correct operations."""
        first_par = 0.1
        sec_par = 0.3

        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(4, reset=True)
            qp.cond(m_0, qp.RY)(first_par, wires=1)

            m_1 = qp.measure(3)
            qp.cond(m_1, qp.RZ)(sec_par, wires=1)
            qp.apply(terminal_measurement)

        tape = qp.tape.QuantumScript.from_queue(q)

        tapes, _ = qp.defer_measurements(tape)
        tape = tapes[0]

        assert len(tape.operations) == 4
        assert len(tape.measurements) == 1

        # Check the two underlying Controlled instances
        first_ctrl_op = tape.operations[2]
        assert isinstance(first_ctrl_op, Controlled)
        qp.assert_equal(first_ctrl_op.base, qp.RY(first_par, 1))

        sec_ctrl_op = tape.operations[3]
        assert isinstance(sec_ctrl_op, Controlled)
        qp.assert_equal(sec_ctrl_op.base, qp.RZ(sec_par, 1))

        assert tape.measurements[0] == terminal_measurement

    def test_correct_ops_in_tape_inversion(self):
        """Test that the underlying tape contains the correct operations if a
        measurement value was inverted."""
        first_par = 0.1
        terminal_measurement = qp.expval(qp.PauliZ(1))

        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0)
            qp.cond(~m_0, qp.RY)(first_par, wires=1)
            qp.apply(terminal_measurement)

        tape = qp.tape.QuantumScript.from_queue(q)
        tapes, _ = qp.defer_measurements(tape)
        tape = tapes[0]
        # Conditioned on 0 as the control value, PauliX is applied before and after
        assert len(tape.operations) == 1
        assert len(tape.measurements) == 1

        # Check the two underlying Controlled instance
        ctrl_op = tape.operations[0]
        assert isinstance(ctrl_op, Controlled)
        qp.assert_equal(ctrl_op.base, qp.RY(first_par, 1))

        assert ctrl_op.wires == qp.wires.Wires([0, 1])

    def test_correct_ops_in_tape_assert_zero_state(self):
        """Test that the underlying tape contains the correct operations if a
        conditional operation was applied in the zero state case.

        Note: this case is the same as inverting right after obtaining a
        measurement value."""
        first_par = 0.1
        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0)
            qp.cond(m_0 == 0, qp.RY)(first_par, wires=1)
            qp.expval(qp.PauliZ(1))

        tape = qp.tape.QuantumScript.from_queue(q)
        tapes, _ = qp.defer_measurements(tape)
        tape = tapes[0]
        # Conditioned on 0 as the control value, PauliX is applied before and after
        assert len(tape.operations) == 1
        assert len(tape.measurements) == 1

        # Check the underlying Controlled instance
        ctrl_op = tape.operations[0]
        assert isinstance(ctrl_op, Controlled)
        qp.assert_equal(ctrl_op.base, qp.RY(first_par, 1))

    @pytest.mark.parametrize("rads", np.linspace(0.0, np.pi, 3))
    def test_quantum_teleportation(self, rads):
        """Test quantum teleportation."""

        terminal_measurement = qp.probs(wires=2)

        with qp.queuing.AnnotatedQueue() as q:
            # Create Alice's secret qubit state
            qp.RY(rads, wires=0)

            # create an EPR pair with wires 1 and 2. 1 is held by Alice and 2 held by Bob
            qp.Hadamard(wires=1)
            qp.CNOT(wires=[1, 2])

            # Alice sends her qubits through a CNOT gate.
            qp.CNOT(wires=[0, 1])

            # Alice then sends the first qubit through a Hadamard gate.
            qp.Hadamard(wires=0)

            # Alice measures her qubits, obtaining one of four results, and sends this information to Bob.
            m_0 = qp.measure(0)
            m_1 = qp.measure(1, reset=True)

            # Given Alice's measurements, Bob performs one of four operations on his half of the EPR pair and
            # recovers the original quantum state.
            qp.cond(m_1, qp.RX)(math.pi, wires=2)
            qp.cond(m_0, qp.RZ)(math.pi, wires=2)

            qp.apply(terminal_measurement)

        tape = qp.tape.QuantumScript.from_queue(q)

        tapes, _ = qp.defer_measurements(tape)
        tape = tapes[0]

        assert (
            len(tape.operations) == 5 + 1 + 1 + 2
        )  # 5 regular ops + 1 measurement op + 1 reset op + 2 conditional ops

        assert len(tape.measurements) == 1

        # Check the each operation
        op1 = tape.operations[0]
        assert isinstance(op1, qp.RY)
        assert op1.wires == qp.wires.Wires(0)
        assert op1.data == (rads,)

        op2 = tape.operations[1]
        assert isinstance(op2, qp.Hadamard)
        assert op2.wires == qp.wires.Wires(1)

        op3 = tape.operations[2]
        assert isinstance(op3, qp.CNOT)
        assert op3.wires == qp.wires.Wires([1, 2])

        op4 = tape.operations[3]
        assert isinstance(op4, qp.CNOT)
        assert op4.wires == qp.wires.Wires([0, 1])

        op5 = tape.operations[4]
        assert isinstance(op5, qp.Hadamard)
        assert op5.wires == qp.wires.Wires([0])

        # Check the two underlying CNOTs for storing measurement state
        meas_op1 = tape.operations[5]
        assert isinstance(meas_op1, qp.CNOT)
        assert meas_op1.wires == qp.wires.Wires([1, 3])

        meas_op2 = tape.operations[6]
        assert isinstance(meas_op2, qp.CNOT)
        assert meas_op2.wires == qp.wires.Wires([3, 1])

        # Check the two underlying Controlled instances
        ctrl_op1 = tape.operations[7]
        assert isinstance(ctrl_op1, Controlled)
        qp.assert_equal(ctrl_op1.base, qp.RX(math.pi, 2))
        assert ctrl_op1.wires == qp.wires.Wires([3, 2])

        ctrl_op2 = tape.operations[8]
        assert isinstance(ctrl_op2, Controlled)
        qp.assert_equal(ctrl_op2.base, qp.RZ(math.pi, 2))
        assert ctrl_op2.wires == qp.wires.Wires([0, 2])

        # Check the measurement
        qp.assert_equal(tape.measurements[0], terminal_measurement)

    @pytest.mark.parametrize("r", np.linspace(0.1, 2 * np.pi - 0.1, 4))
    @pytest.mark.parametrize(
        "device",
        [
            "default.qubit",
            "default.mixed",
            "lightning.qubit",
        ],
    )
    @pytest.mark.parametrize("ops", [(qp.RX, qp.CRX), (qp.RY, qp.CRY), (qp.RZ, qp.CRZ)])
    def test_conditional_rotations(self, device, r, ops):
        """Test that the quantum conditional operations match the output of
        controlled rotations."""
        dev = qp.device(device, wires=3)

        op, controlled_op = ops

        @qp.qnode(dev)
        def normal_circuit(rads):
            qp.Hadamard(0)
            controlled_op(rads, wires=[0, 1])
            return qp.probs(wires=1)

        @qp.defer_measurements
        @qp.qnode(dev)
        def quantum_control_circuit(rads):
            qp.Hadamard(0)
            m_0 = qp.measure(0)
            qp.cond(m_0, op)(rads, wires=1)
            return qp.probs(wires=1)

        normal_probs = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(normal_probs, cond_probs)

    def test_hermitian_queued(self):
        """Test that the defer_measurements transform works with
        qp.Hermitian."""
        rads = 0.3

        mat = np.eye(8)
        measurement = qp.expval(qp.Hermitian(mat, wires=[3, 1, 2]))

        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0, reset=True)
            qp.cond(m_0, qp.RY)(rads, wires=4)
            qp.apply(measurement)

        tape = qp.tape.QuantumScript.from_queue(q)
        tapes, _ = qp.defer_measurements(tape)
        tape = tapes[0]

        assert len(tape.operations) == 3
        assert len(tape.measurements) == 1

        # Check the underlying CNOT for storing measurement state
        meas_op1 = tape.operations[0]
        assert isinstance(meas_op1, qp.CNOT)
        assert meas_op1.wires == qp.wires.Wires([0, 5])

        # Check the underlying CNOT for resetting measured wire
        meas_op1 = tape.operations[1]
        assert isinstance(meas_op1, qp.CNOT)
        assert meas_op1.wires == qp.wires.Wires([5, 0])

        # Check the underlying Controlled instances
        first_ctrl_op = tape.operations[2]
        assert isinstance(first_ctrl_op, Controlled)
        qp.assert_equal(first_ctrl_op.base, qp.RY(rads, 4))

        assert len(tape.measurements) == 1
        qp.assert_equal(tape.measurements[0], measurement)

    def test_hamiltonian_queued(self):
        """Test that the defer_measurements transform works with
        qp.Hamiltonian."""
        rads = 0.3
        a = qp.PauliX(3)
        b = qp.PauliX(1)
        c = qp.PauliZ(2)
        obs = [a, b, c]
        coeffs = [1.0, 2.0, 3.0]

        H = qp.Hamiltonian(coeffs, obs, grouping_type="qwc")

        with qp.queuing.AnnotatedQueue() as q:
            m_0 = qp.measure(0)
            qp.cond(m_0, qp.RY)(rads, wires=4)
            qp.expval(H)

        tape = qp.tape.QuantumScript.from_queue(q)
        tapes, _ = qp.defer_measurements(tape)
        tape = tapes[0]
        assert len(tape.operations) == 1
        assert len(tape.measurements) == 1

        # Check the underlying Controlled instance
        first_ctrl_op = tape.operations[0]
        assert isinstance(first_ctrl_op, Controlled)
        qp.assert_equal(first_ctrl_op.base, qp.RY(rads, 4))
        assert len(tape.measurements) == 1
        assert isinstance(tape.measurements[0], qp.measurements.MeasurementProcess)
        qp.assert_equal(tape.measurements[0].obs, H)

    @pytest.mark.parametrize(
        "device",
        [
            "default.qubit",
            "default.mixed",
            "lightning.qubit",
        ],
    )
    @pytest.mark.parametrize("ops", [(qp.RX, qp.CRX), (qp.RY, qp.CRY), (qp.RZ, qp.CRZ)])
    def test_conditional_rotations_assert_zero_state(self, device, ops):
        """Test that the quantum conditional operations applied by controlling
        on the zero outcome match the output of controlled rotations."""
        dev = qp.device(device, wires=3)
        r = 2.345

        op, controlled_op = ops

        @qp.qnode(dev)
        def normal_circuit(rads):
            qp.Hadamard(0)
            controlled_op(rads, wires=[0, 1])
            return qp.probs(wires=1)

        @qp.defer_measurements
        @qp.qnode(dev)
        def quantum_control_circuit(rads):
            qp.Hadamard(0)
            qp.PauliX(0)
            m_0 = qp.measure(0)
            qp.cond(m_0 == 0, op)(rads, wires=1)
            return qp.probs(wires=1)

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
        """Test that an else operation can also defined using qp.cond."""
        dev = qp.device(device, wires=3)
        r = 2.345

        op1, controlled_op1 = qp.RY, qp.CRY
        op2, controlled_op2 = qp.RX, qp.CRX

        @qp.qnode(dev)
        def normal_circuit(rads):
            qp.Hadamard(0)
            controlled_op1(rads, wires=[0, 1])

            qp.PauliX(0)
            controlled_op2(rads, wires=[0, 1])
            qp.PauliX(0)
            return qp.probs(wires=1)

        @qp.defer_measurements
        @qp.qnode(dev)
        def quantum_control_circuit(rads):
            qp.Hadamard(0)
            m_0 = qp.measure(0)
            qp.cond(m_0, op1, op2)(rads, wires=1)
            return qp.probs(wires=1)

        exp = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(exp, cond_probs)

    def test_keyword_syntax(self):
        """Test that passing an argument to the conditioned operation using the
        keyword syntax works."""
        op = qp.RY

        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def qnode1(par):
            qp.Hadamard(0)
            qp.ctrl(op, control=0)(phi=par, wires=1)
            return qp.expval(qp.PauliZ(1))

        @qp.defer_measurements
        @qp.qnode(dev)
        def qnode2(par):
            qp.Hadamard(0)
            m_0 = qp.measure(0)
            qp.cond(m_0, op)(phi=par, wires=1)
            return qp.expval(qp.PauliZ(1))

        par = np.array(0.3)

        assert np.allclose(qnode1(par), qnode2(par))

    @pytest.mark.parametrize("control_val, expected", [(0, -1), (1, 1)])
    def test_condition_using_measurement_outcome(self, control_val, expected):
        """Apply a conditional bitflip by selecting the measurement
        outcome."""
        dev = qp.device("default.qubit", wires=3)

        @qp.defer_measurements
        @qp.qnode(dev)
        def qnode():
            m_0 = qp.measure(0)
            qp.cond(m_0 == control_val, qp.PauliX)(wires=1)
            return qp.expval(qp.PauliZ(1))

        assert qnode() == expected

    @pytest.mark.parametrize(
        "device",
        ["default.qubit", "default.mixed", "lightning.qubit"],
    )
    def test_cond_qfunc(self, device):
        """Test that a qfunc can also used with qp.cond."""
        dev = qp.device(device, wires=4)

        r = 2.324

        @qp.qnode(dev)
        def normal_circuit(rads):
            qp.Hadamard(0)

            qp.CNOT(wires=[0, 1])
            qp.CRY(rads, wires=[0, 1])
            qp.CZ(wires=[0, 1])
            qp.ctrl(qp.CRX, control=0, control_values=[1])(0.5, [1, 2])
            return qp.probs(wires=[1, 2])

        def f(x):
            qp.PauliX(1)
            qp.RY(x, wires=1)
            qp.PauliZ(1)
            qp.CRX(0.5, [1, 2])

        @qp.defer_measurements
        @qp.qnode(dev)
        def quantum_control_circuit(r):
            qp.Hadamard(0)
            m_0 = qp.measure(0)
            qp.cond(m_0, f)(r)
            return qp.probs(wires=[1, 2])

        exp = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(exp, cond_probs)

    @pytest.mark.parametrize(
        "device",
        ["default.qubit", "default.mixed", "lightning.qubit"],
    )
    def test_cond_qfunc_with_else(self, device):
        """Test that a qfunc can also used with qp.cond even when an else
        qfunc is provided."""
        dev = qp.device(device, wires=3)

        x = 0.3
        y = 3.123

        @qp.qnode(dev)
        def normal_circuit(x, y):
            qp.RY(x, wires=1)

            qp.ctrl(f, 1)(y)

            # Flip the qubit before/after to control on 0
            qp.PauliX(1)
            qp.ctrl(g, 1)(y)
            qp.PauliX(1)
            return qp.probs(wires=[0])

        def f(a):
            qp.PauliX(0)
            qp.RY(a, wires=0)
            qp.PauliZ(0)

        def g(a):
            qp.RX(a, wires=0)
            qp.PhaseShift(a, wires=0)

        @qp.defer_measurements
        @qp.qnode(dev)
        def cond_qnode(x, y):
            qp.RY(x, wires=1)
            m_0 = qp.measure(1)
            qp.cond(m_0, f, g)(y)
            return qp.probs(wires=[0])

        assert np.allclose(normal_circuit(x, y), cond_qnode(x, y))

    def test_cond_on_measured_wire(self):
        """Test that applying a conditional operation on the same wire
        that is measured works as expected."""
        dev = qp.device("default.qubit", wires=2)

        @qp.defer_measurements
        @qp.qnode(dev)
        def qnode():
            qp.Hadamard(0)
            m = qp.measure(0)
            qp.cond(m, qp.PauliX)(0)
            return qp.density_matrix(0)

        # Above circuit will cause wire 0 to go back to the |0> computational
        # basis state. We can inspect the reduced density matrix to confirm this
        # without inspecting the extra wires
        expected_dmat = np.array([[1, 0], [0, 0]])
        assert np.allclose(qnode(), expected_dmat)


class TestExpressionConditionals:
    """Test Conditionals that rely on expressions of mid-circuit measurements."""

    @pytest.mark.parametrize("r", np.linspace(0.1, 2 * np.pi - 0.1, 4))
    @pytest.mark.parametrize("op", [qp.RX, qp.RY, qp.RZ])
    def test_conditional_rotations(self, r, op):
        """Test that the quantum conditional operations match the output of
        controlled rotations. And additionally that summing measurements works as expected."""
        dev = qp.device("default.qubit", wires=5)

        @qp.qnode(dev)
        def normal_circuit(rads):
            qp.RX(2.4, wires=0)
            qp.RY(1.3, wires=1)
            qp.ctrl(op, control=(0, 1), control_values=[True, True])(rads, wires=2)
            return qp.probs(wires=2)

        @qp.defer_measurements
        @qp.qnode(dev)
        def quantum_control_circuit(rads):
            qp.RX(2.4, wires=0)
            qp.RY(1.3, wires=1)
            m_0 = qp.measure(0)
            m_1 = qp.measure(1)
            qp.cond(m_0 + m_1 == 2, op)(rads, wires=2)
            return qp.probs(wires=2)

        normal_probs = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(normal_probs, cond_probs)

    @pytest.mark.parametrize("r", np.linspace(0.1, 2 * np.pi - 0.1, 4))
    def test_triple_measurement_condition_expression(self, r):
        """Test that combining the results of three mid-circuit measurements works as expected."""
        dev = qp.device("default.qubit", wires=7)

        @qp.defer_measurements
        @qp.qnode(dev)
        def normal_circuit(rads):
            qp.RX(2.4, wires=0)
            qp.RY(1.3, wires=1)
            qp.RX(1.7, wires=2)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.CNOT(wires=[0, 2])

            qp.ctrl(qp.RX, (0, 1, 2), [False, True, True])(rads, wires=3)

            return qp.probs(wires=3)

        @qp.defer_measurements
        @qp.qnode(dev)
        def quantum_control_circuit(rads):
            qp.RX(2.4, wires=0)
            qp.RY(1.3, wires=1)
            qp.RX(1.7, wires=2)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.CNOT(wires=[0, 2])
            m_0 = qp.measure(0)
            m_1 = qp.measure(1)
            m_2 = qp.measure(2)

            expression = 4 * m_0 + 2 * m_1 + m_2
            qp.cond(expression == 3, qp.RX)(rads, wires=3)
            return qp.probs(wires=3)

        normal_probs = normal_circuit(r)
        cond_probs = quantum_control_circuit(r)

        assert np.allclose(normal_probs, cond_probs)

    def test_multiple_conditions(self):
        """Test that when multiple "branches" of the mid-circuit measurements all satisfy the criteria then
        this translates to multiple control gates.
        """
        dev = qp.device("default.qubit", wires=7)

        @qp.defer_measurements
        @qp.qnode(dev)
        def normal_circuit(rads):
            qp.RX(2.4, wires=0)
            qp.RY(1.3, wires=1)
            qp.RX(1.7, wires=2)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.CNOT(wires=[0, 2])

            qp.ctrl(qp.RX, (0, 1, 2), [False, True, True])(rads, wires=3)
            qp.ctrl(qp.RX, (0, 1, 2), [True, False, False])(rads, wires=3)
            qp.ctrl(qp.RX, (0, 1, 2), [True, False, True])(rads, wires=3)
            qp.ctrl(qp.RX, (0, 1, 2), [True, True, False])(rads, wires=3)

            return qp.probs(wires=3)

        @qp.defer_measurements
        @qp.qnode(dev)
        def quantum_control_circuit(rads):
            qp.RX(2.4, wires=0)
            qp.RY(1.3, wires=1)
            qp.RX(1.7, wires=2)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.CNOT(wires=[0, 2])
            m_0 = qp.measure(0)
            m_1 = qp.measure(1)
            m_2 = qp.measure(2)

            expression = 4 * m_0 + 2 * m_1 + m_2
            qp.cond((expression >= 3) & (expression <= 6), qp.RX)(rads, wires=3)
            return qp.probs(wires=3)

        normal_probs = normal_circuit(1.0)
        cond_probs = quantum_control_circuit(1.0)

        assert np.allclose(normal_probs, cond_probs)

    def test_composed_conditions(self):
        """Test that a complex nested expression gets resolved correctly to the corresponding correct control gates."""
        dev = qp.device("default.qubit", wires=7)

        @qp.defer_measurements
        @qp.qnode(dev)
        def normal_circuit(rads):
            qp.RX(2.4, wires=0)
            qp.RY(1.3, wires=1)
            qp.RX(1.7, wires=2)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.CNOT(wires=[0, 2])

            qp.ctrl(qp.RX, (0, 1, 2), [False, False, False])(rads, wires=3)
            qp.ctrl(qp.RX, (0, 1, 2), [False, False, True])(rads, wires=3)
            qp.ctrl(qp.RX, (0, 1, 2), [True, False, True])(rads, wires=3)
            qp.ctrl(qp.RX, (0, 1, 2), [False, True, False])(rads, wires=3)
            qp.ctrl(qp.RX, (0, 1, 2), [False, True, True])(rads, wires=3)

            return qp.probs(wires=3)

        @qp.defer_measurements
        @qp.qnode(dev)
        def quantum_control_circuit(rads):
            qp.RX(2.4, wires=0)
            qp.RY(1.3, wires=1)
            qp.RX(1.7, wires=2)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            qp.CNOT(wires=[0, 2])
            m_0 = qp.measure(0)
            expr1 = 2 * m_0
            m_1 = qp.measure(1)
            expr2 = (3 * m_1 + 2) * (4 * expr1 + 2)
            m_2 = qp.measure(2)
            expr3 = expr2 / (m_2 + 3)
            qp.cond(expr3 <= 6, qp.RX)(rads, wires=3)
            return qp.probs(wires=3)

        normal_probs = normal_circuit(1.0)
        cond_probs = quantum_control_circuit(1.0)

        assert np.allclose(normal_probs, cond_probs)


class TestTemplates:
    """Tests templates being conditioned on mid-circuit measurement outcomes."""

    def test_angle_embedding(self):
        """Test the angle embedding template conditioned on mid-circuit
        measurement outcomes."""
        template = qp.AngleEmbedding
        feature_vector = [1, 2, 3]

        dev = qp.device("default.qubit", wires=6)

        @qp.qnode(dev)
        def qnode1():
            qp.Hadamard(0)
            qp.ctrl(template, control=0)(features=feature_vector, wires=range(1, 5), rotation="Z")
            return qp.expval(qp.PauliZ(1) @ qp.PauliZ(2) @ qp.PauliZ(3) @ qp.PauliZ(4))

        @qp.qnode(dev)
        @qp.defer_measurements
        def qnode2():
            qp.Hadamard(0)
            m_0 = qp.measure(0)
            qp.cond(m_0, template)(features=feature_vector, wires=range(1, 5), rotation="Z")
            return qp.expval(qp.PauliZ(1) @ qp.PauliZ(2) @ qp.PauliZ(3) @ qp.PauliZ(4))

        res1 = qnode1()
        res2 = qnode2()

        assert np.allclose(res1, res2)

        tape1 = qp.workflow.construct_tape(qnode1)()
        tape2 = qp.workflow.construct_tape(qnode2)()
        assert len(tape2.operations) == len(tape1.operations)
        assert len(tape1.measurements) == len(tape2.measurements)

        # Check the operations
        for op1, op2 in zip(tape1.operations, tape2.operations):
            assert isinstance(op1, type(op2))
            assert np.allclose(op1.data, op2.data)

        # Check the measurements
        for op1, op2 in zip(tape1.measurements, tape2.measurements):
            assert isinstance(op1, type(op2))

    @pytest.mark.parametrize("template", [qp.StronglyEntanglingLayers, qp.BasicEntanglerLayers])
    def test_layers(self, template):
        """Test layers conditioned on mid-circuit measurement outcomes."""
        dev = qp.device("default.qubit", wires=4)

        num_wires = 2

        @qp.qnode(dev)
        def qnode1(parameters):
            qp.Hadamard(0)
            qp.ctrl(template, control=0)(parameters, wires=range(1, 3))
            return qp.expval(qp.PauliZ(1) @ qp.PauliZ(2))

        @qp.qnode(dev)
        @qp.defer_measurements
        def qnode2(parameters):
            qp.Hadamard(0)
            m_0 = qp.measure(0)
            qp.cond(m_0, template)(parameters, wires=range(1, 3))
            return qp.expval(qp.PauliZ(1) @ qp.PauliZ(2))

        shape = template.shape(n_layers=2, n_wires=num_wires)
        weights = np.random.random(size=shape)

        assert np.allclose(qnode1(weights), qnode2(weights))
        tape1 = qp.workflow.construct_tape(qnode1)(weights)
        tape2 = qp.workflow.construct_tape(qnode2)(weights)
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
    """Tests for the qubit reuse/reset functionality of `qp.measure`."""

    def test_new_wire_for_multiple_measurements(self):
        """Test that a new wire is added if there are multiple mid-circuit measurements
        on the same wire."""
        dev = qp.device("default.qubit", wires=4)

        @qp.qnode(dev)
        @qp.defer_measurements
        def circ(x, y):
            qp.RX(x, 0)
            qp.measure(0)
            qp.RY(y, 1)
            qp.measure(0)
            qp.RZ(x + y, 1)
            qp.measure(0)
            return qp.expval(qp.PauliZ(1))

        expected = [
            qp.RX(1.0, 0),
            qp.CNOT([0, 2]),
            qp.RY(2.0, 1),
            qp.CNOT([0, 3]),
            qp.RZ(3.0, 1),
        ]

        tape = qp.workflow.construct_tape(circ)(1.0, 2.0)
        assert tape.operations == expected

    def test_correct_cnot_for_reset(self):
        """Test that a CNOT is applied from the wire that stores the measurement
        to the measured wire after the measurement."""
        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def qnode1(x):
            qp.Hadamard(0)
            qp.CRX(x, [0, 1])
            return qp.expval(qp.PauliZ(1))

        @qp.qnode(dev)
        @qp.defer_measurements
        def qnode2(x):
            qp.Hadamard(0)
            m0 = qp.measure(0, reset=True)
            qp.cond(m0, qp.RX)(x, 1)
            return qp.expval(qp.PauliZ(1))

        assert np.allclose(qnode1(0.123), qnode2(0.123))

        expected_circuit = [
            qp.Hadamard(0),
            qp.CNOT([0, 2]),
            qp.CNOT([2, 0]),
            qp.CRX(0.123, wires=[2, 1]),
            qp.expval(qp.PauliZ(1)),
        ]

        tape2 = qp.workflow.construct_tape(qnode2)(0.123)
        assert len(tape2.circuit) == len(expected_circuit)
        for actual, expected in zip(tape2.circuit, expected_circuit):
            qp.assert_equal(actual, expected)

    def test_measurements_add_new_qubits(self):
        """Test that qubit reuse related logic is applied if a wire with mid-circuit
        measurements is included in terminal measurements."""
        tape = qp.tape.QuantumScript(
            ops=[qp.Hadamard(0), MidMeasure(0)], measurements=[qp.density_matrix(wires=[0])]
        )
        expected = np.eye(2) / 2

        tapes, _ = qp.defer_measurements(tape)

        dev = qp.device("default.qubit")
        res = qp.execute(tapes, dev)

        assert np.allclose(res, expected)

        deferred_tape = tapes[0]
        assert deferred_tape.operations == [qp.Hadamard(0), qp.CNOT([0, 1])]
        assert deferred_tape.measurements == [qp.density_matrix(wires=[0])]

    def test_wire_is_reset(self):
        """Test that a wire is reset to the |0> state without any local phases
        after measurement if reset is requested."""
        dev = qp.device("default.qubit", wires=3)

        @qp.defer_measurements
        @qp.qnode(dev)
        def qnode(x):
            qp.Hadamard(0)
            qp.PhaseShift(np.pi / 4, 0)
            m = qp.measure(0, reset=True)
            qp.cond(m, qp.RX)(x, 1)
            return qp.density_matrix(wires=[0])

        # Expected reduced density matrix on wire 0
        expected_mat = np.array([[1, 0], [0, 0]])
        assert np.allclose(qnode(0.123), expected_mat)

    def test_multiple_measurements_mixed_reset(self, mocker):
        """Test that a QNode with multiple mid-circuit measurements with
        different resets is transformed correctly."""
        dev = qp.device("default.qubit", wires=6)

        @qp.qnode(dev)
        def qnode(p, x, y):
            qp.Hadamard(0)
            qp.PhaseShift(p, 0)
            # Set measurement_ids so that the order of wires in combined
            # measurement values is consistent

            mp0 = qp.ops.MidMeasure(0, reset=True, id=0)
            m0 = qp.ops.MeasurementValue([mp0], lambda v: v)
            qp.cond(~m0, qp.RX)(x, 1)
            mp1 = qp.ops.MidMeasure(1, reset=True, id=1)
            m1 = qp.ops.MeasurementValue([mp1], lambda v: v)
            qp.cond(m0 & m1, qp.Hadamard)(0)
            mp2 = qp.ops.MidMeasure(0, id=2)
            m2 = qp.ops.MeasurementValue([mp2], lambda v: v)
            qp.cond(m1 | m2, qp.RY)(y, 2)
            return qp.expval(qp.PauliZ(2))

        spy = mocker.spy(qp.defer_measurements, "_tape_transform")
        _ = qnode(0.123, 0.456, 0.789)
        assert spy.call_count == 1

        expected_circuit = [
            qp.Hadamard(0),
            qp.PhaseShift(0.123, 0),
            qp.CNOT([0, 3]),
            qp.CNOT([3, 0]),
            Controlled(qp.RX(0.456, 1), 3, [False]),
            qp.CNOT([1, 4]),
            qp.CNOT([4, 1]),
            Controlled(qp.Hadamard(0), [3, 4]),
            qp.CNOT([0, 5]),
            Controlled(qp.RY(0.789, 2), [4, 5], [False, True]),
            Controlled(qp.RY(0.789, 2), [4, 5], [True, False]),
            Controlled(qp.RY(0.789, 2), [4, 5], [True, True]),
            qp.expval(qp.PauliZ(2)),
        ]

        tape = qp.workflow.construct_tape(qnode)(0.123, 0.456, 0.789)
        deferred_tapes, _ = qp.defer_measurements(tape)
        deferred_tape = deferred_tapes[0]
        assert len(deferred_tape.circuit) == len(expected_circuit)
        for actual, expected in zip(deferred_tape.circuit, expected_circuit):
            qp.assert_equal(actual, expected)


class TestDrawing:
    """Tests drawing circuits with mid-circuit measurements and conditional
    operations that have been transformed"""

    def test_drawing_no_reuse(self):
        """Test that drawing a func with mid-circuit measurements works and
        that controlled operations are drawn for conditional operations when
        the measured wires are not reused."""

        # TODO: Update after drawing for mid-circuit measurements is updated.

        def qfunc():
            m_0 = qp.measure(0)
            qp.cond(m_0, qp.RY)(0.312, wires=1)

            m_2 = qp.measure(2)
            qp.cond(m_2, qp.RY)(0.312, wires=1)
            return qp.expval(qp.PauliZ(1))

        dev = qp.device("default.qubit", wires=4)

        transformed_qfunc = qp.transforms.defer_measurements(qfunc)
        transformed_qnode = qp.QNode(transformed_qfunc, dev)

        expected = (
            "0: ─╭●──────────────────┤     \n"
            "1: ─╰RY(0.31)─╭RY(0.31)─┤  <Z>\n"
            "2: ───────────╰●────────┤     "
        )
        assert qp.draw(transformed_qnode)() == expected

    def test_drawing_with_reuse(self):
        """Test that drawing a func with mid-circuit measurements works and
        that controlled operations are drawn for conditional operations when
        the measured wires are reused."""

        # TODO: Update after drawing for mid-circuit measurements is updated.

        def qfunc():
            m_0 = qp.measure(0, reset=True)
            qp.cond(m_0, qp.RY)(0.312, wires=1)

            m_2 = qp.measure(2)
            qp.cond(m_2, qp.RY)(0.312, wires=1)
            return qp.expval(qp.PauliZ(1))

        dev = qp.device("default.qubit", wires=4)

        transformed_qfunc = qp.transforms.defer_measurements(qfunc)
        transformed_qnode = qp.QNode(transformed_qfunc, dev)

        expected = (
            "0: ─╭●─╭X─────────────────────┤     \n"
            "1: ─│──│──╭RY(0.31)─╭RY(0.31)─┤  <Z>\n"
            "2: ─│──│──│─────────╰●────────┤     \n"
            "3: ─╰X─╰●─╰●──────────────────┤     "
        )
        assert qp.draw(transformed_qnode)() == expected

    @pytest.mark.parametrize(
        "mp, label",
        [
            (qp.sample, "Sample"),
            (qp.probs, "Probs"),
            (qp.var, "Var[None]"),
            (qp.counts, "Counts"),
            (qp.expval, "<None>"),
        ],
    )
    def test_drawing_with_mcm_terminal_measure(self, mp, label):
        """Test that drawing a func works correctly when collecting statistics on
        mid-circuit measurements."""

        def qfunc():
            m_0 = qp.measure(0, reset=True)
            qp.cond(m_0, qp.RY)(0.312, wires=1)

            return mp(op=m_0), qp.expval(qp.Z(1))

        dev = qp.device("default.qubit", wires=4)

        transformed_qfunc = qp.transforms.defer_measurements(qfunc)
        transformed_qnode = qp.QNode(transformed_qfunc, dev)

        spaces = " " * len(label)
        expval = "<Z>".ljust(len(label))
        expected = (
            f"0: ─╭●─╭X───────────┤  {spaces}\n"
            f"1: ─│──│──╭RY(0.31)─┤  {expval}\n"
            f"2: ─╰X─╰●─╰●────────┤  {label}"
        )
        assert qp.draw(transformed_qnode)() == expected

    @pytest.mark.parametrize("mp", [qp.sample, qp.probs, qp.var, qp.counts, qp.expval])
    def test_draw_mpl_with_mcm_terminal_measure(self, mp):
        """Test that no error is raised when drawing a circuit which collects
        statistics on mid-circuit measurements"""

        def qfunc():
            m_0 = qp.measure(0, reset=True)
            qp.cond(m_0, qp.RY)(0.312, wires=1)

            return mp(op=m_0), qp.expval(qp.Z(1))

        dev = qp.device("default.qubit", wires=4)

        transformed_qfunc = qp.transforms.defer_measurements(qfunc)
        transformed_qnode = qp.QNode(transformed_qfunc, dev)
        _ = qp.draw_mpl(transformed_qnode)()


def test_custom_wire_labels_allowed_without_reuse():
    """Test that custom wire labels work if no qubits are re-used."""
    with qp.queuing.AnnotatedQueue() as q:
        qp.Hadamard("a")
        ma = qp.measure("a", reset=False)
        qp.cond(ma, qp.PauliX)("b")
        qp.probs(wires="b")

    tape = qp.tape.QuantumScript.from_queue(q)
    tapes, _ = qp.defer_measurements(tape)
    tape = tapes[0]

    assert len(tape) == 3
    qp.assert_equal(tape[0], qp.Hadamard("a"))
    qp.assert_equal(tape[1], qp.CNOT(["a", "b"]))
    qp.assert_equal(tape[2], qp.probs(wires="b"))


def test_integer_wire_labels_with_reset():
    """Tests that integer wire labels work when qubits are re-used."""

    # Reset example
    with qp.queuing.AnnotatedQueue() as q:
        qp.Hadamard(0)
        ma = qp.measure(0, reset=True)
        qp.cond(ma, qp.PauliX)(1)
        qp.probs(wires=0)

    tape = qp.tape.QuantumScript.from_queue(q)
    tapes, _ = qp.defer_measurements(tape)
    tape = tapes[0]

    assert len(tape) == 5
    qp.assert_equal(tape[0], qp.Hadamard(0))
    qp.assert_equal(tape[1], qp.CNOT([0, 2]))
    qp.assert_equal(tape[2], qp.CNOT([2, 0]))
    qp.assert_equal(tape[3], qp.CNOT([2, 1]))
    qp.assert_equal(tape[4], qp.probs(wires=0))

    # Reuse example
    with qp.queuing.AnnotatedQueue() as q:
        qp.Hadamard(0)
        ma = qp.measure(0)
        qp.cond(ma, qp.PauliX)(1)
        qp.Hadamard(0)
        qp.probs(wires=1)

    tape = qp.tape.QuantumScript.from_queue(q)
    tapes, _ = qp.defer_measurements(tape)
    tape = tapes[0]

    assert len(tape) == 5
    qp.assert_equal(tape[0], qp.Hadamard(0))
    qp.assert_equal(tape[1], qp.CNOT([0, 2]))
    qp.assert_equal(tape[2], qp.CNOT([2, 1]))
    qp.assert_equal(tape[3], qp.Hadamard(0))
    qp.assert_equal(tape[4], qp.probs(wires=1))


def test_custom_wire_labels_with_reset():
    """Test that custom wire labels work if any qubits are re-used."""

    # Reset example (should be the same circuit as in previous test but with wire labels)
    with qp.queuing.AnnotatedQueue() as q:
        qp.Hadamard("a")
        ma = qp.measure("a", reset=True)
        qp.cond(ma, qp.PauliX)("b")
        qp.probs(wires="a")

    tape = qp.tape.QuantumScript.from_queue(q)
    tapes, _ = qp.defer_measurements(tape)
    tape = tapes[0]

    assert len(tape) == 5
    qp.assert_equal(tape[0], qp.Hadamard("a"))
    qp.assert_equal(tape[1], qp.CNOT(["a", 0]))
    qp.assert_equal(tape[2], qp.CNOT([0, "a"]))
    qp.assert_equal(tape[3], qp.CNOT([0, "b"]))
    qp.assert_equal(tape[4], qp.probs(wires="a"))

    # Reuse example (should be the same circuit as in previous test but with wire labels)
    with qp.queuing.AnnotatedQueue() as q:
        qp.Hadamard("a")
        ma = qp.measure("a")
        qp.cond(ma, qp.PauliX)("b")
        qp.Hadamard("a")
        qp.probs(wires="b")

    tape = qp.tape.QuantumScript.from_queue(q)
    tapes, _ = qp.defer_measurements(tape)
    tape = tapes[0]

    assert len(tape) == 5
    qp.assert_equal(tape[0], qp.Hadamard("a"))
    qp.assert_equal(tape[1], qp.CNOT(["a", 0]))
    qp.assert_equal(tape[2], qp.CNOT([0, "b"]))
    qp.assert_equal(tape[3], qp.Hadamard("a"))
    qp.assert_equal(tape[4], qp.probs(wires="b"))
