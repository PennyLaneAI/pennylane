# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for qutrit mixed device preprocessing."""
import warnings

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.devices.default_qutrit_mixed import (
    DefaultQutritMixed,
    observable_stopping_condition,
    stopping_condition,
)
from pennylane.exceptions import DeviceError


class NoMatOp(qml.operation.Operation):
    """Dummy operation for expanding circuit."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False

    def decomposition(self):
        return [qml.TShift(self.wires), qml.TClock(self.wires)]


# pylint: disable=too-few-public-methods
class NoMatNoDecompOp(qml.operation.Operation):
    """Dummy operation for checking check_validity throws error when
    expected."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False


# pylint: disable=too-few-public-methods
class TestPreprocessing:
    """Unit tests for the preprocessing method."""

    def test_error_if_device_option_not_available(self):
        """Test that an error is raised if a device option is requested but not a valid option."""
        dev = DefaultQutritMixed()

        config = ExecutionConfig(device_options={"bla": "val"})
        with pytest.raises(DeviceError, match="device option bla"):
            dev.preprocess(config)

    def test_chooses_best_gradient_method(self):
        """Test that preprocessing chooses backprop as the best gradient method."""
        dev = DefaultQutritMixed()

        config = ExecutionConfig(gradient_method="best")

        new_config = dev.setup_execution_config(config)

        assert new_config.gradient_method == "backprop"
        assert not new_config.use_device_gradient
        assert not new_config.grad_on_execution

    def test_circuit_wire_validation(self):
        """Test that preprocessing validates wires on the circuits being executed."""
        dev = DefaultQutritMixed(wires=3)

        circuit_valid_0 = qml.tape.QuantumScript([qml.TShift(0)])
        program = dev.preprocess_transforms()
        circuits, _ = program([circuit_valid_0])
        assert circuits[0].circuit == circuit_valid_0.circuit

        circuit_valid_1 = qml.tape.QuantumScript([qml.TShift(1)])
        program = dev.preprocess_transforms()
        circuits, _ = program([circuit_valid_0, circuit_valid_1])
        assert circuits[0].circuit == circuit_valid_0.circuit
        assert circuits[1].circuit == circuit_valid_1.circuit

        invalid_circuit = qml.tape.QuantumScript([qml.TShift(4)])
        program = dev.preprocess_transforms()

        with pytest.raises(qml.wires.WireError, match=r"Cannot run circuit\(s\) on"):
            program([invalid_circuit])

        with pytest.raises(qml.wires.WireError, match=r"Cannot run circuit\(s\) on"):
            program([circuit_valid_0, invalid_circuit])

    @pytest.mark.parametrize(
        "mp_fn,mp_cls,shots",
        [
            (qml.sample, qml.measurements.SampleMP, 10),
            (qml.state, qml.measurements.StateMP, None),
            (qml.probs, qml.measurements.ProbabilityMP, None),
        ],
    )
    def test_measurement_is_swapped_out(self, mp_fn, mp_cls, shots):
        """Test that preprocessing swaps out any MeasurementProcess with no wires or obs"""
        dev = DefaultQutritMixed(wires=3)
        original_mp = mp_fn()
        exp_z = qml.expval(qml.GellMann(0, 3))
        qs = qml.tape.QuantumScript([qml.THadamard(0)], [original_mp, exp_z], shots=shots)
        program = dev.preprocess_transforms()
        tapes, _ = program([qs])
        assert len(tapes) == 1
        tape = tapes[0]
        assert tape.operations == qs.operations
        assert tape.measurements != qs.measurements
        qml.assert_equal(tape.measurements[0], mp_cls(wires=[0, 1, 2]))
        assert tape.measurements[1] is exp_z

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.TShift(0), True),
            (qml.GellMann(0, 1), False),
            (qml.Snapshot(), True),
            (qml.TRX(1.1, 0), True),
            (qml.QutritDepolarizingChannel(0.4, 0), True),
            (qml.QutritAmplitudeDamping(0.1, 0.2, 0.12, 0), True),
            (qml.TritFlip(0.4, 0.1, 0.02, 0), True),
        ],
    )
    def test_accepted_operator(self, op, expected):
        """Test that stopping_condition works correctly"""
        res = stopping_condition(op)
        assert res == expected

    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.TShift(0), False),
            (qml.QutritDepolarizingChannel(0.4, 0), False),
            (qml.GellMann(0, 1), True),
            (qml.Snapshot(), False),
            (qml.ops.op_math.SProd(1.2, qml.GellMann(0, 1)), True),
            (qml.sum(qml.ops.op_math.SProd(1.2, qml.GellMann(0, 1)), qml.GellMann(1, 3)), True),
            (qml.ops.op_math.Prod(qml.GellMann(0, 1), qml.GellMann(3, 3)), True),
        ],
    )
    def test_accepted_observable(self, obs, expected):
        """Test that observable_stopping_condition works correctly"""
        res = observable_stopping_condition(obs)
        assert res == expected


class TestPreprocessingIntegration:
    """Test preprocess produces output that can be executed by the device."""

    def test_batch_transform_no_batching(self):
        """Test that batch_transform does nothing when no batching is required."""
        ops = [qml.THadamard(0), qml.TAdd([0, 1]), qml.TRX(0.123, wires=1)]
        measurements = [qml.expval(qml.GellMann(1, 3))]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)
        device = DefaultQutritMixed()

        program = device.preprocess_transforms()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        assert tapes[0].circuit == ops + measurements

    def test_batch_transform_broadcast(self):
        """Test that batch_transform does nothing when batching is required but
        internal PennyLane broadcasting can be used (diff method != adjoint)"""
        ops = [qml.THadamard(0), qml.TAdd([0, 1]), qml.TRX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.GellMann(1, 3))]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)
        device = DefaultQutritMixed()

        program = device.preprocess_transforms()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        assert tapes[0].circuit == ops + measurements

    def test_preprocess_batch_transform(self):
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qml.THadamard(0), qml.TAdd([0, 1]), qml.TRX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.GellMann(0, 4)), qml.expval(qml.GellMann(1, 3))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        program = DefaultQutritMixed().preprocess_transforms()
        res_tapes, batch_fn = program(tapes)

        assert len(res_tapes) == 2
        for res_tape, measurement in zip(res_tapes, measurements):
            for op, expected_op in zip(res_tape.operations, ops):
                qml.assert_equal(op, expected_op)
            assert res_tape.measurements == [measurement]

        val = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_expand(self):
        """Test that preprocess returns the correct tapes when expansion is needed."""
        ops = [qml.THadamard(0), NoMatOp(1), qml.TRZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.GellMann(0, 3))], [qml.expval(qml.GellMann(1, 1))]]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=measurements[0]),
            qml.tape.QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        program, _ = DefaultQutritMixed().preprocess()
        res_tapes, batch_fn = program(tapes)

        expected = [qml.THadamard(0), qml.TShift(1), qml.TClock(1), qml.TRZ(0.123, wires=1)]

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, exp in zip(t.circuit, expected + measurements[i]):
                qml.assert_equal(op, exp)

        val = (("a", "b"), "c", "d")
        assert batch_fn(val) == (("a", "b"), "c")

    def test_preprocess_batch_and_expand(self):
        """Test that preprocess returns the correct tapes when batching and expanding
        is needed."""
        ops = [qml.THadamard(0), NoMatOp(1), qml.TRX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.GellMann(0, 1)), qml.expval(qml.GellMann(1, 3))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        program = DefaultQutritMixed().preprocess_transforms()
        res_tapes, batch_fn = program(tapes)
        expected_ops = [
            qml.THadamard(0),
            qml.TShift(1),
            qml.TClock(1),
            qml.TRX([np.pi, np.pi / 2], wires=1),
        ]

        assert len(res_tapes) == 2
        for res_tape, measurement in zip(res_tapes, measurements):
            for op, expected_op in zip(res_tape.operations, expected_ops):
                qml.assert_equal(op, expected_op)
            assert res_tape.measurements == [measurement]

        val = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_check_validity_fail(self):
        """Test that preprocess throws an error if the batched and expanded tapes have
        unsupported operators."""
        ops = [qml.THadamard(0), NoMatNoDecompOp(1), qml.TRZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.GellMann(0, 3))], [qml.expval(qml.GellMann(1, 1))]]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=measurements[0]),
            qml.tape.QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        program = DefaultQutritMixed().preprocess_transforms()
        with pytest.raises(DeviceError, match="Operator NoMatNoDecompOp"):
            program(tapes)

    @pytest.mark.parametrize(
        "relaxations,misclassifications,req_warn",
        [
            [(0.1, 0.2, 0.3), None, True],
            [None, (0.1, 0.2, 0.3), True],
            [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3), True],
            [None, None, False],
        ],
    )
    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.state()],
            [qml.density_matrix(0)],
            [qml.state(), qml.density_matrix([1, 2])],
            [qml.state(), qml.expval(qml.GellMann(1))],
        ],
    )
    def test_preprocess_warns_measurement_error_state(
        self, relaxations, misclassifications, req_warn, measurements
    ):
        """Test that preprocess raises a warning if there is an analytic state measurement and
        measurement error."""
        tapes = [
            qml.tape.QuantumScript(ops=[], measurements=measurements),
            qml.tape.QuantumScript(
                ops=[qml.THadamard(0), qml.TRZ(0.123, wires=1)], measurements=measurements
            ),
        ]
        device = DefaultQutritMixed(
            readout_relaxation_probs=relaxations, readout_misclassification_probs=misclassifications
        )
        program = device.preprocess_transforms()

        with warnings.catch_warnings(record=True) as warning:
            program(tapes)
            if req_warn:
                assert len(warning) != 0
                for warn in list(warning):
                    print(type(warn.message))
                    assert "is not affected by readout error" in warn.message.args[0]
            else:
                assert len(warning) == 0
