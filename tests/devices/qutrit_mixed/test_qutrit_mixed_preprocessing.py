# Copyright 2024 Xanadu Quantum Technologies Inc.

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
import pytest

import numpy as np

from pennylane import numpy as pnp
import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.devices.default_qutrit_mixed import DefaultQutritMixed, stopping_condition


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


class TestConfigSetup:
    """Tests involving setting up the execution config."""

    def test_error_if_device_option_not_available(self):
        """Test that an error is raised if a device option is requested but not a valid option."""
        dev = DefaultQutritMixed()

        config = ExecutionConfig(device_options={"bla": "val"})
        with pytest.raises(qml.DeviceError, match="device option bla"):
            dev.preprocess(config)

    def test_choose_best_gradient_method(self):
        """Test that preprocessing chooses backprop as the best gradient method."""
        dev = DefaultQutritMixed()

        config = ExecutionConfig(gradient_method="best")
        _, config = dev.preprocess(config)
        assert config.gradient_method == "backprop"
        assert config.use_device_gradient
        assert not config.grad_on_execution


# pylint: disable=too-few-public-methods
class TestPreprocessing:
    """Unit tests for the preprocessing method."""

    def test_chooses_best_gradient_method(self):
        """Test that preprocessing chooses backprop as the best gradient method."""
        dev = DefaultQutritMixed()

        config = ExecutionConfig(
            gradient_method="best", use_device_gradient=None, grad_on_execution=None
        )

        _, new_config = dev.preprocess(config)

        assert new_config.gradient_method == "backprop"
        assert new_config.use_device_gradient
        assert not new_config.grad_on_execution

    def test_circuit_wire_validation(self):
        """Test that preprocessing validates wires on the circuits being executed."""
        dev = DefaultQutritMixed(wires=3)

        circuit_valid_0 = qml.tape.QuantumScript([qml.TShift(0)])
        program, _ = dev.preprocess()
        circuits, _ = program([circuit_valid_0])
        assert circuits[0].circuit == circuit_valid_0.circuit

        circuit_valid_1 = qml.tape.QuantumScript([qml.TShift(1)])
        program, _ = dev.preprocess()
        circuits, _ = program([circuit_valid_0, circuit_valid_1])
        assert circuits[0].circuit == circuit_valid_0.circuit
        assert circuits[1].circuit == circuit_valid_1.circuit

        invalid_circuit = qml.tape.QuantumScript([qml.TShift(4)])
        with pytest.raises(qml.wires.WireError, match=r"Cannot run circuit\(s\) on"):
            program, _ = dev.preprocess()
            program(
                [
                    invalid_circuit,
                ]
            )

        with pytest.raises(qml.wires.WireError, match=r"Cannot run circuit\(s\) on"):
            program, _ = dev.preprocess()
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
        """Test that preprocessing swaps out any MP with no wires or obs"""
        dev = DefaultQutritMixed(wires=3)
        original_mp = mp_fn()
        exp_z = qml.expval(qml.GellMann(0, 3))
        qs = qml.tape.QuantumScript([qml.THadamard(0)], [original_mp, exp_z], shots=shots)
        program, _ = dev.preprocess()
        tapes, _ = program([qs])
        assert len(tapes) == 1
        tape = tapes[0]
        assert tape.operations == qs.operations
        assert tape.measurements != qs.measurements
        assert qml.equal(tape.measurements[0], mp_cls(wires=[0, 1, 2]))
        assert tape.measurements[1] is exp_z

    # @pytest.mark.parametrize( TODO
    #     "op, expected",
    #     [
    #         (qml.PauliX(0), True),
    #         (qml.CRX(0.1, wires=[0, 1]), True),
    #         (qml.Snapshot(), True),
    #         (qml.Barrier(), False),
    #         (qml.QFT(wires=range(5)), True),
    #         (qml.QFT(wires=range(10)), False),
    #         (qml.GroverOperator(wires=range(10)), True),
    #         (qml.GroverOperator(wires=range(14)), False),
    #         (qml.pow(qml.RX(1.1, 0), 3), True),
    #         (qml.pow(qml.RX(qml.numpy.array(1.1), 0), 3), False),
    #     ],
    # )
    # def test_accepted_operator(self, op, expected):
    #     """Test that _accepted_operator works correctly"""
    #     res = stopping_condition(op)
    #     assert res == expected


class TestPreprocessingIntegration:
    """Test preprocess produces output that can be executed by the device."""

    def test_batch_transform_no_batching(self):
        """Test that batch_transform does nothing when no batching is required."""
        ops = [qml.THadamard(0), qml.TAdd([0, 1]), qml.TRX(0.123, wires=1)]
        measurements = [qml.expval(qml.GellMann(1, 3))]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)
        device = DefaultQutritMixed()

        program, _ = device.preprocess()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        for op, expected in zip(tapes[0].circuit, ops + measurements):
            assert qml.equal(op, expected)

    def test_batch_transform_broadcast_not_adjoint(self):
        """Test that batch_transform does nothing when batching is required but
        internal PennyLane broadcasting can be used (diff method != adjoint)"""
        ops = [qml.THadamard(0), qml.TAdd([0, 1]), qml.TRX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.GellMann(1, 3))]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)
        device = DefaultQutritMixed()

        program, _ = device.preprocess()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        assert tapes[0].circuit == ops + measurements

    def test_preprocess_batch_transform_not_adjoint(self):  # TODO rename
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qml.THadamard(0), qml.TAdd([0, 1]), qml.TRX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.GellMann(0, 4)), qml.expval(qml.GellMann(1, 3))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        program, _ = DefaultQutritMixed().preprocess()
        res_tapes, batch_fn = program(tapes)

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, ops):
                assert qml.equal(op, expected_op)
            assert len(t.measurements) == 1
            if i == 0:
                assert qml.equal(t.measurements[0], measurements[0])
            else:
                assert qml.equal(t.measurements[0], measurements[1])

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
                assert qml.equal(op, exp)

        val = (("a", "b"), "c", "d")
        assert batch_fn(val) == (("a", "b"), "c")

    def test_preprocess_split_and_expand_not_adjoint(self):
        """Test that preprocess returns the correct tapes when splitting and expanding
        is needed."""
        ops = [qml.THadamard(0), NoMatOp(1), qml.TRX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.GellMann(0, 1)), qml.expval(qml.GellMann(1, 3))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        program, _ = DefaultQutritMixed().preprocess()
        res_tapes, batch_fn = program(tapes)
        expected_ops = [
            qml.THadamard(0),
            qml.TShift(1),
            qml.TClock(1),
            qml.TRX([np.pi, np.pi / 2], wires=1),
        ]

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, expected_ops):
                assert qml.equal(op, expected_op)
            assert len(t.measurements) == 1
            if i == 0:
                assert qml.equal(t.measurements[0], measurements[0])
            else:
                assert qml.equal(t.measurements[0], measurements[1])

        val = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_check_validity_fail(self):
        """Test that preprocess throws an error if the split and expanded tapes have
        unsupported operators."""
        ops = [qml.THadamard(0), NoMatNoDecompOp(1), qml.TRZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.GellMann(0, 3))], [qml.expval(qml.GellMann(1, 1))]]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=measurements[0]),
            qml.tape.QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        program, _ = DefaultQutritMixed().preprocess()
        with pytest.raises(qml.DeviceError, match="Operator NoMatNoDecompOp"):
            program(tapes)
