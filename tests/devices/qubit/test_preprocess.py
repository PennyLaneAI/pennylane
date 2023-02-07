# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for preprocess in devices/qubit."""

import pytest

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane.devices.qubit.preprocess import (
    _stopping_condition,
    _supports_observable,
    expand_fn,
    check_validity,
    batch_transform,
    preprocess,
)
from pennylane.measurements import MidMeasureMP, MeasurementValue
from pennylane.tape import QuantumScript
from pennylane import DeviceError

# pylint: disable=too-few-public-methods


class NoMatOp(Operation):
    """Dummy operation for expanding circuit."""

    # pylint: disable=missing-function-docstring
    num_wires = 1

    @property
    def has_matrix(self):
        return False

    def decomposition(self):
        return [qml.PauliX(self.wires), qml.PauliY(self.wires)]


class NoMatNoDecompOp(Operation):
    """Dummy operation for checking check_validity throws error when
    expected."""

    # pylint: disable=missing-function-docstring
    num_wires = 1

    @property
    def has_matrix(self):
        return False


class TestHelpers:
    """Unit tests for helper functions in qml.devices.qubit.preprocess"""

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.PauliX(0), True),
            (qml.CRX(0.1, wires=[0, 1]), True),
            (qml.Snapshot(), False),
            (qml.Barrier(), False),
            (qml.TShift(1), False),
        ],
    )
    def test_stopping_condition(self, op, expected):
        """Test that _stopping_condition works correctly"""
        res = _stopping_condition(op)
        assert res == expected

    @pytest.mark.parametrize(
        "obs, expected",
        [("Hamiltonian", True), (qml.Identity, True), ("QubitUnitary", False), (qml.RX, False)],
    )
    def test_supports_observable(self, obs, expected):
        """Test that _supports_observable works correctly"""
        res = _supports_observable(obs)
        assert res == expected

    def test_check_validity_invalid_op(self):
        """Test that check_validity throws an error when an operation is invalid."""
        tape = QuantumScript(ops=[qml.TShift(0)], measurements=[qml.expval(qml.Hadamard(0))])
        with pytest.raises(DeviceError, match="Gate TShift not supported on Python Device"):
            check_validity(tape)

    def test_check_validity_invalid_observable(self):
        """Test that check_validity throws an error when an observable is invalid."""
        tape = QuantumScript(
            ops=[qml.PauliX(0)], measurements=[qml.expval(qml.GellMann(wires=0, index=1))]
        )
        with pytest.raises(DeviceError, match="Observable GellMann not supported on Python Device"):
            check_validity(tape)

    def test_check_validity_invalid_tensor_observable(self):
        """Test that check_validity throws an error when a tensor includes invalid obserables"""
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.PauliY(1)],
            measurements=[
                qml.expval(qml.GellMann(wires=0, index=1) @ qml.GellMann(wires=1, index=2))
            ],
        )
        with pytest.raises(DeviceError, match="Observable GellMann not supported on Python Device"):
            check_validity(tape)

    def test_check_validity_passes(self):
        """Test that check_validity doesn't throw any errors for a valid circuit"""
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.RZ(0.123, wires=0)], measurements=[qml.state()]
        )
        check_validity(tape)

    def test_expand_fn_expand_unsupported_op(self):
        """Test that expand_fn expands the tape when unsupported operators are present"""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RZ(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)

        expanded_tape = expand_fn(tape)
        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        for op, exp in zip(expanded_tape.circuit, expected + measurements):
            assert qml.equal(op, exp)

    def test_expand_fn_defer_measurement(self):
        """Test that expand_fn defers mid-circuit measurements."""
        mv = MeasurementValue(["test_id"], processing_fn=lambda v: v)
        ops = [
            qml.Hadamard(0),
            MidMeasureMP(wires=[0], id="test_id"),
            qml.transforms.Conditional(mv, qml.RX(0.123, wires=1)),
        ]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)

        expanded_tape = expand_fn(tape)
        expected = [qml.Hadamard(0), qml.ops.Controlled(qml.RX(0.123, wires=1), 0)]

        for op, exp in zip(expanded_tape, expected + measurements):
            assert qml.equal(op, exp)

    def test_expand_fn_no_expansion(self):
        """Test that expand_fn does nothing to a fully supported quantum script."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RZ(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)
        expanded_tape = expand_fn(tape)

        for op, exp in zip(expanded_tape.circuit, ops + measurements):
            assert qml.equal(op, exp)

    def test_batch_transform_no_batching(self):
        """Test that batch_transform does nothing when no batching is required."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)

        tapes, batch_fn = batch_transform(tape)

        assert len(tapes) == 1
        for op, expected in zip(tapes[0].circuit, ops + measurements):
            assert qml.equal(op, expected)

        # Replace with Python Device once added
        dev = qml.device("default.qubit", wires=2)
        expected = dev.execute(tape)
        res = batch_fn(dev.batch_execute(tapes))
        assert np.allclose(res, expected)

    def test_batch_transform_expval_sum(self):
        """Test that batch_transform creates a batch of tapes when a Sum expectation value
        is measured."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi, wires=1)]
        obs = qml.sum(qml.s_prod(0.5, qml.PauliZ(0)), qml.prod(qml.PauliX(0), qml.PauliZ(1)))
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(obs)]
        tape = QuantumScript(ops=ops, measurements=measurements)

        tapes, batch_fn = batch_transform(tape)

        assert len(tapes) == 2
        for t in tapes:
            for op, expected_op in zip(t.operations, ops):
                assert qml.equal(op, expected_op)
        assert len(tapes[0].measurements) == 1 == len(tapes[0].measurements)
        assert qml.equal(tapes[0].measurements[0], qml.expval(qml.PauliZ(0)))
        assert qml.equal(
            tapes[1].measurements[0], qml.expval(qml.prod(qml.PauliX(0), qml.PauliZ(1)))
        )

        # Replace with Python Device once added
        dev = qml.device("default.qubit", wires=2)
        expected = dev.execute(tape)
        res = batch_fn(dev.batch_execute(tapes))
        assert np.allclose(res, expected)

    def test_batch_transform_broadcast(self):
        """Test that batch_transform splits broadcasted tapes correctly."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)

        tapes, batch_fn = batch_transform(tape)
        expected_ops = [
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi, wires=1)],
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi / 2, wires=1)],
        ]

        assert len(tapes) == 2
        for i, t in enumerate(tapes):
            for op, expected in zip(t.circuit, expected_ops[i] + measurements):
                assert qml.equal(op, expected)

        # Replace with Python Device once added
        dev = qml.device("default.qubit", wires=2)
        expected = dev.execute(tape)
        res = batch_fn(dev.batch_execute(tapes))
        assert np.allclose(res, expected)

    def test_batch_transform_expval_sum_broadcast(self):
        """Test that batch_transform splits broadcasted tapes with Sum expectation values
        correctly"""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        obs = qml.sum(qml.s_prod(0.5, qml.PauliZ(0)), qml.prod(qml.PauliX(0), qml.PauliZ(1)))
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(obs)]
        tape = QuantumScript(ops=ops, measurements=measurements)

        tapes, batch_fn = batch_transform(tape)
        expected_ops = [
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi, wires=1)],
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi / 2, wires=1)],
        ]

        assert len(tapes) == 4
        for i, t in enumerate(tapes):
            for op, expected_op in zip(t.operations, expected_ops[i % 2]):
                assert qml.equal(op, expected_op)
            assert len(t.measurements) == 1
            if i < 2:
                assert qml.equal(t.measurements[0], qml.expval(qml.PauliZ(0)))
            else:
                assert qml.equal(
                    t.measurements[0], qml.expval(qml.prod(qml.PauliX(0), qml.PauliZ(1)))
                )

        # Replace with Python Device once added
        dev = qml.device("default.qubit", wires=2)
        expected = dev.execute(tape)
        res = batch_fn(dev.batch_execute(tapes))
        assert np.allclose(res, expected)


class TestPreprocess:
    """Unit tests for ``qml.devices.qubit.preprocess``."""

    def test_preprocess_finite_shots_error(self):
        """Test that preprocess throws an error if finite shots are present."""
        config = qml.devices.ExecutionConfig(shots=100)
        tape = QuantumScript(ops=[], measurements=[])
        with pytest.raises(
            qml.DeviceError, match="The Python Device does not support finite shots."
        ):
            _ = preprocess([tape], execution_config=config)

    def test_preprocess_batch_transform(self):
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            QuantumScript(ops=ops, measurements=[measurements[0]]),
            QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        res_tapes, batch_fn = preprocess(tapes)
        expected_ops = [
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi, wires=1)],
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi / 2, wires=1)],
        ]

        assert len(res_tapes) == 4
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, expected_ops[i % 2]):
                assert qml.equal(op, expected_op)
            assert len(t.measurements) == 1
            if i < 2:
                assert qml.equal(t.measurements[0], measurements[0])
            else:
                assert qml.equal(t.measurements[0], measurements[1])

        # Replace with Python Device once added
        dev = qml.device("default.qubit", wires=2)
        expected = qml.execute(tapes, dev)
        res = batch_fn(dev.batch_execute(res_tapes))
        assert np.allclose(res, expected)

    def test_preprocess_expand(self):
        """Test that preprocess returns the correct tapes when expansion is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.PauliZ(0))], [qml.expval(qml.PauliX(1))]]
        tapes = [
            QuantumScript(ops=ops, measurements=measurements[0]),
            QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        res_tapes, batch_fn = preprocess(tapes)
        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, exp in zip(t.circuit, expected + measurements[i]):
                assert qml.equal(op, exp)

        # Replace with Python Device once added
        dev = qml.device("default.qubit", wires=2)
        expected = qml.execute(tapes, dev)
        res = batch_fn(dev.batch_execute(res_tapes))
        assert np.allclose(res, expected)

    def test_preprocess_split_and_expand(self):
        """Test that preprocess returns the correct tapes when splitting and expanding
        is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            QuantumScript(ops=ops, measurements=[measurements[0]]),
            QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        res_tapes, batch_fn = preprocess(tapes)
        expected_ops = [
            [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RX(np.pi, wires=1)],
            [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RX(np.pi / 2, wires=1)],
        ]

        assert len(res_tapes) == 4
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, expected_ops[i % 2]):
                assert qml.equal(op, expected_op)
            assert len(t.measurements) == 1
            if i < 2:
                assert qml.equal(t.measurements[0], measurements[0])
            else:
                assert qml.equal(t.measurements[0], measurements[1])

        # Replace with Python Device once added
        dev = qml.device("default.qubit", wires=2)
        expected = qml.execute(tapes, dev)
        res = batch_fn(dev.batch_execute(res_tapes))
        # Need to check each row individually because the dimensions of ``res`` and
        # ``expected`` are different
        for r, e in zip(res, expected):
            assert np.allclose(r, e)

    def test_preprocess_check_validity_fail(self):
        """Test that preprocess throws an error if the split and expanded tapes have
        unsupported operators."""
        ops = [qml.Hadamard(0), NoMatNoDecompOp(1), qml.RZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.PauliZ(0))], [qml.expval(qml.PauliX(1))]]
        tapes = [
            QuantumScript(ops=ops, measurements=measurements[0]),
            QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        with pytest.raises(qml.DeviceError, match="Gate NoMatNoDecompOp not supported"):
            _ = preprocess(tapes)
