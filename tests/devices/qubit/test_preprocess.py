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

# pylint: disable=unused-import
import pytest

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
from pennylane.measurements import ExpectationMP, ProbabilityMP
from pennylane.tape import QuantumScript
from pennylane import DeviceError


# pylint: disable=missing-docstring
class NoMatOp(Operation):
    num_wires = 1

    def has_matrix(self):
        return False

    def decomposition(self):
        return [qml.PauliX(self.wires), qml.PauliY(self.wires)]


class TestPreprocess:
    """Unit tests for functions in qml.devices.qubit.preprocess"""

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
            ops=[qml.PauliX(0), qml.RZ(0.123, wires=0)], measurements=[qml.expval(qml.Hadamard(0))]
        )
        check_validity(tape)

    def test_batch_transform(self):
        """Test that batch_transform works correctly"""

    def test_expand_fn_expand_unsupported_op(self):
        """Test that expand_fn expands the tape when unsupported operators are present"""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RZ(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)

        expanded_tape = expand_fn(tape)
        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        for expanded_op, expected_op in zip(expanded_tape.operations, expected):
            assert qml.equal(expanded_op, expected_op)

        mps = [ExpectationMP, ProbabilityMP]
        for meas, exp_meas, mp in zip(expanded_tape.measurements, measurements, mps):
            assert qml.equal(meas.obs, exp_meas.obs)
            assert isinstance(meas, mp)

    def test_expand_fn_defer_measurement(self):
        """Test that expand_fn defers mid-circuit measurements."""
        # ops = [qml.Hadamard(0), qml.measure(1)]

    def test_preprocess(self):
        """Test that preprocess works correctly"""
