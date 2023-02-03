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
from pennylane.devices.qubit.preprocess import (
    _stopping_condition,
    _supports_observable,
    expand_fn,
    check_validity,
    batch_transform,
    preprocess,
)
from pennylane.tape import QuantumScript
from pennylane import DeviceError


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
        dev = qml.device("default.qubit", wires=2)
        res = _supports_observable(dev, obs)
        assert res == expected

    @pytest.mark.parametrize(
        "tape, err_msg",
        [
            (
                QuantumScript([qml.measurements.MidMeasureMP(wires=qml.wires.Wires([0]))]),
                "Mid-circuit measurements are not natively supported on device default.qubit.",
            ),
            (QuantumScript(ops=[qml.TShift(0)]), "Gate TShift not supported on device"),
            (
                QuantumScript(
                    measurements=[
                        qml.expval(qml.GellMann(wires=0, index=1) @ qml.GellMann(wires=1, index=2))
                    ]
                ),
                "Observable GellMann not supported on device",
            ),
            (
                QuantumScript(measurements=[qml.expval(qml.GellMann(wires=0, index=1))]),
                "Observable GellMann not supported on device",
            ),
        ],
    )
    def test_check_validity_fails(self, tape, err_msg):
        """Test that check_validity throws the appropriate error when expected"""
        dev = qml.device("default.qubit", wires=2)
        with pytest.raises(DeviceError, match=err_msg):
            check_validity(dev, tape)

    def test_check_validity_passes(self):
        """Test that check_validity doesn't throw any errors for a valid circuit"""
        dev = qml.device("default.qubit", wires=1)
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.RZ(0.123, wires=0)], measurements=[qml.expval(qml.Hadamard(0))]
        )
        check_validity(dev, tape)

    def test_batch_transform(self):
        """Test that batch_transform works correctly"""

    def test_expand_fn(self):
        """Test that expand_fn works correctly"""

    def test_preprocess(self):
        """Test that preprocess works correctly"""
