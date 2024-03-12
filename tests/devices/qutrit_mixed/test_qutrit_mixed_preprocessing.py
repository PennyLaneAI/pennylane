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
from pennylane.devices import DefaultQubit, ExecutionConfig

from pennylane.devices.default_qubit import stopping_condition


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


def test_snapshot_multiprocessing_execute():
    """DefaultQubit cannot execute tapes with Snapshot if `max_workers` is not `None`"""
    dev = qml.device("default.qubit", max_workers=2)

    tape = qml.tape.QuantumScript(
        [
            qml.Snapshot(),
            qml.THadamard(wires=0),
            qml.Snapshot("very_important_state"),
            qml.TAdd(wires=[0, 1]),
            qml.Snapshot(),
        ],
        [qml.expval(qml.GellMann(0, 1))],
    )
    with pytest.raises(RuntimeError, match="ProcessPoolExecutor cannot execute a QuantumScript"):
        program, _ = dev.preprocess()
        program([tape])


class TestConfigSetup:
    """Tests involving setting up the execution config."""
    pass


# pylint: disable=too-few-public-methods
class TestPreprocessing:
    """Unit tests for the preprocessing method."""
    pass


class TestPreprocessingIntegration:
    """Test preprocess produces output that can be executed by the device."""
    pass
