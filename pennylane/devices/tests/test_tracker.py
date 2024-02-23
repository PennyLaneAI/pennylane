# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests that the device uses a tracker properly if it is supported."""
# pylint: disable=no-self-use,no-member
import pytest

import pennylane as qml


class TestTracker:
    """Tests that the device uses a tracker attribute properly"""

    def test_tracker_initialization(self, device):
        """Tests a tracker instance is assigned at initialization."""

        dev = device(1)

        if isinstance(dev, qml.Device) and not dev.capabilities().get("supports_tracker", False):
            pytest.skip("Device does not support a tracker")

        assert isinstance(dev.tracker, qml.Tracker)

    def test_tracker_updated_in_execution_mode(self, device):
        """Tests that device update and records during tracking mode"""

        dev = device(1)

        if isinstance(dev, qml.Device) and not dev.capabilities().get("supports_tracker", False):
            pytest.skip("Device does not support a tracker")

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ():
            return qml.expval(qml.X(0))

        dev.tracker.active = False
        with dev.tracker:
            circ()
        assert dev.tracker.history["batches"] == [1]
        assert dev.tracker.history["executions"] == [1]
