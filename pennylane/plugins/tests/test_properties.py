# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests that a device has the right attributes, arguments and methods."""
import pennylane as qml
import pytest
from pennylane._device import DeviceError


class TestDeviceProperties:
    """Test the device is created with the expected properties."""

    def test_load_device(self, device_name):
        """Test that the QVM device loads correctly."""
        dev = qml.device(device_name, wires=2, shots=424)
        assert dev.num_wires == 2
        assert dev.shots == 424
        assert dev.short_name == device_name

    def test_not_enough_args(self, device_name):
        """Test that the device requires correct arguments."""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device(device_name)

    def test_nonanalytic_no_0_shots(self, device_name, skip_if):
        """Test that non-analytic devices cannot accept 0 shots."""
        # first create a valid device to extract its capabilities
        dev = qml.device(device_name, wires=2)
        skip_if(not dev.analytic)

        with pytest.raises(DeviceError, match="The specified number of shots needs to be"):
            qml.device(device_name, wires=2, shots=0)

    def test_analytic_0_shots(self, device_name, skip_if):
        """Test that analytic devices can accept 0 shots."""
        # first create a valid device to extract its capabilities
        dev = qml.device(device_name, wires=1)
        skip_if(dev.analytic)

        # a state simulator will allow shots=0
        dev = qml.device(device_name, wires=1, shots=0)
        assert dev.shots == 0
