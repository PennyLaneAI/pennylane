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

    def test_load_device(self, device_kwargs):
        """Test that the QVM device loads correctly."""
        device_kwargs["wires"] = 2
        device_kwargs["shots"] = 1234

        dev = qml.device(**device_kwargs)
        assert dev.num_wires == 2
        assert dev.shots == 1234
        assert dev.short_name == device_kwargs["name"]

    def test_no_wires_given(self, device_kwargs):
        """Test that the device requires correct arguments."""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device(**device_kwargs)

    def test_no_0_shots(self, device_kwargs):
        """Test that non-analytic devices cannot accept 0 shots."""
        # first create a valid device to extract its capabilities
        device_kwargs["wires"] = 2
        dev = qml.device(**device_kwargs)

        device_kwargs["shots"] = 0

        with pytest.raises(DeviceError, match="The specified number of shots needs to be"):
            qml.device(**device_kwargs)
