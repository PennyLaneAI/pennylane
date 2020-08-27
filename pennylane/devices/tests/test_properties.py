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
# pylint: disable=no-self-use
import pytest
import numpy as np
import pennylane as qml
from pennylane._device import DeviceError


class TestDeviceProperties:
    """Test the device is created with the expected properties."""

    def test_load_device(self, device_kwargs):
        """Test that the device loads correctly."""
        device_kwargs["wires"] = 2
        device_kwargs["shots"] = 1234

        dev = qml.device(**device_kwargs)
        assert dev.num_wires == 2
        assert dev.shots == 1234
        assert dev.short_name == device_kwargs["name"]
        assert hasattr(dev, "analytic")

    def test_no_wires_given(self, device_kwargs):
        """Test that the device requires correct arguments."""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device(**device_kwargs)

    def test_no_0_shots(self, device_kwargs):
        """Test that non-analytic devices cannot accept 0 shots."""
        # first create a valid device to extract its capabilities
        device_kwargs["wires"] = 2
        device_kwargs["shots"] = 0

        with pytest.raises(DeviceError, match="The specified number of shots needs to be"):
            qml.device(**device_kwargs)


class TestCapabilities:
    """Test that the device has a valid capabilities dictionary."""

    def test_has_capabilities_dictionary(self, device_kwargs):
        """Test that the device class has a capabilities() method returning a dictionary."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        cap = dev.capabilities()
        assert isinstance(cap, dict)

    def test_model_is_defined_and_valid(self, device_kwargs):
        """Test that the capabilities dictionary defines a valid model."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        cap = dev.capabilities()
        assert "model" in cap
        assert cap["model"] in ["qubit", "cv"]

    def test_passthru_is_valid(self, device_kwargs):
        """Test that the capabilities dictionary defines a valid passthru interface, if not None."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        cap = dev.capabilities()
        passthru_interf = cap.get("passthru_interface", None)
        assert passthru_interf in [None, "tf", "autograd", "numpy", "torch"]

    def test_supports_sampled_mode(self, device_kwargs):
        """Test that the device's "analytic" attribute can be set to false if it claims to support sampled mode."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        cap = dev.capabilities()
        supports_sampled = cap.get("supports_sampled", False)

        if not supports_sampled:
            pytest.skip("Device does not support sampled mode.")

        else:
            device_kwargs["analytic"] = False
            dev_sampled = qml.device(**device_kwargs)
            assert not dev_sampled.analytic

    def test_supports_exact_mode(self, device_kwargs):
        """Test that the device's "analytic" attribute can be set to true if it claims to support exact mode."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        cap = dev.capabilities()
        supports_exact = cap.get("supports_exact", False)

        if not supports_exact:
            pytest.skip("Device does not support exact mode.")

        else:
            device_kwargs["analytic"] = True
            dev_exact = qml.device(**device_kwargs)
            assert dev_exact.analytic

    def test_provides_jacobian(self, device_kwargs):
        """Test that the device computes the jacobian."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        cap = dev.capabilities()
        provides_jacobian = cap.get("provides_jacobian", False)

        if not provides_jacobian:
            pytest.skip("Device does not provide jacobian.")

        else:

            @qml.qnode(dev)
            def circuit():
                return qml.expval(qml.Identity(wires=0))

            assert hasattr(circuit, "jacobian")

    def test_reversible_diff(self, device_kwargs):
        """Test that the device supports reversible differentiation."""
        device_kwargs["wires"] = 1
        dev = qml.device(**device_kwargs)
        cap = dev.capabilities()
        rev_diff = cap.get("supports_reversible_diff", False)

        if not rev_diff:
            pytest.skip("Device does not support reversible differentiation.")

        else:

            @qml.qnode(dev, diff_method="reversible")
            def circuit():
                return qml.expval(qml.Identity(wires=0))

            assert isinstance(circuit(), (float, np.ndarray))
