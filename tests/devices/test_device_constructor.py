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
"""Unit tests for the qp.device constructor."""

from importlib import metadata, reload

import pytest

import pennylane as qp
from pennylane.exceptions import DeviceError


class TestDeviceInit:
    """Tests for device loader in __init__.py"""

    def test_no_config(self):
        """Test that an error is raised if config is passed as a kwarg."""

        with pytest.raises(ValueError, match="config has been removed"):
            qp.device("default.qubit", config=None)

    def test_no_device(self):
        """Test that an exception is raised for a device that doesn't exist"""

        with pytest.raises(DeviceError, match="Device None does not exist"):
            qp.device("None", wires=0)

    def test_outdated_API(self, monkeypatch):
        """Test that an exception is raised if plugin that targets an old API is loaded"""

        with monkeypatch.context() as m:
            m.setattr(qp.devices.device_constructor, "__version__", "0.0.1")
            with pytest.raises(DeviceError, match="plugin requires PennyLane versions"):
                qp.device("default.qutrit", wires=0)

    def test_plugin_devices_from_devices_triggers_getattr(self, mocker):
        spied = mocker.spy(qp.devices, "__getattr__")

        _ = qp.devices.plugin_devices

        spied.assert_called_once()

    def test_refresh_entrypoints(self, monkeypatch):
        """Test that new entrypoints are found by the refresh_devices function"""
        assert qp.plugin_devices

        with monkeypatch.context() as m:
            # remove all entry points
            retval = []
            m.setattr(metadata, "entry_points", lambda **kwargs: retval)

            # reimporting PennyLane within the context sets qp.plugin_devices to {}
            reload(qp)
            reload(qp.devices.device_constructor)

            # since there are no entry points, there will be no plugin devices
            assert not qp.plugin_devices

        # outside of the context, entrypoints will now be found
        assert not qp.plugin_devices
        qp.refresh_devices()
        assert qp.plugin_devices

        # Test teardown: re-import PennyLane to revert all changes and
        # restore the plugin_device dictionary
        reload(qp)
        reload(qp.devices.device_constructor)

    def test_hot_refresh_entrypoints(self, monkeypatch):
        """Test that new entrypoints are found by the device loader if not currently present"""
        assert qp.plugin_devices

        with monkeypatch.context() as m:
            # remove all entry points
            retval = []
            m.setattr(metadata, "entry_points", lambda **kwargs: retval)

            # reimporting PennyLane within the context sets qp.plugin_devices to {}
            reload(qp.devices)
            reload(qp.devices.device_constructor)

            m.setattr(qp.devices.device_constructor, "refresh_devices", lambda: None)
            assert not qp.plugin_devices

            # since there are no entry points, there will be no plugin devices
            with pytest.raises(DeviceError, match="Device default.qubit does not exist"):
                qp.device("default.qubit", wires=0)

        # outside of the context, entrypoints will now be found automatically
        assert not qp.plugin_devices
        dev = qp.device("default.qubit", wires=0)
        assert qp.plugin_devices
        assert dev.name == "default.qubit"

        # Test teardown: re-import PennyLane to revert all changes and
        # restore the plugin_device dictionary
        reload(qp)
        reload(qp.devices.device_constructor)
