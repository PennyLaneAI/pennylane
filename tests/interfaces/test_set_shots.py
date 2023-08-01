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
"""
Tests for interfaces.set_shots
"""

import pytest

import pennylane as qml
from pennylane.interfaces import set_shots
from pennylane.measurements import Shots


def test_shots_new_device_interface():
    """Test that calling set_shots on a device implementing the new interface leaves it
    untouched.
    """
    dev = qml.devices.experimental.DefaultQubit2()
    with pytest.raises(ValueError):
        with set_shots(dev, 10):
            pass


def test_set_with_shots_class():
    """Test that shots can be set on the old device interface with a Shots class."""

    dev = qml.devices.DefaultQubit(wires=1)
    with set_shots(dev, Shots(10)):
        assert dev.shots == 10

    assert dev.shots is None

    shot_tuples = Shots((10, 10))
    with set_shots(dev, shot_tuples):
        assert dev.shots == 20
        assert dev.shot_vector == list(shot_tuples.shot_vector)

    assert dev.shots is None


def test_shots_not_altered_if_False():
    """Test a value of False can be passed to shots, indicating to not override
    shots on the device."""

    dev = qml.devices.DefaultQubit(wires=1)
    with set_shots(dev, False):
        assert dev.shots is None
