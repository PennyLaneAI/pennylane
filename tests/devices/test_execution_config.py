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
Unit tests for the :class:`~pennylane.devices.ExecutionConfig` class.
"""

# pylint: disable=protected-access

from dataclasses import replace
import pytest

from pennylane.devices import ExecutionConfig, DeviceError


def test_default_values():
    """Tests that the default values are as expected."""
    config = ExecutionConfig()
    assert config.derivative_order == 1
    assert config.device_options == {}
    assert config.interface == "autograd"
    assert config.gradient_method is None
    assert config.gradient_keyword_arguments == {}
    assert config.shots is None


def test_invalid_interface():
    """Tests that unknown frameworks raise a ValueError."""
    with pytest.raises(ValueError, match="interface must be in"):
        _ = ExecutionConfig(interface="nonsense")


def test_invalid_gradient_method():
    """Tests that unknown gradient_methods raise a ValueError."""
    with pytest.raises(ValueError, match="gradient_method must be in"):
        _ = ExecutionConfig(gradient_method="nonsense")


def test_invalid_gradient_keyword_arguments():
    """Tests that unknown gradient_keyword_arguments raise a ValueError."""
    with pytest.raises(ValueError, match="All gradient_keyword_arguments keys must be in"):
        _ = ExecutionConfig(gradient_keyword_arguments={"nonsense": 0})


def test_shots():
    """Tests that shots are initialized correctly"""
    shotlist = [1, 3, 3, 4, 4, 4, 3]
    exec_conf = ExecutionConfig(shots=shotlist)
    shot_vector = exec_conf.shot_vector

    assert len(shot_vector) == 4
    assert shot_vector[0].shots == 1
    assert shot_vector[0].copies == 1
    assert shot_vector[1].shots == 3
    assert shot_vector[1].copies == 2
    assert shot_vector[2].shots == 4
    assert shot_vector[2].copies == 3
    assert shot_vector[3].shots == 3
    assert shot_vector[3].copies == 1
    assert exec_conf._raw_shot_sequence == shotlist
    assert exec_conf.shots == 22

    exec_conf = replace(exec_conf, shots=3)
    assert exec_conf.shot_vector is None
    assert exec_conf._raw_shot_sequence is None
    assert exec_conf.shots == 3

    exec_conf = replace(exec_conf, shots=None)
    assert exec_conf.shot_vector is None
    assert exec_conf._raw_shot_sequence is None
    assert exec_conf.shots is None


def test_invalid_shots():
    """Test that correct errors are raised when invalid shots are specified"""
    with pytest.raises(DeviceError, match="The specified number of shots needs to be at least 1"):
        _ = ExecutionConfig(shots=0)

    with pytest.raises(
        DeviceError, match="Shots must be a single non-negative integer or a sequence"
    ):
        _ = ExecutionConfig(shots="Hello world")
