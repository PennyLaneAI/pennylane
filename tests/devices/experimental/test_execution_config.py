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

import pytest

from pennylane.devices import ExecutionConfig


def test_default_values():
    """Tests that the default values are as expected."""
    config = ExecutionConfig()
    assert config.derivative_order == 1
    assert config.device_options == {}
    assert config.interface is None
    assert config.gradient_method is None
    assert config.gradient_keyword_arguments == {}
    assert config.grad_on_execution is None
    assert config.use_device_gradient is None


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


@pytest.mark.parametrize("option", (True, False, None))
def test_valid_grad_on_execution(option):
    """Test execution config allows True, False and None"""
    config = ExecutionConfig(grad_on_execution=option)
    assert config.grad_on_execution == option


def test_invalid_grad_on_execution():
    """Test invalid values for grad on execution raise an error."""
    with pytest.raises(ValueError, match=r"grad_on_execution must be True, False,"):
        ExecutionConfig(grad_on_execution="forward")
