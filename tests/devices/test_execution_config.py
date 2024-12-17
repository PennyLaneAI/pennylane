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

from pennylane.devices.execution_config import ExecutionConfig, MCMConfig
from pennylane.gradients import param_shift
from pennylane.math import Interface


def test_default_values():
    """Tests that the default values are as expected."""
    config = ExecutionConfig()
    assert config.derivative_order == 1
    assert config.device_options == {}
    assert config.interface == Interface.NUMPY
    assert config.gradient_method is None
    assert config.gradient_keyword_arguments == {}
    assert config.grad_on_execution is None
    assert config.use_device_gradient is None
    assert config.mcm_config == MCMConfig()


def test_mcm_config_default_values():
    """Test that the default values of MCMConfig are correct"""
    mcm_config = MCMConfig()
    assert mcm_config.postselect_mode is None
    assert mcm_config.mcm_method is None


def test_invalid_interface():
    """Tests that unknown frameworks raise a ValueError."""
    with pytest.raises(ValueError, match="Interface must be one of"):
        _ = ExecutionConfig(interface="nonsense")


@pytest.mark.parametrize("option", (True, False, None))
def test_valid_grad_on_execution(option):
    """Test execution config allows True, False and None"""
    config = ExecutionConfig(grad_on_execution=option)
    assert config.grad_on_execution == option


def test_invalid_grad_on_execution():
    """Test invalid values for grad on execution raise an error."""
    with pytest.raises(ValueError, match=r"grad_on_execution must be True, False,"):
        ExecutionConfig(grad_on_execution="forward")


@pytest.mark.parametrize(
    "option", [MCMConfig(mcm_method="deferred"), {"mcm_method": "deferred"}, None]
)
def test_valid_execution_config_mcm_config(option):
    """Test that the mcm_config attribute is set correctly"""
    config = ExecutionConfig(mcm_config=option) if option else ExecutionConfig()
    if option is None:
        assert config.mcm_config == MCMConfig()
    else:
        assert config.mcm_config == MCMConfig(mcm_method="deferred")


def test_invalid_execution_config_mcm_config():
    """Test that an error is raised if mcm_config is set incorrectly"""
    option = "foo"
    with pytest.raises(ValueError, match="Got invalid type"):
        _ = ExecutionConfig(mcm_config=option)


def test_mcm_config_invalid_postselect_mode():
    """Test that an error is raised if creating MCMConfig with invalid postselect_mode"""
    option = "foo"
    with pytest.raises(ValueError, match="Invalid postselection mode"):
        _ = MCMConfig(postselect_mode=option)


@pytest.mark.parametrize("method", ("parameter-shift", None, param_shift))
def test_valid_gradient_method(method):
    """Test valid gradient_method types."""
    config = ExecutionConfig(gradient_method=method)
    assert config.gradient_method == method


def test_invalid_gradient_method():
    """Test that invalid types for gradient_method raise an error."""
    with pytest.raises(
        ValueError, match=r"Differentiation method 123 must be a str, TransformDispatcher, or None"
    ):
        ExecutionConfig(gradient_method=123)
