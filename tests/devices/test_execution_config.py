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


class TestExecutionConfig:
    """Tests for the ExecutionConfig class."""

    def test_default_values(self):
        """Tests that the default values are as expected."""
        config = ExecutionConfig()
        assert config.grad_on_execution is None
        assert config.use_device_gradient is None
        assert config.use_device_jacobian_product is None
        assert config.gradient_method is None
        assert config.gradient_keyword_arguments == {}
        assert config.gradient_keyword_arguments is not ExecutionConfig().gradient_keyword_arguments
        assert config.device_options == {}
        assert config.device_options is not ExecutionConfig().device_options
        assert config.interface == Interface.NUMPY
        assert config.derivative_order == 1
        # Default MCMConfig should be initialized with default values
        assert config.mcm_config.mcm_method is None
        assert config.mcm_config.postselect_mode is None

        assert config.convert_to_numpy is True

    def test_mcm_config_with_dict_input(self):
        """Test mcm_config converts dictionary input to MCMConfig instance."""
        mcm_dict = {"mcm_method": "deferred", "postselect_mode": "hw-like"}
        config = ExecutionConfig(mcm_config=mcm_dict)
        assert isinstance(config.mcm_config, MCMConfig)
        assert config.mcm_config.mcm_method == "deferred"
        assert config.mcm_config.postselect_mode == "hw-like"
        assert config.mcm_config is not mcm_dict  # New object should be created

    def test_mcm_config_with_mcm_config_instance(self):
        """Test mcm_config can be assigned an MCMConfig instance."""
        mcm = MCMConfig(mcm_method="one-shot", postselect_mode="fill-shots")
        config = ExecutionConfig(mcm_config=mcm)
        assert config.mcm_config.mcm_method == "one-shot"
        assert config.mcm_config.postselect_mode == "fill-shots"
        assert config.mcm_config is mcm  # Should be the same instance

    def test_invalid_interface(self):
        """Tests that unknown frameworks raise a ValueError."""
        with pytest.raises(ValueError, match="Interface must be one of"):
            _ = ExecutionConfig(interface="nonsense")

    @pytest.mark.parametrize("option", (True, False, None))
    def test_valid_grad_on_execution(self, option):
        """Test execution config allows True, False and None"""
        config = ExecutionConfig(grad_on_execution=option)
        assert config.grad_on_execution == option

    def test_invalid_grad_on_execution(self):
        """Test invalid values for grad on execution raise an error."""
        with pytest.raises(ValueError, match=r"grad_on_execution must be True, False,"):
            ExecutionConfig(grad_on_execution="forward")

    def test_invalid_execution_config_mcm_config(self):
        """Test that an error is raised if mcm_config is set incorrectly"""
        with pytest.raises(ValueError, match="Got invalid type"):
            _ = ExecutionConfig(mcm_config="foo")

    @pytest.mark.parametrize("method", ("parameter-shift", None, param_shift))
    def test_valid_gradient_method(self, method):
        """Test valid gradient_method types."""
        config = ExecutionConfig(gradient_method=method)
        assert config.gradient_method == method

    def test_invalid_gradient_method(self):
        """Test that invalid types for gradient_method raise an error."""
        with pytest.raises(
            ValueError,
            match=r"Differentiation method 123 must be a str, TransformDispatcher, or None",
        ):
            _ = ExecutionConfig(gradient_method=123)

    def test_immutability(self):
        """Test that ExecutionConfig instances are immutable if frozen."""
        config = ExecutionConfig(grad_on_execution=True)
        with pytest.raises(AttributeError, match="cannot assign to field 'grad_on_execution'"):
            config.grad_on_execution = False

        assert config.grad_on_execution is True


class TestMCMConfig:
    """Tests for the MCMConfig class."""

    def test_default_values(self):
        """Tests that the default values are as expected."""
        mcm_config = MCMConfig()
        assert mcm_config.mcm_method is None
        assert mcm_config.postselect_mode is None

    def test_all_fields_set(self):
        """Test that MCMConfig correctly sets all fields when provided since we are
        overwriting __post_init__."""
        config = MCMConfig(mcm_method="deferred", postselect_mode="hw-like")
        assert config.mcm_method == "deferred"
        assert config.postselect_mode == "hw-like"

    @pytest.mark.parametrize(
        "mcm_method",
        [
            None,
            "deferred",
            "one-shot",
            "some_custom_method",  # Any string is valid as per docstring
        ],
    )
    def test_valid_mcm_method(self, mcm_method):
        """Test that MCMConfig can be instantiated with valid mcm_method values."""
        config = MCMConfig(mcm_method=mcm_method)
        assert config.mcm_method == mcm_method

    @pytest.mark.parametrize(
        "postselect_mode",
        [
            None,
            "hw-like",
            "fill-shots",
            "pad-invalid-samples",
        ],
    )
    def test_valid_postselect_mode(self, postselect_mode):
        """Test that MCMConfig can be instantiated with valid postselect_mode values."""
        config = MCMConfig(postselect_mode=postselect_mode)
        assert config.postselect_mode == postselect_mode

    @pytest.mark.parametrize(
        "invalid_mode",
        [
            "foo",
            123,
            True,
        ],
    )
    def test_invalid_postselect_mode_raises_value_error(self, invalid_mode):
        """Test that MCMConfig raises ValueError for invalid postselect_mode."""
        with pytest.raises(ValueError, match=f"Invalid postselection mode '{invalid_mode}'."):
            MCMConfig(postselect_mode=invalid_mode)

    def test_immutability(self):
        """Test that MCMConfig instances are immutable."""
        config = MCMConfig(mcm_method="deferred", postselect_mode="hw-like")

        with pytest.raises(AttributeError, match="cannot assign to field 'mcm_method'"):
            config.mcm_method = "one-shot"

        with pytest.raises(AttributeError, match="cannot assign to field 'postselect_mode'"):
            config.postselect_mode = "fill-shots"

        assert config.mcm_method == "deferred"
        assert config.postselect_mode == "hw-like"

    def test_equality(self):
        """Test equality comparison of MCMConfig instances."""
        config1 = MCMConfig(mcm_method="deferred", postselect_mode="hw-like")
        config2 = MCMConfig(mcm_method="deferred", postselect_mode="hw-like")
        config3 = MCMConfig(mcm_method="one-shot", postselect_mode="hw-like")

        assert config1 == config2
        assert config1 != config3
