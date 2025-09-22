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

from copy import copy, deepcopy
from dataclasses import replace

import pytest

from pennylane.devices.execution_config import ExecutionConfig, FrozenMapping, MCMConfig
from pennylane.gradients import param_shift
from pennylane.math import Interface


def test_default_execution_config_deprecation():
    """Test that the DefaultExecutionConfig is deprecated."""
    with pytest.warns(
        UserWarning,
        match="`pennylane.devices.DefaultExecutionConfig` is deprecated and will be removed in v0.44.",
    ):
        from pennylane.devices import (  # pylint: disable=unused-import, import-outside-toplevel, no-name-in-module
            DefaultExecutionConfig,
        )

    with pytest.warns(
        UserWarning,
        match="`pennylane.devices.execution_config.DefaultExecutionConfig` is deprecated and will be removed in v0.44.",
    ):
        from pennylane.devices.execution_config import (  # pylint: disable=unused-import, import-outside-toplevel, no-name-in-module
            DefaultExecutionConfig,
        )


class TestFrozenMapping:
    """Tests all of the dunders for FrozenMapping."""

    def test_frozen_mapping(self):
        """Test that FrozenMapping is immutable."""
        fm = FrozenMapping(a=1, b=2)
        with pytest.raises(TypeError, match="FrozenMapping is immutable"):
            fm["a"] = 3
        with pytest.raises(TypeError, match="FrozenMapping is immutable"):
            del fm["b"]

    def test_frozen_mapping_methods(self):
        """Test the methods of FrozenMapping."""
        fm = FrozenMapping(a=1, b=2)
        assert list(fm.keys()) == ["a", "b"]
        assert list(fm.values()) == [1, 2]
        assert list(fm.items()) == [("a", 1), ("b", 2)]
        assert fm.get("a") == 1
        assert fm.get("c") is None

        with pytest.raises(TypeError, match="FrozenMapping is immutable"):
            fm["a"] = 3
        with pytest.raises(TypeError, match="FrozenMapping is immutable"):
            del fm["b"]

        assert repr(fm) == "{'a': 1, 'b': 2}"

    def test_copy_frozen_mapping(self):
        """Test that copying a FrozenMapping creates a new instance."""
        fm1 = FrozenMapping(a=1, b=2)
        fm2 = copy(fm1)
        assert fm1 is not fm2
        assert fm1 == fm2

    def test_deepcopy_frozen_mapping(self):
        """Test that deep copying a FrozenMapping creates a new instance."""
        # Set up with mutable value
        fm1 = FrozenMapping(a=1, b=[2, 3])
        fm2 = deepcopy(fm1)
        assert fm1 is not fm2
        assert fm1 == fm2
        # Check that mutable contents are deep-copied
        assert fm1["b"] is not fm2["b"]


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
        with pytest.raises(ValueError, match="'nonsense' is not a valid Interface."):
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

    @pytest.mark.parametrize(
        "invalid_method",
        [
            123,
            lambda grad_fn: True,
            True,
        ],
    )
    def test_invalid_gradient_method(self, invalid_method):
        """Test that invalid types for gradient_method raise an error."""
        with pytest.raises(
            ValueError,
            match=r"Differentiation method .* must be a str, TransformDispatcher, or None",
        ):
            _ = ExecutionConfig(gradient_method=invalid_method)

    @pytest.mark.parametrize(
        "invalid_device_options",
        [
            "hi",
            123,
            lambda grad_fn: True,
            True,
        ],
    )
    def test_invalid_device_options(self, invalid_device_options):
        """Test that invalid types for device_options raise an error."""
        with pytest.raises(
            TypeError,
            match=r"Got invalid type .* for 'device_options'",
        ):
            _ = ExecutionConfig(device_options=invalid_device_options)

    @pytest.mark.parametrize(
        "invalid_gradient_keyword_arguments",
        [
            "hi",
            123,
            lambda grad_fn: True,
            True,
        ],
    )
    def test_invalid_gradient_keyword_arguments(self, invalid_gradient_keyword_arguments):
        """Test that invalid types for gradient_keyword_arguments raise an error."""
        with pytest.raises(
            TypeError,
            match=r"Got invalid type .* for 'gradient_keyword_arguments'",
        ):
            _ = ExecutionConfig(gradient_keyword_arguments=invalid_gradient_keyword_arguments)

    def test_execution_config_replace_interactions(self):
        """
        Tests that `dataclasses.replace` interacts correctly with the
        custom validation and transformation in `__post_init__`.
        """
        original_config = ExecutionConfig(
            device_options={"hi": "bye"}, gradient_keyword_arguments={"foo": "bar"}
        )

        new_config = replace(original_config, device_options={"hi": "there"})

        assert new_config.device_options == {"hi": "there"}

        assert new_config.gradient_keyword_arguments == original_config.gradient_keyword_arguments

        with pytest.raises(TypeError, match=r"Got invalid type .* for 'device_options'"):
            replace(original_config, device_options=["this", "is", "a", "list"])

    def test_immutability_of_fields_with_custom_handling(self):
        """Test fields with custom logic are immutable."""
        config = ExecutionConfig()
        with pytest.raises(AttributeError, match="cannot assign to field 'grad_on_execution'"):
            config.grad_on_execution = False

        with pytest.raises(AttributeError, match="cannot assign to field 'interface'"):
            config.interface = "torch"

        with pytest.raises(AttributeError, match="cannot assign to field 'gradient_method'"):
            config.gradient_method = None

        with pytest.raises(AttributeError, match="cannot assign to field 'mcm_config'"):
            config.mcm_config = {}

        with pytest.raises(AttributeError, match="cannot assign to field 'executor_backend'"):
            config.executor_backend = None

    @pytest.mark.parametrize(
        "config",
        (
            ExecutionConfig(),
            ExecutionConfig(
                device_options={"hi": "bye"}, gradient_keyword_arguments={"foo": "bar"}
            ),
        ),
    )
    def test_dict_immutability(self, config):
        """Test that the device_options and gradient_keyword_arguments are immutable."""

        og_device_options = copy(config.device_options)
        og_gradient_keyword_arguments = copy(config.gradient_keyword_arguments)

        with pytest.raises(
            TypeError,
            match=r".* is immutable",
        ):
            config.device_options["hi"] = "there"

        with pytest.raises(
            TypeError,
            match=r".* is immutable",
        ):
            config.gradient_keyword_arguments["foo"] = "buzz"

        # Verify the original dictionaries were not changed
        assert config.device_options == og_device_options
        assert config.gradient_keyword_arguments == og_gradient_keyword_arguments

    def test_hashability(self):
        """Test that ExecutionConfig instances are hashable."""
        config = ExecutionConfig()
        assert isinstance(hash(config), int)

        config = ExecutionConfig(
            device_options={"option1": 42},
            gradient_keyword_arguments={"arg1": "value"},
            mcm_config={"mcm_method": "deferred", "postselect_mode": "hw-like"},
        )
        assert isinstance(hash(config), int)


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
