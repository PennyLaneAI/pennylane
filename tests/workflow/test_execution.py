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
Unit tests for the :mod:`qml.workflow <pennylane.workflow>` module.
"""

import pytest

from pennylane.workflow import ExecutionConfig


class TestExecutionConfig:
    """Tests the ExecutionConfig class."""

    def test_default_values(self):
        """Tests that the default values are as expected."""
        config = ExecutionConfig()
        assert config.derivative_order == 1
        assert config.device_options == {}
        assert config.interface == "jax"
        assert config.gradient_method is None
        assert config.gradient_keyword_arguments == {}
        assert config.shots is None

    def test_invalid_interface(self):
        """Tests that unknown frameworks raise a ValueError."""
        with pytest.raises(ValueError, match="interface must be in"):
            _ = ExecutionConfig(interface="nonsense")

    def test_invalid_gradient_method(self):
        """Tests that unknown gradient_methods raise a ValueError."""
        with pytest.raises(ValueError, match="gradient_method must be in"):
            _ = ExecutionConfig(gradient_method="nonsense")

    def test_invalid_gradient_keyword_arguments(self):
        """Tests that unknown gradient_keyword_arguments raise a ValueError."""
        with pytest.raises(ValueError, match="All gradient_keyword_arguments keys must be in"):
            _ = ExecutionConfig(gradient_keyword_arguments={"nonsense": 0})
