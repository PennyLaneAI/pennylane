# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lightning qubit executing jaxpr."""

import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig, MCMConfig

jax = pytest.importorskip("jax")
pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class TestPreprocess:
    """Unit tests for lightning.qubit preprocessing with program capture."""

    def test_execution_config_invalid_device_option(self):
        """Test that specifying an invalid device option raises an error."""
        dev = qml.device("lightning.qubit", wires=1)
        config = ExecutionConfig(device_options={"foo": "bar"})

        with pytest.raises(qml.DeviceError, match="device option foo not present"):
            _ = dev.preprocess(execution_config=config)

    def test_execution_config_best_gradient_method(self):
        """Test that specifying diff_method="best" changes to "adjoint"."""
        dev = qml.device("lightning.qubit", wires=1)
        config = ExecutionConfig(gradient_method="best")

        _, new_config = dev.preprocess(execution_config=config)
        assert new_config.gradient_method == "adjoint"

    def test_execution_config_adjoint_gradient_method(self):
        """Test that specifying diff_method="adjoint" updates the execution config correctly."""
        dev = qml.device("lightning.qubit", wires=1)
        config = ExecutionConfig(gradient_method="adjoint")

        _, new_config = dev.preprocess(execution_config=config)
        assert new_config.gradient_method == "adjoint"
        assert new_config.use_device_jacobian_product
        assert new_config.grad_on_execution

    def test_execution_config_invalid_mcm_method_error(self):
        """Test that an error is raised if mcm_method is invalid."""
        dev = qml.device("lightning.qubit", wires=1)
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="foo"))

        with pytest.raises(qml.DeviceError, match="mcm_method='foo' is not supported"):
            _ = dev.preprocess(execution_config=config)

    def test_execution_config_default_mcm_config(self):
        """Test that not specifying MCM config updates the execution config correctly."""
        dev = qml.device("lightning.qubit", wires=1)
        config = ExecutionConfig()

        _, new_config = dev.preprocess(execution_config=config)
        assert new_config.mcm_config == MCMConfig(mcm_method="deferred", postselect_mode=None)

    @pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
    def test_single_branch_statistics_postselect_mode_warning(self, postselect_mode):
        """Test that setting a postselect_mode with single-branch-statistics raises a warning."""
        dev = qml.device("lightning.qubit", wires=1)
        config = ExecutionConfig(
            mcm_config=MCMConfig(
                mcm_method="single-branch-statistics", postselect_mode=postselect_mode
            )
        )

        with pytest.warns(UserWarning, match="Setting 'postselect_mode' is not supported"):
            _, new_config = dev.preprocess(execution_config=config)

        assert new_config.mcm_config == MCMConfig(
            mcm_method="single-branch-statistics", postselect_mode=None
        )

    def test_transform_program(self):
        """Test that the transform program returned by preprocess has the correct transforms."""
        dev = qml.device("lightning.qubit", wires=1)

        # Default config
        config = ExecutionConfig()
        program, _ = dev.preprocess(execution_config=config)
        assert len(program) == 2
        # pylint: disable=protected-access
        assert program[0].transform == qml.defer_measurements._transform
        assert program[1].transform == qml.transforms.decompose._transform

        # mcm_method="deferred"
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="deferred"))
        program, _ = dev.preprocess(execution_config=config)
        assert len(program) == 2
        # pylint: disable=protected-access
        assert program[0].transform == qml.defer_measurements._transform
        assert program[1].transform == qml.transforms.decompose._transform

        # mcm_method="single-branch-statistics"
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="single-branch-statistics"))
        program, _ = dev.preprocess(execution_config=config)
        assert len(program) == 1
        # pylint: disable=protected-access
        assert program[0].transform == qml.transforms.decompose._transform
