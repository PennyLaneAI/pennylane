# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for default qubit executing jaxpr."""

import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig, MCMConfig
from pennylane.exceptions import DeviceError

jax = pytest.importorskip("jax")
pytestmark = [pytest.mark.jax, pytest.mark.capture]


class TestPreprocess:
    """Unit tests for default.qubit preprocessing with program capture."""

    def test_execution_config_invalid_device_option(self):
        """Test that specifying an invalid device option raises an error."""
        dev = qml.device("default.qubit", wires=1)
        config = ExecutionConfig(device_options={"foo": "bar"})

        with pytest.raises(DeviceError, match="device option foo not present"):
            _ = dev.preprocess(execution_config=config)

    def test_execution_config_max_workers(self):
        """Test that specifying max_workers raises an error."""
        dev = qml.device("default.qubit", wires=1)
        config = ExecutionConfig(device_options={"max_workers": "1"})

        with pytest.raises(DeviceError, match="Cannot set 'max_workers'"):
            _ = dev.preprocess(execution_config=config)

    def test_execution_config_best_gradient_method(self):
        """Test that specifying diff_method="best" changes to "backprop"."""
        dev = qml.device("default.qubit", wires=1)
        config = ExecutionConfig(gradient_method="best")

        _, new_config = dev.preprocess(execution_config=config)
        assert new_config.gradient_method == "backprop"

    def test_execution_config_adjoint_gradient_method(self):
        """Test that specifying diff_method="adjoint" updates the execution config correctly."""
        dev = qml.device("default.qubit", wires=1)
        config = ExecutionConfig(gradient_method="adjoint")

        _, new_config = dev.preprocess(execution_config=config)
        assert new_config.gradient_method == "adjoint"
        assert new_config.use_device_jacobian_product
        assert new_config.grad_on_execution

    def test_execution_config_default_mcm_config(self):
        """Test that not specifying MCM config updates the execution config correctly."""
        dev = qml.device("default.qubit", wires=1)
        config = ExecutionConfig()

        _, new_config = dev.preprocess(execution_config=config)
        assert new_config.mcm_config == MCMConfig(mcm_method="deferred", postselect_mode=None)

    @pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
    def test_single_branch_statistics_postselect_mode_warning(self, postselect_mode):
        """Test that setting a postselect_mode with single-branch-statistics raises a warning."""
        dev = qml.device("default.qubit", wires=1)
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

    def test_default_mcm_method_deferred(self):
        """Test that the default mcm_method is deferred."""
        config = qml.device("default.qubit").setup_execution_config()
        assert config.mcm_config.mcm_method == "deferred"

    def test_transform_program(self):
        """Test that the transform program returned by preprocess has the correct transforms."""
        dev = qml.device("default.qubit", wires=1)

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


class TestExecution:
    """Unit tests for default.qubit execution with program capture."""

    def test_requires_wires(self):
        """Test that a device error is raised if device wires are not specified."""

        jaxpr = jax.make_jaxpr(lambda x: x + 1)(0.1)
        dev = qml.device("default.qubit")

        with pytest.raises(DeviceError, match="Device wires are required."):
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.2)

    def test_no_partitioned_shots(self):
        """Test that an error is raised if the device has partitioned shots."""

        jaxpr = jax.make_jaxpr(lambda x: x + 1)(0.1)
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(
            DeviceError,
            match="Shot vectors are unsupported with jaxpr execution.",
        ):
            dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.2, shots=(100, 100))

    def test_use_device_prng(self):
        """Test that sampling depends on the device prng."""

        key1 = jax.random.PRNGKey(1234)
        key2 = jax.random.PRNGKey(1234)

        dev1 = qml.device("default.qubit", wires=1, seed=key1)
        dev2 = qml.device("default.qubit", wires=1, seed=key2)

        def f():
            qml.H(0)
            return qml.sample(wires=0)

        jaxpr = jax.make_jaxpr(f)()

        samples1 = dev1.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, shots=100)
        samples2 = dev2.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, shots=100)

        assert qml.math.allclose(samples1, samples2)

    def test_no_prng_key(self):
        """Test that that sampling works without a provided prng key."""

        dev = qml.device("default.qubit", wires=1)

        def f():
            return qml.sample(wires=0)

        jaxpr = jax.make_jaxpr(f)()
        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, shots=100)
        assert qml.math.allclose(res, jax.numpy.zeros(100))

    def test_simple_execution(self):
        """Test the execution, jitting, and gradient of a simple quantum circuit."""

        def f(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.123)

        dev = qml.device("default.qubit", wires=1)

        res = dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)
        assert qml.math.allclose(res, jax.numpy.cos(0.5))


class TestJVP:
    """Unit tests for default.qubit JVP with program capture."""

    def test_error_unsupported_diff_method(self):

        dev = qml.device("default.qubit", wires=1)
        jaxpr = jax.make_jaxpr(lambda x: x + 1)(1)
        config = qml.devices.ExecutionConfig(gradient_method="hello")
        with pytest.raises(NotImplementedError, match="does not support gradient_method=hello"):
            dev.jaxpr_jvp(jaxpr.jaxpr, jaxpr.consts, 2, execution_config=config)

    def test_adjoint_unsupported_control_flow(self):

        def circuit(x):
            @qml.for_loop(3)
            def f(i):
                qml.RX(x, i)

            f()

            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(circuit)(0.5)
        dev = qml.device("default.qubit", wires=4)
        config = qml.devices.ExecutionConfig(gradient_method="adjoint")

        with pytest.raises(NotImplementedError, match="does not have a jvp rule"):
            dev.jaxpr_jvp(jaxpr.jaxpr, (0.5,), (1.0,), execution_config=config)
