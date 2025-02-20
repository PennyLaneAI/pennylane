# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.devices.DefaultMixed` device.
"""
# pylint: disable=protected-access
import numpy as np
import pytest

import pennylane as qml
from pennylane.devices import DefaultMixed
from pennylane.devices.default_mixed import _apply_state_vector
from pennylane.math import Interface

ML_INTERFACES = ["numpy", "autograd", "torch", "tensorflow", "jax"]


class TestDefaultMixedInit:
    """Unit tests for DefaultMixed initialization"""

    def test_name_property(self):
        """Test the name property returns correct device name"""
        dev = DefaultMixed(wires=1)
        assert dev.name == "default.mixed"

    @pytest.mark.parametrize("readout_prob", [-0.1, 1.1, 2.0])
    def test_readout_probability_validation(self, readout_prob):
        """Test readout probability validation during initialization"""
        with pytest.raises(ValueError, match="readout error probability should be in the range"):
            DefaultMixed(wires=1, readout_prob=readout_prob)

    @pytest.mark.parametrize("readout_prob", ["0.5", [0.5], (0.5,)])
    def test_readout_probability_type_validation(self, readout_prob):
        """Test readout probability type validation"""
        with pytest.raises(TypeError, match="readout error probability should be an integer"):
            DefaultMixed(wires=1, readout_prob=readout_prob)

    def test_seed_global(self):
        """Test global seed initialization"""
        dev = DefaultMixed(wires=1, seed="global")
        assert dev._rng is not None
        assert dev._prng_key is None

    @pytest.mark.jax
    def test_seed_jax(self):
        """Test JAX PRNGKey seed initialization"""
        # pylint: disable=import-outside-toplevel
        import jax

        dev = DefaultMixed(wires=1, seed=jax.random.PRNGKey(0))
        assert dev._rng is not None
        assert dev._prng_key is not None

    def test_supports_derivatives(self):
        """Test supports_derivatives method"""
        dev = DefaultMixed(wires=1)
        assert dev.supports_derivatives()
        assert not dev.supports_derivatives(
            execution_config=qml.devices.execution_config.ExecutionConfig(
                gradient_method="finite-diff"
            )
        )

    @pytest.mark.parametrize("nr_wires", [1, 2, 3, 10, 22])
    def test_valid_wire_numbers(self, nr_wires):
        """Test initialization with different valid wire numbers"""
        dev = DefaultMixed(wires=nr_wires)
        assert len(dev.wires) == nr_wires

    def test_wire_initialization_list(self):
        """Test initialization with wire list"""
        dev = DefaultMixed(wires=["a", "b", "c"])
        assert dev.wires == qml.wires.Wires(["a", "b", "c"])

    def test_too_many_wires(self):
        """Test error raised when too many wires requested"""
        with pytest.raises(ValueError, match="This device does not currently support"):
            DefaultMixed(wires=24)

    def test_execute_no_diff_method(self):
        """Test that the execute method is defined"""
        dev = DefaultMixed(wires=[0, 1])
        execution_config = qml.devices.execution_config.ExecutionConfig(
            gradient_method="finite-diff"
        )  # in-valid one for this device
        processed_config = dev._setup_execution_config(execution_config)
        assert (
            processed_config.interface is Interface.NUMPY
        ), "The interface should be set to numpy for an invalid gradient method"


# pylint: disable=protected-access, too-few-public-methods
class TestLegacyDefaultMixed:
    """
    Tests that covered parts of legacy method of the DefaultMixed device.
    """

    # @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ML_INTERFACES)
    @pytest.mark.parametrize("device_wires", [qml.wires.Wires([0]), qml.wires.Wires([0, 1])])
    def test_apply_state_vector(self, device_wires, interface):
        """Initialize the internal state in a specified pure state."""
        num_wires = len(device_wires)
        tolerance = 1e-10
        state_np = np.zeros(shape=(2**num_wires,), dtype=np.complex128)
        state_np[0] = 1.0
        _state = _apply_state_vector(
            device_wires=device_wires, state_vector=state_np, interface=interface
        )

        rho_expected_numpy = np.zeros([2] * 2 * num_wires, dtype=np.complex128)
        rho_expected_numpy[tuple([0] * 2 * num_wires)] = 1.0
        rho_expected = qml.math.asarray(rho_expected_numpy, like=interface)
        assert qml.math.allclose(_state, rho_expected, atol=tolerance)
