# Copyright 2018-2024 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the `run` helper function in the `qml.workflow` module."""

from dataclasses import replace

import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.devices import DefaultExecutionConfig, DefaultQubit, ExecutionConfig
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformContainer, TransformProgram
from pennylane.transforms.optimization import merge_rotations
from pennylane.workflow import run


# pylint: disable=inconsistent-return-statements
def convert_to_interface(arr, interface):
    """Dispatch arrays for different interfaces"""
    import jax.numpy as jnp
    import tensorflow as tf
    import torch

    if interface == "autograd":
        return pnp.array(arr)

    if interface == "jax":
        return jnp.array(arr)

    if interface == "tf":
        return tf.constant(arr)

    if interface == "torch":
        return torch.tensor(arr)


# Create the device and execution configurations
qubit_device_and_config = [
    [
        DefaultQubit(),
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
            grad_on_execution=False,
            use_device_gradient=False,
        ),
    ],
]


class TestNoInterface:

    def test_numpy_interface(self):
        """Test that tapes are executed correctly with the NumPy interface."""
        container = TransformContainer(merge_rotations)
        inner_tp = TransformProgram((container,))
        device = qml.device("default.qubit")
        tapes = [
            QuantumScript(
                [qml.RX(pnp.pi, wires=0), qml.RX(pnp.pi, wires=0)], [qml.expval(qml.PauliZ(0))]
            )
        ]
        config = ExecutionConfig(interface="numpy", gradient_method=qml.gradients.param_shift)
        results = run(tapes, device, config, inner_tp)

        assert qml.math.get_interface(results) == "numpy"
        assert qml.math.allclose(results[0], 1.0)

    @pytest.mark.torch
    @pytest.mark.parametrize(
        "interface, gradient_method",
        [("torch", None), ("torch", "backprop")],
    )
    def test_no_gradient_computation_required(self, interface, gradient_method):
        """Test that tapes execute without an ML boundary when no gradient computation is required."""
        container = TransformContainer(merge_rotations)
        inner_tp = TransformProgram((container,))
        device = qml.device("default.qubit")
        tapes = [
            QuantumScript(
                [qml.RX(pnp.pi, wires=0), qml.RX(pnp.pi, wires=0)], [qml.expval(qml.PauliZ(0))]
            )
        ]
        config = ExecutionConfig(interface=interface, gradient_method=gradient_method)
        results = run(tapes, device, config, inner_tp)

        assert qml.math.get_interface(results) == "numpy"
        assert qml.math.allclose(results[0], 1.0)


# pylint: disable=too-few-public-methods
@pytest.mark.all_interfaces
class TestJPCInterface:

    @pytest.mark.parametrize("interface", ["torch", "jax", "tensorflow", "autograd"])
    @pytest.mark.parametrize("device, config", qubit_device_and_config)
    def test_run_with_interface(self, interface, device, config):
        container = TransformContainer(merge_rotations)
        inner_tp = TransformProgram((container,))

        config = replace(config, interface=interface)

        x = pnp.array(pnp.pi, requires_grad=True)
        phi = convert_to_interface(x, config.interface.value)
        assert qml.math.get_deep_interface(phi) == interface
        tapes = [
            QuantumScript([qml.RX(phi, wires=0), qml.RX(phi, wires=0)], [qml.expval(qml.PauliZ(0))])
        ]

        results = run(tapes, device, config, inner_tp)
        assert qml.math.allclose(results[0], 1.0)

        # TODO: Why is this not autograd?
        expected_interface = interface
        if interface == "autograd":
            expected_interface = "numpy"

        assert qml.math.get_deep_interface(results) == expected_interface


# pylint: disable=too-few-public-methods
@pytest.mark.tf
class TestTFAutograph:

    interface = "tf-autograph"

    def test_grad_on_execution_error(self):
        """Tests that a ValueError is raised if the config uses grad_on_execution."""
        inner_tp = TransformProgram()
        device = qml.device("default.qubit")
        tapes = [
            QuantumScript(
                [qml.RX(pnp.pi, wires=0), qml.RX(pnp.pi, wires=0)], [qml.expval(qml.PauliZ(0))]
            )
        ]
        config = ExecutionConfig(
            interface=self.interface,
            gradient_method=qml.gradients.param_shift,
            grad_on_execution=True,
            use_device_jacobian_product=False,
            use_device_gradient=False,
        )

        with pytest.raises(
            ValueError, match="Gradient transforms cannot be used with grad_on_execution=True"
        ):
            run(tapes, device, config, inner_tp)
