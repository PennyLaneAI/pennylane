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

# pylint: disable=too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.devices import ExecutionConfig
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformContainer, TransformProgram
from pennylane.transforms.optimization import merge_rotations
from pennylane.workflow import run


class TestNoInterfaceRequired:

    def test_numpy_interface(self, seed):
        """Test that tapes are executed correctly with the NumPy interface."""
        container = TransformContainer(merge_rotations)
        inner_tp = TransformProgram((container,))
        device = qml.device("default.qubit", seed=seed)
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
    def test_no_gradient_computation_required(self, interface, gradient_method, seed):
        """Test that tapes execute without an ML boundary when no gradient computation is required."""
        container = TransformContainer(merge_rotations)
        inner_tp = TransformProgram((container,))
        device = qml.device("default.qubit", seed=seed)
        tapes = [
            QuantumScript(
                [qml.RX(pnp.pi, wires=0), qml.RX(pnp.pi, wires=0)], [qml.expval(qml.PauliZ(0))]
            )
        ]
        config = ExecutionConfig(interface=interface, gradient_method=gradient_method)
        results = run(tapes, device, config, inner_tp)

        assert qml.math.get_interface(results) == "numpy"
        assert qml.math.allclose(results[0], 1.0)


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", ["autograd", "jax-jit", "jax", "torch"])
def test_grad_on_execution_error(interface):
    """Tests that a ValueError is raised if the config uses grad_on_execution."""
    inner_tp = TransformProgram()
    device = qml.device("default.qubit")
    tapes = [
        QuantumScript(
            [qml.RX(pnp.pi, wires=0), qml.RX(pnp.pi, wires=0)], [qml.expval(qml.PauliZ(0))]
        )
    ]
    config = ExecutionConfig(
        interface=interface,
        gradient_method=qml.gradients.param_shift,
        grad_on_execution=True,
        use_device_jacobian_product=False,
        use_device_gradient=False,
    )

    with pytest.raises(
        ValueError, match="Gradient transforms cannot be used with grad_on_execution=True"
    ):
        run(tapes, device, config, inner_tp)
