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

import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.numpy import pi as PI
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformContainer, TransformProgram
from pennylane.transforms.optimization import merge_rotations
from pennylane.workflow import run


@pytest.mark.torch
@pytest.mark.parametrize(
    "interface, gradient_method",
    (["numpy", qml.gradients.param_shift], ["torch", None], ["torch", "backprop"]),
)
def test_no_interface_boundary_required(interface, gradient_method):
    """Test that tapes are executed without an ML boundary if no interface boundary required."""
    container = TransformContainer(merge_rotations)
    inner_tp = TransformProgram((container,))
    device = qml.device("default.qubit")
    tapes = [QuantumScript([qml.RX(PI, wires=0), qml.RX(PI, wires=0)], [qml.expval(qml.PauliZ(0))])]
    config = ExecutionConfig(interface=interface, gradient_method=gradient_method)
    results = run(tapes, device, config, inner_tp)

    assert qml.math.get_interface(results) == "numpy"
    assert qml.math.allclose(results[0], 1.0)


def test_grad_on_execution_error():
    """Tests that a ValueError is raised if the config uses grad_on_execution."""
    inner_tp = TransformProgram()
    device = qml.device("default.qubit")
    tapes = [QuantumScript([qml.RX(PI, wires=0), qml.RX(PI, wires=0)], [qml.expval(qml.PauliZ(0))])]
    config = ExecutionConfig(
        interface="torch",
        gradient_method=qml.gradients.param_shift,
        grad_on_execution=True,
        use_device_jacobian_product=False,
        use_device_gradient=False,
    )

    with pytest.raises(
        ValueError, match="Gradient transforms cannot be used with grad_on_execution=True"
    ):
        run(tapes, device, config, inner_tp)
