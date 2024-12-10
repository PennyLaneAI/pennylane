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
from dataclasses import replace

import numpy as np
import pytest
from param_shift_dev import ParamShiftDerivativesDevice

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.devices import DefaultExecutionConfig, ExecutionConfig
from pennylane.measurements import Shots
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformContainer, TransformProgram
from pennylane.transforms.optimization import merge_rotations
from pennylane.workflow import _resolve_execution_config, _setup_transform_program, run


def atol_for_shots(shots):
    """Return higher tolerance if finite shots."""
    return 1e-2 if shots else 1e-6


def get_device(device_name, seed):
    if device_name == "param_shift.qubit":
        return ParamShiftDerivativesDevice(seed=seed)
    return qml.device(device_name, seed=seed)


# Create the device and execution configurations
test_matrix = [
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots((100000, 100000)),
    ],
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(100000),
    ],
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(None),
    ],
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method="backprop",
        ),
        Shots(None),
    ],
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method="adjoint",
        ),
        Shots(None),
    ],
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig, gradient_method="adjoint", use_device_jacobian_product=True
        ),
        Shots(None),
    ],
    [
        "reference.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots((100000, 100000)),
    ],
    [
        "reference.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(100000),
    ],
    [
        "reference.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(None),
    ],
    [
        "param_shift.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method="device",
        ),
        Shots((100000, 100000)),
    ],
]


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


torch = pytest.importorskip("torch")


@pytest.mark.torch
@pytest.mark.parametrize("device, config, shots", test_matrix)
class TestTorchRun:
    def test_run(self, device, config, shots, seed):
        """Test execution of tapes on 'torch' interface."""
        device = get_device(device, seed=seed)
        config = replace(config, interface="torch")

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            resolved_config = _resolve_execution_config(
                config, device, [tape1, tape2], TransformProgram()
            )
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape1, tape2], device, resolved_config, inner_tp)

        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=False)

        with device.tracker:
            res = cost(a, b)

        assert len(res) == 2

        if getattr(config, "grad_on_execution", False):
            assert device.tracker.totals["execute_and_derivate_batches"] == 1
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2

        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        exp = torch.cos(a) * torch.cos(b)
        if shots.has_partitioned_shots:
            for shot in range(2):
                for wire in range(2):
                    assert qml.math.allclose(res[shot][wire], exp, atol=atol_for_shots(shots))
        else:
            for wire in range(2):
                assert qml.math.allclose(res[wire], exp, atol=atol_for_shots(shots))


@pytest.mark.parametrize("device, config, shots", test_matrix)
class TestAutogradRun:

    def test_run(self, device, config, shots, seed):
        """Test execution of tapes on 'autograd' interface."""
        device = get_device(device, seed=seed)
        config = replace(config, interface="autograd")

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            resolved_config = _resolve_execution_config(
                config, device, [tape1, tape2], TransformProgram()
            )
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape1, tape2], device, resolved_config, inner_tp)

        a = pnp.array(0.1, requires_grad=True)
        b = pnp.array(0.2, requires_grad=False)

        with device.tracker:
            res = cost(a, b)

        assert len(res) == 2

        if getattr(config, "grad_on_execution", False):
            assert device.tracker.totals["execute_and_derivate_batches"] == 1
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2

        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        assert qml.math.allclose(res[0], np.cos(a) * np.cos(b), atol=atol_for_shots(shots))
        assert qml.math.allclose(res[1], np.cos(a) * np.cos(b), atol=atol_for_shots(shots))


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)


@pytest.mark.jax
@pytest.mark.parametrize("device, config, shots", test_matrix)
class TestJaxRun:
    def test_run(self, device, config, shots, seed):
        """Test execution of tapes on 'jax' interface."""
        device = get_device(device, seed=seed)
        config = replace(config, interface="jax")

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            resolved_config = _resolve_execution_config(
                config, device, [tape1, tape2], TransformProgram()
            )
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape1, tape2], device, resolved_config, inner_tp)

        a = jnp.array(0.1)
        b = np.array(0.2)

        with device.tracker:
            res = cost(a, b)

        assert len(res) == 2

        if getattr(config, "diff_method", None) == "adjoint":
            assert device.tracker.totals.get("execute_and_derivative_batches", 0) == 0
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2

        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        assert qml.math.allclose(res[0], jnp.cos(a) * jnp.cos(b), atol=atol_for_shots(shots))
        assert qml.math.allclose(res[1], jnp.cos(a) * jnp.cos(b), atol=atol_for_shots(shots))


tf = pytest.importorskip("tensorflow")


@pytest.mark.tf
@pytest.mark.parametrize("device, config, shots", test_matrix)
class TestTensorFlowRun:
    def test_run(self, device, config, shots, seed):
        """Test execution of tapes on 'tensorflow' interface."""
        device = get_device(device, seed=seed)
        config = replace(config, interface="tensorflow")

        def cost(a, b):
            ops1 = [qml.RY(a, wires=0), qml.RX(b, wires=0)]
            tape1 = qml.tape.QuantumScript(ops1, [qml.expval(qml.PauliZ(0))], shots=shots)

            ops2 = [qml.RY(a, wires="a"), qml.RX(b, wires="a")]
            tape2 = qml.tape.QuantumScript(ops2, [qml.expval(qml.PauliZ("a"))], shots=shots)

            resolved_config = _resolve_execution_config(
                config, device, [tape1, tape2], TransformProgram()
            )
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape1, tape2], device, resolved_config, inner_tp)

        a = tf.Variable(0.1, dtype="float64")
        b = tf.constant(0.2, dtype="float64")

        with device.tracker:
            res = cost(a, b)

        assert len(res) == 2

        if getattr(config, "diff_method", None) == "adjoint" and not getattr(
            config, "use_device_jacobian_product", False
        ):
            assert device.tracker.totals["execute_and_derivative_batches"] == 1
        else:
            assert device.tracker.totals["batches"] == 1
        assert device.tracker.totals["executions"] == 2

        if not shots.has_partitioned_shots:
            assert res[0].shape == ()
            assert res[1].shape == ()

        assert qml.math.allclose(res[0], tf.cos(a) * tf.cos(b), atol=atol_for_shots(shots))
        assert qml.math.allclose(res[1], tf.cos(a) * tf.cos(b), atol=atol_for_shots(shots))


@pytest.mark.tf
class TestTFAutographRun:

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
