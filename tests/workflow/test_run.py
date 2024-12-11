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

import autograd
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
    # 0
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots((100000, 100000)),
    ],
    # 1
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(100000),
    ],
    # 2
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(None),
    ],
    # 3
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method="backprop",
        ),
        Shots(None),
    ],
    # 4
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method="adjoint",
            use_device_jacobian_product=True,
        ),
        Shots(None),
    ],
    # 5
    [
        "default.qubit",
        replace(
            DefaultExecutionConfig, gradient_method="adjoint", use_device_jacobian_product=True
        ),
        Shots(None),
    ],
    # 6
    [
        "reference.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots((100000, 100000)),
    ],
    # 7
    [
        "reference.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(100000),
    ],
    # 8
    [
        "reference.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(None),
    ],
    # 9
    [
        "param_shift.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method="device",
            use_device_jacobian_product=False,
        ),
        Shots((100000, 100000)),
    ],
    # 10
    [
        "param_shift.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method="device",
            use_device_jacobian_product=False,
        ),
        Shots(100000),
    ],
    # 11
    [
        "param_shift.qubit",
        replace(
            DefaultExecutionConfig,
            gradient_method="device",
            use_device_jacobian_product=False,
        ),
        Shots(None),
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

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        torch.set_default_dtype(torch.float64)
        yield
        torch.set_default_dtype(torch.float32)

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

    def test_scalar_jacobian(self, device, config, shots, seed):
        """Test scalar jacobian calculation"""
        device = get_device(device, seed)
        config = replace(config, interface="torch")

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            _, inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)
            return run([tape], device, resolved_config, inner_tp)[0]

        a = torch.tensor(0.1, requires_grad=True)
        print(cost(a))
        res = torch.autograd.functional.jacobian(cost, a)
        if not shots.has_partitioned_shots:
            assert res.shape == ()  # pylint: disable=no-member

        # compare to standard tape jacobian
        tape = qml.tape.QuantumScript([qml.RY(a, wires=0)], [qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        tapes, fn = qml.gradients.param_shift(tape)
        expected = fn(device.execute(tapes))

        assert expected.shape == ()
        if shots.has_partitioned_shots:
            for i in range(shots.num_copies):
                assert torch.allclose(res[i], expected, atol=atol_for_shots(shots), rtol=0)
                assert torch.allclose(res[i], -torch.sin(a), atol=atol_for_shots(shots))
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
            assert torch.allclose(res, -torch.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, device, config, shots, seed):
        """Test jacobian calculation"""
        device = get_device(device, seed)
        config = replace(config, interface="torch")

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)

            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            _, inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)
            res = run([tape], device, resolved_config, inner_tp)[0]

            if shots.has_partitioned_shots:
                return torch.hstack(res[0] + res[1])

            return torch.hstack(res)

        a = torch.tensor(0.1, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)

        res = cost(a, b)
        expected = torch.tensor([torch.cos(a), -torch.cos(a) * torch.sin(b)])
        if shots.has_partitioned_shots:
            assert torch.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert torch.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert torch.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = torch.autograd.functional.jacobian(cost, (a, b))
        assert isinstance(res, tuple) and len(res) == 2

        expected = (
            torch.tensor([-torch.sin(a), torch.sin(a) * torch.sin(b)]),
            torch.tensor([0, -torch.cos(a) * torch.cos(b)]),
        )
        if shots.has_partitioned_shots:
            assert res[0].shape == (4,)
            assert res[1].shape == (4,)

            for _r, _e in zip(res, expected):
                assert torch.allclose(_r[:2], _e, atol=atol_for_shots(shots))
                assert torch.allclose(_r[2:], _e, atol=atol_for_shots(shots))

        else:
            assert res[0].shape == (2,)
            assert res[1].shape == (2,)

            for _r, _e in zip(res, expected):
                assert torch.allclose(_r, _e, atol=atol_for_shots(shots))


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

    def test_scalar_jacobian(self, device, config, shots, seed):
        """Test scalar jacobian calculation"""
        device = get_device(device, seed=seed)
        config = replace(config, interface="autograd")

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape], device, resolved_config, inner_tp)[0]

        a = pnp.array(0.1, requires_grad=True)
        if shots.has_partitioned_shots:
            res = qml.jacobian(lambda x: qml.math.hstack(cost(x)))(a)
        else:
            res = qml.jacobian(cost)(a)
            assert res.shape == ()  # pylint: disable=no-member

        # compare to standard tape jacobian
        tape = qml.tape.QuantumScript([qml.RY(a, wires=0)], [qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        tapes, fn = qml.gradients.param_shift(tape)
        expected = fn(device.execute(tapes))

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -np.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, device, config, shots, seed):
        """Test jacobian calculation"""
        device = get_device(device, seed=seed)
        config = replace(config, interface="autograd")

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return autograd.numpy.hstack(run([tape], device, resolved_config, inner_tp)[0])

        a = pnp.array(0.1, requires_grad=True)
        b = pnp.array(0.2, requires_grad=True)
        res = cost(a, b)
        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        if shots.has_partitioned_shots:
            assert np.allclose(res[:2], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[2:], expected, atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        res = qml.jacobian(cost)(a, b)
        assert isinstance(res, tuple) and len(res) == 2
        if shots.has_partitioned_shots:
            assert res[0].shape == (4,)
            assert res[1].shape == (4,)

            expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
            for _r, _e in zip(res, expected):
                assert np.allclose(_r[:2], _e, atol=atol_for_shots(shots))
                assert np.allclose(_r[2:], _e, atol=atol_for_shots(shots))
        else:
            assert res[0].shape == (2,)
            assert res[1].shape == (2,)

            expected = ([-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)])
            for _r, _e in zip(res, expected):
                assert np.allclose(_r, _e, atol=atol_for_shots(shots))


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

    def test_scalar_jacobian(self, device, config, shots, seed):
        """Test scalar jacobian calculation"""
        device = get_device(device, seed)
        config = replace(config, interface="jax")

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape], device, resolved_config, inner_tp)[0]

        a = jnp.array(0.1)
        res = jax.jacobian(cost)(a)
        if not shots.has_partitioned_shots:
            assert res.shape == ()  # pylint: disable=no-member

        # compare to standard tape jacobian
        tape = qml.tape.QuantumScript([qml.RY(a, wires=0)], [qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        tapes, fn = qml.gradients.param_shift(tape)
        expected = fn(device.execute(tapes))

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -jnp.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, device, config, shots, seed):
        """Test jacobian calculation"""
        device = get_device(device, seed)
        config = replace(config, interface="jax")

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)

            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape], device, resolved_config, inner_tp)[0]

        a = jnp.array(0.1)
        b = jnp.array(0.2)
        res = cost(a, b)

        expected = [jnp.cos(a), -jnp.cos(a) * jnp.sin(b)]
        if shots.has_partitioned_shots:
            assert np.allclose(res[0], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[1], expected, atol=2 * atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        g = jax.jacobian(cost, argnums=[0, 1])(a, b)
        assert isinstance(g, tuple) and len(g) == 2

        expected = ([-jnp.sin(a), jnp.sin(a) * jnp.sin(b)], [0, -jnp.cos(a) * jnp.cos(b)])

        if shots.has_partitioned_shots:
            for i in (0, 1):
                assert np.allclose(g[i][0][0], expected[0][0], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][1][0], expected[0][1], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][0][1], expected[1][0], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][1][1], expected[1][1], atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(g[0][0], expected[0][0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[1][0], expected[0][1], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[0][1], expected[1][0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[1][1], expected[1][1], atol=atol_for_shots(shots), rtol=0)


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)


@pytest.mark.jax
@pytest.mark.parametrize("device, config, shots", test_matrix)
class TestJaxJitRun:
    def test_run(self, device, config, shots, seed):
        """Test execution of tapes on 'jax-jit' interface."""
        device = get_device(device, seed=seed)
        config = replace(config, interface="jax-jit")

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
        b = jnp.array(0.2)

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

    def test_scalar_jacobian(self, device, config, shots, seed):
        """Test scalar jacobian calculation"""
        device = get_device(device, seed)
        config = replace(config, interface="jax-jit")

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape], device, resolved_config, inner_tp)[0]

        a = jnp.array(0.1)
        res = jax.jit(jax.jacobian(cost))(a)
        if not shots.has_partitioned_shots:
            assert res.shape == ()  # pylint: disable=no-member

        # compare to standard tape jacobian
        tape = qml.tape.QuantumScript([qml.RY(a, wires=0)], [qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        tapes, fn = qml.gradients.param_shift(tape)
        expected = fn(device.execute(tapes))

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -jnp.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, device, config, shots, seed):
        """Test jacobian calculation"""
        device = get_device(device, seed)
        config = replace(config, interface="jax-jit")

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)

            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape], device, resolved_config, inner_tp)[0]

        a = jnp.array(0.1)
        b = jnp.array(0.2)
        res = cost(a, b)

        expected = [jnp.cos(a), -jnp.cos(a) * jnp.sin(b)]
        if shots.has_partitioned_shots:
            assert np.allclose(res[0], expected, atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(res[1], expected, atol=2 * atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        g = jax.jit(jax.jacobian(cost, argnums=[0, 1]))(a, b)
        assert isinstance(g, tuple) and len(g) == 2

        expected = ([-jnp.sin(a), jnp.sin(a) * jnp.sin(b)], [0, -jnp.cos(a) * jnp.cos(b)])

        if shots.has_partitioned_shots:
            for i in (0, 1):
                assert np.allclose(g[i][0][0], expected[0][0], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][1][0], expected[0][1], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][0][1], expected[1][0], atol=atol_for_shots(shots), rtol=0)
                assert np.allclose(g[i][1][1], expected[1][1], atol=atol_for_shots(shots), rtol=0)
        else:
            assert np.allclose(g[0][0], expected[0][0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[1][0], expected[0][1], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[0][1], expected[1][0], atol=atol_for_shots(shots), rtol=0)
            assert np.allclose(g[1][1], expected[1][1], atol=atol_for_shots(shots), rtol=0)


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

    def test_scalar_jacobian(self, device, config, shots, seed):
        """Test scalar jacobian calculation"""
        if shots.has_partitioned_shots:
            pytest.xfail(reason="Partitioned shots are not supported yet.")
        device_vjp = config.use_device_jacobian_product
        device = get_device(device, seed=seed)
        config = replace(config, interface="tensorflow")

        def cost(a):
            tape = qml.tape.QuantumScript([qml.RY(a, 0)], [qml.expval(qml.PauliZ(0))], shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return run([tape], device, resolved_config, inner_tp)[0]

        a = tf.Variable(0.1, dtype=tf.float64)
        with tf.GradientTape(persistent=device_vjp) as tape:
            cost_res = cost(a)
        res = tape.jacobian(cost_res, a, experimental_use_pfor=not device_vjp)
        assert res.shape == ()  # pylint: disable=no-member

        # compare to standard tape jacobian
        tape = qml.tape.QuantumScript([qml.RY(a, wires=0)], [qml.expval(qml.PauliZ(0))])
        tape.trainable_params = [0]
        tapes, fn = qml.gradients.param_shift(tape)
        expected = fn(device.execute(tapes))

        assert expected.shape == ()
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)
        assert np.allclose(res, -tf.sin(a), atol=atol_for_shots(shots))

    def test_jacobian(self, device, config, shots, seed):
        """Test jacobian calculation"""
        if shots.has_partitioned_shots:
            pytest.xfail(reason="Partitioned shots are not supported yet.")

        config = replace(config, interface="tensorflow")
        device = get_device(device, seed=seed)
        device_vjp = config.use_device_jacobian_product

        def cost(a, b):
            ops = [qml.RY(a, wires=0), qml.RX(b, wires=1), qml.CNOT(wires=[0, 1])]
            m = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]
            tape = qml.tape.QuantumScript(ops, m, shots=shots)
            resolved_config = _resolve_execution_config(config, device, [tape], TransformProgram())
            inner_tp = _setup_transform_program(TransformProgram(), device, resolved_config)[1]
            return qml.math.hstack(
                run([tape], device, resolved_config, inner_tp)[0], like="tensorflow"
            )

        a = tf.Variable(0.1)
        b = tf.Variable(0.2)
        with tf.GradientTape(persistent=device_vjp) as tape:
            res = cost(a, b)
        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=atol_for_shots(shots), rtol=0)

        jac = tape.jacobian(res, [a, b], experimental_use_pfor=not device_vjp)
        assert isinstance(jac, list) and len(jac) == 2
        assert jac[0].shape == (2,)
        assert jac[1].shape == (2,)

        expected = ([-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)])
        for _r, _e in zip(jac, expected):
            assert np.allclose(_r, _e, atol=atol_for_shots(shots))


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
