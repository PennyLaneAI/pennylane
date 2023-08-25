# Copyright 2023 Xanadu Quantum Technologies Inc.

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
Tests for the jacobian product calculator classes.
"""
# pylint: disable=protected-access
import pytest

import numpy as np

import pennylane as qml
from pennylane.interfaces.jacobian_products import (
    JacobianProductCalculator,
    TransformJacobianProducts,
    DeviceJacobians,
)

dev = qml.devices.experimental.DefaultQubit2()
dev_old = qml.devices.DefaultQubit(wires=5)
adjoint_config = qml.devices.experimental.ExecutionConfig(gradient_method="adjoint")


def inner_execute_numpy(tapes):
    return dev.execute(tapes)


param_shift_jpc = TransformJacobianProducts(inner_execute_numpy, qml.gradients.param_shift)
hadamard_grad_jpc = TransformJacobianProducts(
    inner_execute_numpy, qml.gradients.hadamard_grad, {"aux_wire": "aux"}
)
device_jacs = DeviceJacobians(dev, adjoint_config, {})
legacy_jacs = DeviceJacobians(dev_old, adjoint_config, {"method": "adjoint_jacobian"})

transform_jpc_matrix = [param_shift_jpc, hadamard_grad_jpc]
jpc_matrix = [param_shift_jpc, hadamard_grad_jpc, device_jacs, legacy_jacs]


# pylint: disable=too-few-public-methods
class TestBasics:
    """Test initialization and repr for jacobian product calculator classes."""

    def test_transform_jacobian_product_basics(self):
        """Test the initialization and basic properties of a TransformJacobianProduct class."""
        jpc = TransformJacobianProducts(
            inner_execute_numpy, qml.gradients.hadamard_grad, {"aux_wire": "aux"}
        )

        assert isinstance(jpc, JacobianProductCalculator)
        assert jpc._inner_execute is inner_execute_numpy
        assert jpc._gradient_transform is qml.gradients.hadamard_grad
        assert jpc._gradient_kwargs == {"aux_wire": "aux"}

        expected_repr = (
            f"TransformJacobianProducts({repr(inner_execute_numpy)}, "
            "gradient_transform=<gradient_transform: _hadamard_grad>, "
            "gradient_kwargs={'aux_wire': 'aux'})"
        )
        assert repr(jpc) == expected_repr


@pytest.mark.parametrize("jpc", jpc_matrix)
class TestJacobianProductResults:
    """Test first order results for the matrix of jpc options."""

    def test_execute_jvp_basic(self, jpc):
        """Test execute_and_compute_jvp for a simple single input single output."""
        x = 0.92
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        tangents = ((0.5,),)
        res, jvp = jpc.execute_and_compute_jvp((tape,), tangents)
        assert qml.math.allclose(res[0], np.cos(x))
        assert qml.math.allclose(jvp[0], -0.5 * np.sin(x))

    def test_vjp_basic(self, jpc):
        """Test compute_vjp for a simple single input single output."""
        x = -0.294
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        dy = ((1.8,),)
        vjp = jpc.compute_vjp((tape,), dy)
        assert qml.math.allclose(vjp[0], -1.8 * np.sin(x))

    def test_jacobian_basic(self, jpc):
        """Test compute_jacobian for a simple single input single output."""
        x = 1.62
        tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
        jac = jpc.compute_jacobian((tape,))
        assert qml.math.allclose(jac, -np.sin(x))


@pytest.mark.parametrize("jpc", transform_jpc_matrix)
class TestProbsOut:
    def test_execute_jvp_multi_params_multi_out(self, jpc):
        """Test execute_and_compute_jvp with multiple parameters and multiple outputs"""
        x = 0.62
        y = 2.64
        ops = [qml.RY(y, 0), qml.RX(x, 0)]
        measurements = [qml.probs(wires=0), qml.expval(qml.PauliZ(0))]
        tape1 = qml.tape.QuantumScript(ops, measurements)

        phi = 0.623
        ops2 = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        measurements2 = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        tape2 = qml.tape.QuantumScript(ops2, measurements2)

        tangents = (1.5, 2.5)
        tangents2 = (0.6,)
        res, jvp = jpc.execute_and_compute_jvp((tape1, tape2), (tangents, tangents2))

        expected_res00 = 0.5 * np.array([1 + np.cos(x) * np.cos(y), 1 - np.cos(x) * np.cos(y)])
        assert qml.math.allclose(res[0][0], expected_res00)

        expected_res01 = np.cos(x) * np.cos(y)
        assert qml.math.allclose(res[0][1], expected_res01)

        assert qml.math.allclose(res[1][0], 0)
        assert qml.math.allclose(res[1][1], np.cos(phi))

        res0dx = 0.5 * np.array([-np.sin(x) * np.cos(y), np.sin(x) * np.cos(y)])
        res0dy = 0.5 * np.array([-np.cos(x) * np.sin(y), np.cos(x) * np.sin(y)])
        expected_jvp00 = 2.5 * res0dx + 1.5 * res0dy
        assert qml.math.allclose(expected_jvp00, jvp[0][0])

        expected_jvp01 = -2.5 * np.sin(x) * np.cos(y) - 1.5 * np.cos(x) * np.sin(y)
        assert qml.math.allclose(expected_jvp01, jvp[0][1])

        assert qml.math.allclose(jvp[1][0], 0)
        assert qml.math.allclose(jvp[1][1], -0.6 * np.sin(phi))

    def test_vjp_multi_params_multi_out(self, jpc):
        """Test compute_vjp with multiple parameters and multiple outputs."""

        x = 0.62
        y = 2.64
        ops = [qml.RY(y, 0), qml.RX(x, 0)]
        measurements = [qml.probs(wires=0), qml.expval(qml.PauliZ(0))]
        tape1 = qml.tape.QuantumScript(ops, measurements)

        phi = 0.623
        ops2 = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        measurements2 = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        tape2 = qml.tape.QuantumScript(ops2, measurements2)

        dy = (np.array([0.25, 0.5]), 1.5)
        dy2 = (0.7, 0.8)
        vjps = jpc.compute_vjp((tape1, tape2), (dy, dy2))

        dy = (
            0.5 * 0.25 * np.cos(x) * -np.sin(y)
            + 0.5 * 0.5 * np.cos(x) * np.sin(y)
            + 1.5 * np.cos(x) * -np.sin(y)
        )
        assert qml.math.allclose(vjps[0][0], dy)

        dx = (
            0.5 * 0.25 * -np.sin(x) * np.cos(y)
            + 0.5 * 0.5 * np.sin(x) * np.cos(y)
            + 1.5 * -np.sin(x) * np.cos(y)
        )
        assert qml.math.allclose(vjps[0][1], dx)

        assert qml.math.allclose(vjps[1], -0.8 * np.sin(phi))

    def test_jac_multi_params_multi_out(self, jpc):
        """Test compute_jacobian with multiple parameters and multiple measurements."""

        x = 0.62
        y = 2.64
        ops = [qml.RY(y, 0), qml.RX(x, 0)]
        measurements = [qml.probs(wires=0), qml.expval(qml.PauliZ(0))]
        tape1 = qml.tape.QuantumScript(ops, measurements)

        phi = 0.623
        ops2 = [qml.Hadamard(0), qml.IsingXX(phi, wires=(0, 1))]
        measurements2 = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        tape2 = qml.tape.QuantumScript(ops2, measurements2)

        jac = jpc.compute_jacobian((tape1, tape2))

        # first tape, first measurement, first parameters (y)
        expected = 0.5 * np.array([-np.cos(x) * np.sin(y), np.cos(x) * np.sin(y)])
        assert qml.math.allclose(jac[0][0][0], expected)

        # first tape, first measurement, second parameter (x)
        expected = 0.5 * np.array([-np.sin(x) * np.cos(y), np.sin(x) * np.cos(y)])
        assert qml.math.allclose(jac[0][0][1], expected)

        # first tape, second measurement, first parameter(y)
        expected = -np.cos(x) * np.sin(y)
        assert qml.math.allclose(jac[0][1][0], expected)
        # first tape, second measurement, second parameter (x)
        expected = -np.sin(x) * np.cos(y)
        assert qml.math.allclose(jac[0][1][1], expected)

        # second tape, first measurement, only parameter
        assert qml.math.allclose(jac[1][0], 0)
        # second tape, second measurement, only parameter
        assert qml.math.allclose(jac[1][1], -np.sin(phi))


class TestTransformsDifferentiability:
    """Tests that the transforms are differentiable if the inner execution is differentiable.

    Note that testing is only done for the required method for each ml framework.
    """

    @pytest.mark.jax
    def test_execute_jvp_jax(self):
        """Test that execute_and_compute_jvp is jittable and differentiable with jax."""
        import jax

        jpc = param_shift_jpc

        def f(x, tangents):
            tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
            return jpc.execute_and_compute_jvp((tape,), tangents)[1]

        x = jax.numpy.array(0.1)
        tangents = ((jax.numpy.array(0.5),),)

        res = f(x, tangents=tangents)
        assert qml.math.allclose(res, -tangents[0][0] * np.sin(x))

        jit_res = jax.jit(f)(x, tangents=tangents)
        assert qml.math.allclose(jit_res, -tangents[0][0] * np.sin(x))

        grad = jax.grad(f)(x, tangents=tangents)
        assert qml.math.allclose(grad, -tangents[0][0] * np.cos(x))

        tangent_grad = jax.grad(f, argnums=1)(x, tangents)
        assert qml.math.allclose(tangent_grad[0][0], -np.sin(x))

    @pytest.mark.autograd
    def test_vjp_autograd(self):
        """Test that the derivative of compute_vjp can be taken with autograd."""

        jpc = param_shift_jpc

        def f(x, dy):
            tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
            vjp = jpc.compute_vjp((tape,), (dy,))
            return vjp[0]

        x = qml.numpy.array(0.1)
        dy = qml.numpy.array(2.0)

        res = f(x, dy)
        assert qml.math.allclose(res, -dy * np.sin(x))

        dx, ddy = qml.grad(f)(x, dy)
        assert qml.math.allclose(dx, -dy * np.cos(x))
        assert qml.math.allclose(ddy, -np.sin(x))

    @pytest.mark.torch
    def test_vjp_torch(self):
        """Test that the derivative of compute_vjp can be taken with torch."""

        import torch

        jpc = param_shift_jpc

        def f(x, dy):
            tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
            vjp = jpc.compute_vjp((tape,), (dy,))
            return vjp[0]

        x = torch.tensor(0.1, requires_grad=True)
        dy = torch.tensor(2.0, requires_grad=True)

        res = f(x, dy)
        assert qml.math.allclose(res, -2.0 * np.sin(0.1))

        res.backward()
        assert qml.math.allclose(x.grad, -2.0 * np.cos(0.1))
        assert qml.math.allclose(dy.grad, -np.sin(0.1))

    @pytest.mark.tf
    def test_vjp_tf(self):
        """Test that the derivatives of compute_vjp can be taken with tensorflow."""

        import tensorflow as tf

        jpc = param_shift_jpc

        def f(x, dy):
            tape = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
            vjp = jpc.compute_vjp((tape,), (dy,))
            return vjp[0]

        x = tf.Variable(0.6, dtype=tf.float64)
        dy = tf.Variable(1.5, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = f(x, dy)

        assert qml.math.allclose(res, -1.5 * np.sin(0.6))

        dx, ddy = tape.gradient(res, (x, dy))

        assert qml.math.allclose(dx, -1.5 * np.cos(0.6))
        assert qml.math.allclose(ddy, -np.sin(0.6))
