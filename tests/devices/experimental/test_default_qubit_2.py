# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for default qubit 2."""

import pytest

import numpy as np

import pennylane as qml
from pennylane.devices.experimental import DefaultQubit2, ExecutionConfig


def test_name():
    """Tests the name of DefaultQubit2."""
    assert DefaultQubit2().name == "default.qubit.2"


def test_no_jvp_functionality():
    """Test that jvp is not support on DefaultQubit2."""
    dev = DefaultQubit2()

    assert not dev.supports_jvp(ExecutionConfig())

    with pytest.raises(NotImplementedError):
        dev.compute_jvp(qml.tape.QuantumScript(), (10, 10))

    with pytest.raises(NotImplementedError):
        dev.execute_and_compute_jvp(qml.tape.QuantumScript(), (10, 10))


def test_no_vjp_functionality():
    """Test that vjp is not support on DefaultQubit2."""
    dev = DefaultQubit2()

    assert not dev.supports_vjp(ExecutionConfig())

    with pytest.raises(NotImplementedError):
        dev.compute_vjp(qml.tape.QuantumScript(), (10.0, 10.0))

    with pytest.raises(NotImplementedError):
        dev.execute_and_compute_vjp(qml.tape.QuantumScript(), (10.0, 10.0))


def test_no_device_derivatives():
    """Test that DefaultQubit2 currently doesn't support device derivatives."""
    dev = DefaultQubit2()

    with pytest.raises(NotImplementedError):
        dev.compute_derivatives(qml.tape.QuantumScript())

    with pytest.raises(NotADirectoryError):
        dev.execute_and_compute_derivatives(qml.tape.QuantumScript())


class TestTracking:
    """Testing the tracking capabilities of DefaultQubit2."""

    def test_tracker_set_upon_initialization(self):
        """Test that a new tracker is intialized with each device."""
        assert DefaultQubit2.tracker is not DefaultQubit2().tracker

    def test_tracker_not_updated_if_not_active(self):
        """Test that the tracker is not updated if not active."""
        dev = DefaultQubit2()
        assert len(dev.tracker.totals) == 0

        dev.execute(qml.tape.QuantumScript())
        assert len(dev.tracker.totals) == 0
        assert len(dev.tracker.history) == 0

    def test_tracking_batch(self):
        """Test that the experimental default qubit integrates with the tracker."""

        qs = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))])

        dev = DefaultQubit2()
        with qml.Tracker(dev) as tracker:
            dev.execute(qs)
            dev.execute([qs, qs])  # and a second time

        assert tracker.history == {"batches": [1, 1], "executions": [1, 2]}
        assert tracker.totals == {"batches": 2, "executions": 3}
        assert tracker.latest == {"batches": 1, "executions": 2}


class TestSupportsDerivatives:
    """Test that DefaultQubit2 states what kind of derivatives it supports."""

    def test_supports_backprop(self):
        """Test that DefaultQubit2 says that it supports backpropagation."""
        dev = DefaultQubit2()
        config = ExecutionConfig(gradient_method="backprop")
        assert dev.supports_derivatives(config) == True

        qs = qml.tape.QuantumScript([], [qml.state()])
        assert dev.supports_derivatives(config, qs)

    @pytest.mark.parametrize(
        "gradient_method", ["parameter-shift", "finite-diff", "device", "adjoint"]
    )
    def test_doesnt_support_other_gradient_methods(self, gradient_method):
        """Test that DefaultQubit2 currently does not support other gradient methods natively."""
        dev = DefaultQubit2()
        config = ExecutionConfig(gradient_method=gradient_method)
        assert not dev.supports_derivatives(config)


class TestBasicCircuit:
    """Tests a basic circuit with one rx gate and two simple expectation values."""

    def test_basic_circuit_numpy(self):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )

        dev = DefaultQubit2()
        result = dev.execute(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

    @pytest.mark.autograd
    def test_autograd_results_and_backprop(self):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        dev = DefaultQubit2()

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return qml.numpy.array(dev.execute(qs))

        result = f(phi)
        expected = np.array([-np.sin(phi), np.cos(phi)])
        assert qml.math.allclose(result, expected)

        g = qml.jacobian(f)(phi)
        expected = np.array([-np.cos(phi), -np.sin(phi)])
        assert qml.math.allclose(g, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_results_and_backprop(self, use_jit):
        """Tests exeuction and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        dev = DefaultQubit2()

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return dev.execute(qs)

        if use_jit:
            f = jax.jit(f)

        result = f(phi)
        assert qml.math.allclose(result[0], -np.sin(phi))
        assert qml.math.allclose(result[1], np.cos(phi))

        g = jax.jacobian(f)(phi)
        assert qml.math.allclose(g[0], -np.cos(phi))
        assert qml.math.allclose(g[1], -np.sin(phi))

    @pytest.mark.torch
    def test_torch_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with torch."""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        dev = DefaultQubit2()

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return dev.execute(qs)

        result = f(phi)
        assert qml.math.allclose(result[0], -torch.sin(phi))
        assert qml.math.allclose(result[1], torch.cos(phi))

        g = torch.autograd.functional.jacobian(f, phi + 0j)
        assert qml.math.allclose(g[0], -torch.cos(phi))
        assert qml.math.allclose(g[1], -torch.sin(phi))

    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    def test_tf_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)

        dev = DefaultQubit2()

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = qml.tape.QuantumScript(
                [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            result = dev.execute(qs)

        assert qml.math.allclose(result[0], -tf.sin(phi))
        assert qml.math.allclose(result[1], tf.cos(phi))

        grad0 = grad_tape.jacobian(result[0], [phi])
        grad1 = grad_tape.jacobian(result[1], [phi])

        assert qml.math.allclose(grad0[0], -tf.cos(phi))
        assert qml.math.allclose(grad1[0], -tf.sin(phi))


class TestExecutingBatches:
    def qs1(self, phi):
        """Circuit1."""
        ops = [
            qml.PauliX("a"),
            qml.PauliX("b"),
            qml.ctrl(qml.RX(phi, "target") ** 2, ("a", "b", -3), control_values=[1, 1, 0]),
        ]

        return qml.tape.QuantumScript(
            ops,
            [
                qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
            ],
        )

    def expected1(self, phi):
        return (-np.sin(2 * phi) - 1, 3 * np.sin(2 * phi))

    def qs2(self, theta):
        """Circuit 2."""
        ops = [qml.Hadamard(0), qml.IsingXX(theta, wires=(0, 1))]
        return qml.tape.QuantumScript(ops, [qml.density_matrix(1)])

    def expected2(self, theta):
        return np.array(
            [
                [np.cos(theta / 2) ** 2, 0.5j * np.sin(theta)],
                [-0.5j * np.sin(theta), np.sin(theta / 2) ** 2],
            ]
        )

    def test_numpy(self):

        dev = DefaultQubit2()
        phi = 0.123
        theta = 0.623

        results = dev.execute((self.qs1(phi), self.qs2(phi)))
        assert len(results) == 2

        expected1 = self.expected1(phi)
        assert qml.math.allclose(results[0][0], expected1[0])
        assert qml.math.allclose(results[0][1], expected1[1])

        expected2 = self.expected2(theta)
        assert qml.math.allclose(results[1], expected2)
