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
"""Tests for the gradients.vjp module."""
from functools import partial

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import param_shift


class TestComputeJVP:
    """Tests for the numeric computation of JVPs"""

    def test_compute_single_measurement_single_params(self):
        """Test that the correct JVP is returned"""
        tangent = np.array([1.0])
        jac = np.array(0.2)

        jvp = qml.gradients.compute_jvp_single(tangent, jac)

        assert isinstance(jvp, np.ndarray)
        assert np.allclose(jvp, 0.2)

    def test_compute_single_measurement_multi_dim_single_params(self):
        """Test that the correct JVP is returned"""
        tangent = np.array([2.0])
        jac = np.array([0.3, 0.3])

        jvp = qml.gradients.compute_jvp_single(tangent, jac)

        assert isinstance(jvp, np.ndarray)
        assert np.allclose(jvp, np.array([0.6, 0.6]))

    def test_compute_single_measurement_multiple_params(self):
        """Test that the correct JVP is returned"""
        tangent = np.array([1.0, 2.0])
        jac = tuple([np.array(0.1), np.array(0.2)])

        jvp = qml.gradients.compute_jvp_single(tangent, jac)

        assert isinstance(jvp, np.ndarray)
        assert np.allclose(jvp, 0.5)

    def test_compute_single_measurement_multi_dim_multiple_params(self):
        """Test that the correct JVP is returned"""
        tangent = np.array([1.0, 0.5])
        jac = tuple([np.array([0.1, 0.3]), np.array([0.2, 0.4])])

        jvp = qml.gradients.compute_jvp_single(tangent, jac)

        assert isinstance(jvp, np.ndarray)
        assert np.allclose(jvp, [0.2, 0.5])

    def test_compute_multiple_measurement_single_params(self):
        """Test that the correct JVP is returned"""
        tangent = np.array([2.0])
        jac = tuple([np.array([0.3]), np.array([0.2, 0.5])])

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)
        assert isinstance(jvp, tuple)
        assert len(jvp) == 2

        assert isinstance(jvp[0], np.ndarray)
        assert np.allclose(jvp[0], [0.6])

        assert isinstance(jvp[1], np.ndarray)
        assert np.allclose(jvp[1], [0.4, 1.0])

    def test_compute_multiple_measurement_multi_dim_multiple_params(self):
        """Test that the correct JVP is returned"""
        tangent = np.array([1.0, 2.0])

        jac = tuple(
            [
                tuple([np.array([0.3]), np.array([0.4])]),
                tuple([np.array([0.2, 0.5]), np.array([0.3, 0.8])]),
            ]
        )

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)

        assert isinstance(jvp, tuple)
        assert len(jvp) == 2

        assert isinstance(jvp[0], np.ndarray)
        assert np.allclose(jvp[0], [1.1])

        assert isinstance(jvp[1], np.ndarray)
        assert np.allclose(jvp[1], [0.8, 2.1])

    def test_jacobian_is_none_single(self):
        """A None Jacobian returns a None JVP"""

        tangent = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = None

        jvp = qml.gradients.compute_jvp_single(tangent, jac)
        assert jvp is None

    def test_jacobian_is_none_multi(self):
        """A None Jacobian returns a None JVP"""

        tangent = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = None

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)
        assert jvp is None

    def test_zero_tangent_single_measurement_single_params(self):
        """A zero dy vector will return a zero matrix"""
        tangent = np.zeros([1])
        jac = np.array(0.1)

        jvp = qml.gradients.compute_jvp_single(tangent, jac)
        assert np.all(jvp == np.zeros([1]))

    def test_zero_tangent_single_measurement_multi_params(self):
        """A zero tangent vector will return a zero matrix"""
        tangent = np.zeros([2])
        jac = tuple([np.array(0.1), np.array(0.2)])

        jvp = qml.gradients.compute_jvp_single(tangent, jac)
        assert np.all(jvp == np.zeros([2]))

    def test_zero_dy_multi(self):
        """A zero tangent vector will return a zero matrix"""
        tangent = np.array([0.0, 0.0, 0.0])
        jac = tuple(
            [
                tuple([np.array(0.1), np.array(0.1), np.array(0.1)]),
                tuple([np.array([0.1, 0.2]), np.array([0.1, 0.2]), np.array([0.1, 0.2])]),
            ]
        )

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)

        assert isinstance(jvp, tuple)
        assert np.all(jvp[0] == np.zeros([1]))
        assert np.all(jvp[1] == np.zeros([2]))

    @pytest.mark.jax
    @pytest.mark.parametrize("dtype1,dtype2", [("float32", "float64"), ("float64", "float32")])
    def test_dtype_jax(self, dtype1, dtype2):
        """Test that using the JAX interface the dtype of the result is
        determined by the dtype of the dy."""
        import jax
        from jax.config import config

        config.update("jax_enable_x64", True)

        dtype = dtype1
        dtype1 = getattr(jax.numpy, dtype1)
        dtype2 = getattr(jax.numpy, dtype2)

        tangent = jax.numpy.array([1], dtype=dtype1)
        jac = tuple([jax.numpy.array(1, dtype=dtype2), jax.numpy.array([1, 1], dtype=dtype2)])
        assert qml.gradients.compute_jvp_multi(tangent, jac)[0].dtype == dtype

    def test_no_trainable_params_adjoint_single(self):
        """An empty jacobian return empty array."""
        tangent = np.array([1.0, 2.0])
        jac = tuple()

        jvp = qml.gradients.compute_jvp_single(tangent, jac)
        assert np.allclose(jvp, qml.math.zeros(0))

    def test_no_trainable_params_adjoint_multi_measurement(self):
        """An empty jacobian return an empty tuple."""
        tangent = np.array([1.0, 2.0])
        jac = tuple()

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)
        assert isinstance(jvp, tuple)
        assert len(jvp) == 0

    def test_no_trainable_params_gradient_transform_single(self):
        """An empty jacobian return empty array."""
        tangent = np.array([1.0, 2.0])
        jac = qml.math.zeros(0)

        jvp = qml.gradients.compute_jvp_single(tangent, jac)
        assert np.allclose(jvp, qml.math.zeros(0))

    def test_no_trainable_params_gradient_transform_multi_measurement(self):
        """An empty jacobian return an empty tuple."""
        tangent = np.array([1.0, 2.0])
        jac = tuple([qml.math.zeros(0), qml.math.zeros(0)])

        jvp = qml.gradients.compute_jvp_multi(tangent, jac)
        assert isinstance(jvp, tuple)
        assert len(jvp) == 2
        for j in jvp:
            assert np.allclose(j, qml.math.zeros(0))


class TestJVP:
    """Tests for the jvp function"""

    def test_no_trainable_parameters(self):
        """A tape with no trainable parameters will simply return None"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        tangent = np.array([1.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)

        assert tapes == []
        assert fn(tapes) is None

    def test_zero_tangent_single_measurement_single_param(self):
        """A zero tangent vector will return no tapes and a zero matrix"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0}
        tangent = np.array([0.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)

        assert tapes == []
        assert np.all(fn(tapes) == np.zeros([1]))

    def test_zero_tangent_single_measurement_multiple_param(self):
        """A zero tangent vector will return no tapes and a zero matrix"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0, 1}
        tangent = np.array([0.0, 0.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)

        assert tapes == []
        assert isinstance(fn(tapes), tuple)
        assert np.all(fn(tapes) == np.zeros([1]))

    def test_zero_tangent_multiple_measurement_single_param(self):
        """A zero tangent vector will return no tapes and a zero matrix"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0])

        tape.trainable_params = {0}
        tangent = np.array([0.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        res = fn(tapes)

        assert tapes == []

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert np.allclose(res[0], 0)

        assert isinstance(res[1], np.ndarray)
        assert np.allclose(res[1], [0, 0])

    def test_zero_tangent_multiple_measurement_multiple_param(self):
        """A zero tangent vector will return no tapes and a zero matrix"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0])

        tape.trainable_params = {0, 1}
        tangent = np.array([0.0, 0.0])
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        res = fn(tapes)

        assert tapes == []

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert np.allclose(res[0], 0)

        assert isinstance(res[1], np.ndarray)
        assert np.allclose(res[1], [0, 0])

    def test_single_expectation_value(self, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape.trainable_params = {0, 1}
        tangent = np.array([1.0, 1.0])

        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        assert len(tapes) == 4

        res = fn(dev.batch_execute(tapes))
        assert res.shape == ()

        expected = np.sum(np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        tape.trainable_params = {0, 1}
        tangent = np.array([1.0, 2.0])

        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        assert len(tapes) == 4

        res = fn(dev.batch_execute(tapes))
        assert isinstance(res, tuple)

        expected = np.array([-np.sin(x), 2 * np.cos(y)])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values_single_param(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=0)

        tape.trainable_params = {0}
        tangent = np.array([1.0])

        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        assert len(tapes) == 2

        res = fn(dev.batch_execute(tapes))
        assert isinstance(res, tuple)

        expected_0 = -np.sin(x)
        assert np.allclose(res[0], expected_0, atol=tol, rtol=0)

        expected_1 = [-np.sin(x) / 2, np.sin(x) / 2]
        assert np.allclose(res[1], expected_1, atol=tol, rtol=0)

    def test_prob_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        tape.trainable_params = {0, 1}
        tangent = np.array([1.0, 1.0])

        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        assert len(tapes) == 4

        res = fn(dev.batch_execute(tapes))
        assert isinstance(res, tuple)
        assert len(res) == 2

        expected = (
            np.array(
                [
                    -2 * np.sin(x),
                    [
                        -(np.cos(y / 2) ** 2 * np.sin(x)) - (np.cos(x / 2) ** 2 * np.sin(y)),
                        -(np.sin(x) * np.sin(y / 2) ** 2) + (np.cos(x / 2) ** 2 * np.sin(y)),
                        (np.sin(x) * np.sin(y / 2) ** 2) + (np.sin(x / 2) ** 2 * np.sin(y)),
                        (np.cos(y / 2) ** 2 * np.sin(x)) - (np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                ]
            )
            / 2
        )

        assert np.allclose(res[0], expected[0], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[1], atol=tol, rtol=0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_matches_tangent(self, dtype):
        """Tests that the vjp function matches the dtype of tangent when tangent is
        zero-like."""
        x = np.array([0.1], dtype=np.float64)

        with qml.tape.QuantumTape() as tape:
            qml.RX(x[0], wires=0)
            qml.expval(qml.PauliZ(0))

        dy = np.zeros(1, dtype=dtype)
        _, func = qml.gradients.jvp(tape, dy, qml.gradients.param_shift)

        assert func([]).dtype == dtype


def expected(params):
    x, y = 1.0 * params
    return np.array([-np.sin(x / 2) * np.cos(x / 2), 0.0, 0.0, np.sin(x / 2) * np.cos(x / 2)])


def ansatz(x, y):
    qml.RX(x, wires=[0])
    qml.RZ(y, wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.probs(wires=[0, 1])


class TestJVPGradients:
    """Gradient tests for the jvp function"""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests that the output of the JVP transform
        can be differentiated using autograd."""
        dev = qml.device("default.qubit.autograd", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x, tangent):
            with qml.tape.QuantumTape() as tape:
                ansatz(x[0], x[1])

            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
            jvp = fn(dev.batch_execute(tapes))
            return jvp

        tangent = np.array([1.0, 0.3], requires_grad=False)
        res = cost_fn(params, tangent)
        assert np.allclose(res, expected(params), atol=tol, rtol=0)

        res = qml.jacobian(cost_fn)(params, tangent)
        assert np.allclose(res, qml.jacobian(expected)(params), atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests that the output of the JVP transform
        can be differentiated using Torch."""
        import torch

        dev = qml.device("default.qubit", wires=2)

        params_np = np.array([0.543, -0.654], requires_grad=True)
        params = torch.tensor(params_np, requires_grad=True, dtype=torch.float64)
        tangent = torch.tensor([1.0, 0.0], dtype=torch.float64)

        with qml.tape.QuantumTape() as tape:
            ansatz(params[0], params[1])

        tape.trainable_params = {0, 1}
        tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
        jvp = fn(qml.execute(tapes, dev, qml.gradients.param_shift, interface="torch"))

        assert np.allclose(jvp.detach(), expected(params.detach()), atol=tol, rtol=0)

        cost = jvp[0]
        cost.backward()

        exp = qml.jacobian(lambda x: expected(x)[0])(params_np)
        assert np.allclose(params.grad, exp, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_tf(self, tol):
        """Tests that the output of the JVP transform
        can be differentiated using TF."""
        import tensorflow as tf

        dev = qml.device("default.qubit.tf", wires=2)

        params_np = np.array([0.543, -0.654], requires_grad=True)
        params = tf.Variable(params_np, dtype=tf.float64)
        tangent = tf.constant(
            [
                1.0,
                0.0,
            ],
            dtype=tf.float64,
        )

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                ansatz(params[0], params[1])

            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
            jvp = fn(dev.batch_execute(tapes))
        assert np.allclose(jvp, expected(params), atol=tol, rtol=0)

        res = t.jacobian(jvp, params)
        assert np.allclose(res, qml.jacobian(expected)(params_np), atol=tol, rtol=0)

    # TODO: to be added when lighting and TF compatible with return types
    # @pytest.mark.tf
    # def test_tf_custom_loss(self):
    #     """Tests that the gradient pipeline using the TensorFlow interface with
    #     a custom TF loss and lightning.qubit with a custom dtype does not raise
    #     any errors."""
    #     import tensorflow as tf

    #     nwires = 5
    #     dev = qml.device("lightning.qubit", wires=nwires)
    #     dev.C_DTYPE = vanilla_numpy.complex64
    #     dev.R_DTYPE = vanilla_numpy.float32

    #     @qml.qnode(dev, interface="tf", diff_method="adjoint")
    #     def circuit(weights, features):
    #         for i in range(nwires):
    #            qml.RX(features[i], wires=i)
    #             qml.RX(weights[i], wires=i)
    #         return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    #     vanilla_numpy.random.seed(42)

    #     ndata = 100
    #     data = [vanilla_numpy.random.randn(nwires).astype("float32") for _ in range(ndata)]
    #     label = [vanilla_numpy.random.choice([1, 0]).astype("int") for _ in range(ndata)]

    #     loss = tf.losses.SparseCategoricalCrossentropy()

    #     params = tf.Variable(vanilla_numpy.random.randn(nwires).astype("float32"), trainable=True)
    #     with tf.GradientTape() as tape:
    #         probs = [circuit(params, d) for d in data]
    #         loss_value = loss(label, probs)

    #    grads = tape.gradient(loss_value, [params])
    #    assert len(grads) == 1

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests that the output of the VJP transform
        can be differentiated using JAX."""
        import jax
        from jax import numpy as jnp

        dev = qml.device("default.qubit.jax", wires=2)
        params_np = np.array([0.543, -0.654], requires_grad=True)
        params = jnp.array(params_np)

        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
                ansatz(x[0], x[1])
            tangent = jax.numpy.array([1.0, 0.0])
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.jvp(tape, tangent, param_shift)
            jvp = fn(dev.batch_execute(tapes))
            return jvp

        res = cost_fn(params)
        assert np.allclose(res, expected(params), atol=tol, rtol=0)

        res = jax.jacobian(cost_fn, argnums=0)(params)
        exp = qml.jacobian(expected)(params_np)
        assert np.allclose(res, exp, atol=tol, rtol=0)


class TestBatchJVP:
    """Tests for the batch JVP function"""

    def test_one_tape_no_trainable_parameters(self):
        """A tape with no trainable parameters will simply return None"""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = {}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        tangents = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, param_shift)
        assert len(v_tapes) == 4

        # Even though there are 3 parameters, only two contribute
        # to the JVP, so only 2*2=4 quantum evals
        res = fn(dev.batch_execute(v_tapes))

        assert res[0] is None
        assert res[1] is not None

    def test_all_tapes_no_trainable_parameters(self):
        """If all tapes have no trainable parameters all outputs will be None"""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = set()
        tape2.trainable_params = set()

        tapes = [tape1, tape2]
        tangents = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]

        v_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, param_shift)

        assert v_tapes == []
        assert fn([]) == [None, None]

    def test_zero_tangent(self):
        """A zero dy vector will return no tapes and a zero matrix"""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        tangents = [np.array([0.0]), np.array([1.0, 1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, param_shift)
        res = fn(dev.batch_execute(v_tapes))

        # Even though there are 3 parameters, only two contribute
        # to the JVP, so only 2*2=4 quantum evals
        assert len(v_tapes) == 4
        assert np.allclose(res[0], 0)

    def test_reduction_append(self):
        """Test the 'append' reduction strategy"""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        tangents = [np.array([1.0]), np.array([1.0, 1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, param_shift, reduction="append")
        res = fn(dev.batch_execute(v_tapes))

        # Returned JVPs will be appended to a list, one JVP per tape
        assert len(res) == 2
        assert all(isinstance(r, np.ndarray) for r in res)
        assert res[0].shape == ()
        assert res[1].shape == ()

    def test_reduction_extend(self):
        """Test the 'extend' reduction strategy"""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=1)

        tape1.trainable_params = {0}
        tape2.trainable_params = {1}

        tapes = [tape1, tape2]
        tangents = [np.array([1.0]), np.array([1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(tapes, tangents, param_shift, reduction="extend")
        res = fn(dev.batch_execute(v_tapes))
        assert len(res) == 4

    def test_reduction_extend_special(self):
        """Test the 'extend' reduction strategy"""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=0)

        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        tangents = [np.array([1.0]), np.array([1.0, 1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(
            tapes,
            tangents,
            param_shift,
            reduction=lambda jvps, x: jvps.extend(qml.math.reshape(x, (1,)))
            if not isinstance(x, tuple) and x.shape == ()
            else jvps.extend(x),
        )
        res = fn(dev.batch_execute(v_tapes))

        assert len(res) == 3

    def test_reduction_callable(self):
        """Test the callable reduction strategy"""
        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape1:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=0)

        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        tangents = [np.array([1.0]), np.array([1.0, 1.0])]

        v_tapes, fn = qml.gradients.batch_jvp(
            tapes, tangents, param_shift, reduction=lambda jvps, x: jvps.append(x)
        )
        res = fn(dev.batch_execute(v_tapes))
        # Returned JVPs will be appended to a list, one JVP per tape
        assert len(res) == 2
