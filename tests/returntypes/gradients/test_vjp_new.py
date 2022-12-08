# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
import numpy as vanilla_numpy


class TestComputeVJP:
    """Tests for the numeric computation of VJPs"""

    def test_compute_single_measurement_single_params(self):
        """Test that the correct VJP is returned"""
        dy = np.array(1.0)
        jac = np.array(0.2)

        vjp = qml.gradients.compute_vjp_single_new(dy, jac)

        assert isinstance(vjp, np.ndarray)
        assert np.allclose(vjp, 0.2)

    def test_compute_single_measurement_multi_dim_single_params(self):
        """Test that the correct VJP is returned"""
        dy = np.array([1, 2])
        jac = np.array([0.3, 0.3])

        vjp = qml.gradients.compute_vjp_single_new(dy, jac)

        assert isinstance(vjp, np.ndarray)
        assert np.allclose(vjp, 0.9)

    def test_compute_single_measurement_multiple_params(self):
        """Test that the correct VJP is returned"""
        dy = np.array(1.0)
        jac = tuple([np.array(0.1), np.array(0.2)])

        vjp = qml.gradients.compute_vjp_single_new(dy, jac)

        assert isinstance(vjp, np.ndarray)
        assert np.allclose(vjp, [0.1, 0.2])

    def test_compute_single_measurement_multi_dim_multiple_params(self):
        """Test that the correct VJP is returned"""
        dy = np.array([1.0, 0.5])
        jac = tuple([np.array([0.1, 0.3]), np.array([0.2, 0.4])])

        vjp = qml.gradients.compute_vjp_single_new(dy, jac)

        assert isinstance(vjp, np.ndarray)
        assert np.allclose(vjp, [0.25, 0.4])

    def test_compute_multiple_measurement_single_params(self):
        """Test that the correct VJP is returned"""
        dy = tuple([np.array([1.0]), np.array([1.0, 0.5])])
        jac = tuple([np.array([0.3]), np.array([0.2, 0.5])])

        vjp = qml.gradients.compute_vjp_multi_new(dy, jac)

        assert isinstance(vjp, np.ndarray)
        assert np.allclose(vjp, [0.75])

    def test_compute_multiple_measurement_multi_dim_single_params(self):
        """Test that the correct VJP is returned"""
        dy = tuple([np.array([1.0]), np.array([1.0, 0.5])])
        jac = tuple(
            [
                tuple([np.array([0.3]), np.array([0.4])]),
                tuple([np.array([0.2, 0.5]), np.array([0.3, 0.8])]),
            ]
        )

        vjp = qml.gradients.compute_vjp_multi_new(dy, jac)

        assert isinstance(vjp, np.ndarray)
        assert np.allclose(vjp, [0.75, 1.1])

    def test_compute_multiple_measurement_multi_dim_multiple_params(self):
        """Test that the correct VJP is returned"""
        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = np.array([[[1.0, 0.1, 0.2], [0.2, 0.6, 0.1]], [[0.4, -0.7, 1.2], [-0.5, -0.6, 0.7]]])

        vjp = qml.gradients.compute_vjp(dy, jac)

        assert vjp.shape == (3,)
        assert np.all(vjp == np.tensordot(dy, jac, axes=[[0, 1], [0, 1]]))

    def test_jacobian_is_none_single(self):
        """A None Jacobian returns a None VJP"""

        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = None

        vjp = qml.gradients.compute_vjp_single_new(dy, jac)
        assert vjp is None

    def test_jacobian_is_none_multi(self):
        """A None Jacobian returns a None VJP"""

        dy = np.array([[1.0, 2.0], [3.0, 4.0]])
        jac = None

        vjp = qml.gradients.compute_vjp_multi_new(dy, jac)
        assert vjp is None

    def test_zero_dy_single_measurement_single_params(self):
        """A zero dy vector will return a zero matrix"""
        dy = np.zeros([1])
        jac = np.array(0.1)

        vjp = qml.gradients.compute_vjp_single_new(dy, jac)
        assert np.all(vjp == np.zeros([1]))

    def test_zero_dy_single_measurement_multi_params(self):
        """A zero dy vector will return a zero matrix"""
        dy = np.zeros([2])
        jac = tuple([np.array(0.1), np.array(0.2)])

        vjp = qml.gradients.compute_vjp_single_new(dy, jac)
        assert np.all(vjp == np.zeros([2]))

    def test_zero_dy_multi(self):
        """A zero dy vector will return a zero matrix"""
        dy = tuple([np.array(0.0), np.array([0.0, 0.0])])
        jac = tuple(
            [
                tuple([np.array(0.1), np.array(0.1), np.array(0.1)]),
                tuple([np.array([0.1, 0.2]), np.array([0.1, 0.2]), np.array([0.1, 0.2])]),
            ]
        )

        vjp = qml.gradients.compute_vjp_multi_new(dy, jac)
        assert np.all(vjp == np.zeros([3]))

    @pytest.mark.torch
    @pytest.mark.parametrize("dtype1,dtype2", [("float32", "float64"), ("float64", "float32")])
    def test_dtype_torch(self, dtype1, dtype2):
        """Test that using the Torch interface the dtype of the result is
        determined by the dtype of the dy."""
        import torch

        dtype1 = getattr(torch, dtype1)
        dtype2 = getattr(torch, dtype2)
        a = torch.ones((1), dtype=dtype1)
        b = torch.ones((2), dtype=dtype1)

        dy = tuple([a, b])
        jac = tuple([torch.ones((1), dtype=dtype2), torch.ones((2), dtype=dtype2)])

        assert qml.gradients.compute_vjp_multi_new(dy, jac)[0].dtype == dtype1

    @pytest.mark.tf
    @pytest.mark.parametrize("dtype1,dtype2", [("float32", "float64"), ("float64", "float32")])
    def test_dtype_tf(self, dtype1, dtype2):
        """Test that using the TensorFlow interface the dtype of the result is
        determined by the dtype of the dy."""
        import tensorflow as tf

        dtype = dtype1
        dtype1 = getattr(tf, dtype1)
        dtype2 = getattr(tf, dtype2)

        a = tf.ones((1), dtype=dtype1)
        b = tf.ones((2), dtype=dtype1)

        dy = tuple([a, b])
        jac = tuple([tf.ones((1), dtype=dtype2), tf.ones((2), dtype=dtype2)])

        assert qml.gradients.compute_vjp_multi_new(dy, jac)[0].dtype == dtype

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

        dy = tuple([jax.numpy.array(1, dtype=dtype1), jax.numpy.array([1, 1], dtype=dtype1)])
        jac = tuple([jax.numpy.array(1, dtype=dtype2), jax.numpy.array([1, 1], dtype=dtype2)])
        assert qml.gradients.compute_vjp_multi_new(dy, jac)[0].dtype == dtype


class TestVJP:
    """Tests for the vjp function"""

    def test_no_trainable_parameters(self):
        """A tape with no trainable parameters will simply return None"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {}
        dy = np.array([1.0])
        tapes, fn = qml.gradients.vjp(tape, dy, param_shift)

        assert not tapes
        assert fn(tapes) is None

    def test_zero_dy(self):
        """A zero dy vector will return no tapes and a zero matrix"""

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RX(0.6, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = {0, 1}
        dy = np.array([0.0])
        tapes, fn = qml.gradients.vjp(tape, dy, param_shift)

        assert not tapes
        assert np.all(fn(tapes) == np.zeros([len(tape.trainable_params)]))

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
        dy = np.array(1.0)

        tapes, fn = qml.gradients.vjp(tape, dy, param_shift)
        assert len(tapes) == 4

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (2,)

        expected = np.array([-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)])
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
        dy = np.array([1.0, 2.0])

        tapes, fn = qml.gradients.vjp(tape, dy, param_shift)
        assert len(tapes) == 4

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (2,)

        expected = np.array([-np.sin(x), 2 * np.cos(y)])
        assert np.allclose(res, expected, atol=tol, rtol=0)

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
        dy = tuple([np.array(1.0), np.array([2.0, 3.0, 4.0, 5.0])])

        tapes, fn = qml.gradients.vjp(tape, dy, param_shift)
        assert len(tapes) == 4

        res = fn(dev.batch_execute(tapes))
        assert res.shape == (2,)

        expected = (
            np.array(
                [
                    [-2 * np.sin(x), 0],
                    [
                        -(np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        -(np.sin(x) * np.sin(y / 2) ** 2),
                        (np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.sin(x) * np.sin(y / 2) ** 2),
                        (np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                ]
            )
            / 2
        )
        dy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(res, dy @ expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_matches_dy(self, dtype):
        """Tests that the vjp function matches the dtype of dy when dy is
        zero-like."""
        x = np.array([0.1], dtype=np.float64)

        with qml.tape.QuantumTape() as tape:
            qml.RX(x[0], wires=0)
            qml.expval(qml.PauliZ(0))

        dy = np.zeros(3, dtype=dtype)
        _, func = qml.gradients.vjp(tape, dy, qml.gradients.param_shift)

        assert func([]).dtype == dtype


def expected(params):
    x, y = 1.0 * params
    return (
        np.array(
            [
                (np.cos(y / 2) ** 2 * np.sin(x)) + (np.cos(y / 2) ** 2 * np.sin(x)),
                (np.cos(x / 2) ** 2 * np.sin(y)) - (np.sin(x / 2) ** 2 * np.sin(y)),
            ]
        )
        / 2
    )


def ansatz(x, y):
    qml.RX(x, wires=[0])
    qml.RY(y, wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.probs(wires=[0, 1])


class TestVJPGradients:
    """Gradient tests for the vjp function"""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests that the output of the VJP transform
        can be differentiated using autograd."""
        dev = qml.device("default.qubit.autograd", wires=2)
        params = np.array([0.543, -0.654], requires_grad=True)

        def cost_fn(x, dy):
            with qml.tape.QuantumTape() as tape:
                ansatz(x[0], x[1])

            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.vjp(tape, dy, param_shift)
            vjp = fn(dev.batch_execute(tapes))
            return vjp

        dy = np.array([-1.0, 0.0, 0.0, 1.0], requires_grad=False)
        res = cost_fn(params, dy)
        assert np.allclose(res, expected(params), atol=tol, rtol=0)

        res = qml.jacobian(cost_fn)(params, dy)
        assert np.allclose(res, qml.jacobian(expected)(params), atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests that the output of the VJP transform
        can be differentiated using Torch."""
        import torch

        dev = qml.device("default.qubit", wires=2)

        params_np = np.array([0.543, -0.654], requires_grad=True)
        params = torch.tensor(params_np, requires_grad=True, dtype=torch.float64)
        dy = torch.tensor([-1.0, 0.0, 0.0, 1.0], dtype=torch.float64)

        with qml.tape.QuantumTape() as tape:
            ansatz(params[0], params[1])

        tape.trainable_params = {0, 1}
        tapes, fn = qml.gradients.vjp(tape, dy, param_shift)
        vjp = fn(qml.execute(tapes, dev, qml.gradients.param_shift, interface="torch"))

        assert np.allclose(vjp.detach(), expected(params.detach()), atol=tol, rtol=0)

        cost = vjp[0]
        cost.backward()

        exp = qml.jacobian(lambda x: expected(x)[0])(params_np)
        assert np.allclose(params.grad, exp, atol=tol, rtol=0)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_tf(self, tol):
        """Tests that the output of the VJP transform
        can be differentiated using TF."""
        import tensorflow as tf

        dev = qml.device("default.qubit.tf", wires=2)

        params_np = np.array([0.543, -0.654], requires_grad=True)
        params = tf.Variable(params_np, dtype=tf.float64)
        dy = tf.constant([-1.0, 0.0, 0.0, 1.0], dtype=tf.float64)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                ansatz(params[0], params[1])

            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.vjp(tape, dy, param_shift)
            vjp = fn(dev.batch_execute(tapes))

        assert np.allclose(vjp, expected(params), atol=tol, rtol=0)

        res = t.jacobian(vjp, params)
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
    @pytest.mark.slow
    def test_jax(self, tol):
        """Tests that the output of the VJP transform
        can be differentiated using JAX."""
        import jax
        from jax import numpy as jnp

        dev = qml.device("default.qubit.jax", wires=2)
        params_np = np.array([0.543, -0.654], requires_grad=True)
        params = jnp.array(params_np)

        @partial(jax.jit, static_argnums=1)
        def cost_fn(x):
            with qml.tape.QuantumTape() as tape:
                ansatz(x[0], x[1])
            dy = jax.numpy.array([-1.0, 0.0, 0.0, 1.0])
            tape.trainable_params = {0, 1}
            tapes, fn = qml.gradients.vjp(tape, dy, param_shift)
            vjp = fn(dev.batch_execute(tapes))
            return vjp

        res = cost_fn(params)
        assert np.allclose(res, expected(params), atol=tol, rtol=0)

        res = jax.jacobian(cost_fn, argnums=0)(params)
        exp = qml.jacobian(expected)(params_np)
        assert np.allclose(res, exp, atol=tol, rtol=0)


class TestBatchVJP:
    """Tests for the batch VJP function"""

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
        dys = [np.array(1.0), np.array(1.0)]

        v_tapes, fn = qml.gradients.batch_vjp(tapes, dys, param_shift)
        assert len(v_tapes) == 4

        # Even though there are 3 parameters, only two contribute
        # to the VJP, so only 2*2=4 quantum evals
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
        dys = [np.array(1.0), np.array(1.0)]

        v_tapes, fn = qml.gradients.batch_vjp(tapes, dys, param_shift)

        assert v_tapes == []
        assert fn([]) == [None, None]

    def test_zero_dy(self):
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
        dys = [np.array(0.0), np.array(1.0)]

        v_tapes, fn = qml.gradients.batch_vjp(tapes, dys, param_shift)
        res = fn(dev.batch_execute(v_tapes))

        # Even though there are 3 parameters, only two contribute
        # to the VJP, so only 2*2=4 quantum evals
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
        dys = [np.array(1.0), np.array(1.0)]

        v_tapes, fn = qml.gradients.batch_vjp(tapes, dys, param_shift, reduction="append")
        res = fn(dev.batch_execute(v_tapes))

        # Returned VJPs will be appended to a list, one vjp per tape
        assert len(res) == 2
        assert all(isinstance(r, np.ndarray) for r in res)
        assert all(len(r) == len(t.trainable_params) for t, r in zip(tapes, res))

    def test_reduction_extend(self):
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

        tape1.trainable_params = {0}
        tape2.trainable_params = {0, 1}

        tapes = [tape1, tape2]
        dys = [np.array(1.0), np.array(1.0)]

        v_tapes, fn = qml.gradients.batch_vjp(tapes, dys, param_shift, reduction="extend")
        res = fn(dev.batch_execute(v_tapes))

        # Returned VJPs will be extended into a list. Each element of the returned
        # list will correspond to a single input parameter of the combined
        # tapes.
        assert len(res) == sum(len(t.trainable_params) for t in tapes)
