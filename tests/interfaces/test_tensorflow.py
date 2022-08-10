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
"""Unit tests for the TensorFlow interface"""
import functools

import numpy as np
import pytest

import pennylane as qml
from pennylane.gradients import finite_diff, param_shift
from pennylane.interfaces import execute

pytestmark = pytest.mark.tf

tf = pytest.importorskip("tensorflow", minversion="2.1")


class TestTensorFlowExecuteUnitTests:
    """Unit tests for autograd execution"""

    def test_jacobian_options(self, mocker, tol):
        """Test setting jacobian options"""
        spy = mocker.spy(qml.gradients, "param_shift")

        a = tf.Variable([0.1, 0.2], dtype=tf.float64)

        dev = qml.device("default.qubit", wires=1)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            res = execute(
                [tape],
                dev,
                gradient_fn=param_shift,
                gradient_kwargs={"shifts": [(np.pi / 4,)] * 2},
                interface="tf",
            )[0]

        res = t.jacobian(res, a)

        for args in spy.call_args_list:
            assert args[1]["shifts"] == [(np.pi / 4,)] * 2

    def test_incorrect_mode(self):
        """Test that an error is raised if a gradient transform
        is used with mode=forward"""
        a = tf.Variable([0.1, 0.2])

        dev = qml.device("default.qubit", wires=1)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

        with pytest.raises(
            ValueError, match="Gradient transforms cannot be used with mode='forward'"
        ):
            res = execute([tape], dev, gradient_fn=param_shift, mode="forward", interface="tf")[0]

    def test_forward_mode(self, mocker):
        """Test that forward mode uses the `device.execute_and_gradients` pathway"""
        dev = qml.device("default.qubit", wires=1)
        a = tf.Variable([0.1, 0.2])
        spy = mocker.spy(dev, "execute_and_gradients")

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            res = execute(
                [tape],
                dev,
                gradient_fn="device",
                gradient_kwargs={"method": "adjoint_jacobian", "use_device_state": True},
                interface="tf",
            )[0]

        # adjoint method only performs a single device execution, but gets both result and gradient
        assert dev.num_executions == 1
        spy.assert_called()

    def test_backward_mode(self, mocker):
        """Test that backward mode uses the `device.batch_execute` and `device.gradients` pathway"""
        dev = qml.device("default.qubit", wires=1)
        spy_execute = mocker.spy(qml.devices.DefaultQubit, "batch_execute")
        spy_gradients = mocker.spy(qml.devices.DefaultQubit, "gradients")
        a = tf.Variable([0.1, 0.2])

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.expval(qml.PauliZ(0))

            res = execute(
                [tape],
                dev,
                gradient_fn="device",
                mode="backward",
                gradient_kwargs={"method": "adjoint_jacobian"},
                interface="tf",
            )[0]

        assert dev.num_executions == 1
        spy_execute.assert_called()
        spy_gradients.assert_not_called()

        j = t.jacobian(res, a)
        spy_gradients.assert_called()


class TestCaching:
    """Test for caching behaviour"""

    def test_cache_maxsize(self, mocker):
        """Test the cachesize property of the cache"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(qml.interfaces, "cache_execute")
        a = tf.Variable([0.1, 0.2])

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.probs(wires=0)

            res = execute([tape], dev, gradient_fn=param_shift, cachesize=2, interface="tf")[0]

        t.jacobian(res, a)
        cache = spy.call_args[0][1]

        assert cache.maxsize == 2
        assert cache.currsize == 2
        assert len(cache) == 2

    def test_custom_cache(self, mocker):
        """Test the use of a custom cache object"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(qml.interfaces, "cache_execute")
        a = tf.Variable([0.1, 0.2])
        custom_cache = {}

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.probs(wires=0)

            res = execute([tape], dev, gradient_fn=param_shift, cache=custom_cache, interface="tf")[
                0
            ]

        t.jacobian(res, a)

        cache = spy.call_args[0][1]
        assert cache is custom_cache

    def test_caching_param_shift(self, tol):
        """Test that, when using parameter-shift transform,
        caching reduces the number of evaluations to their optimum."""
        dev = qml.device("default.qubit", wires=1)
        a = tf.Variable([0.1, 0.2], dtype=tf.float64)

        def cost(a, cache):
            with qml.tape.QuantumTape() as tape:
                qml.RY(a[0], wires=0)
                qml.RX(a[1], wires=0)
                qml.probs(wires=0)

            return execute([tape], dev, gradient_fn=param_shift, cache=cache, interface="tf")[0]

        # Without caching, and non-vectorized, 9 evaluations are required to compute
        # the Jacobian: 1 (forward pass) + 2 (backward pass) * (2 shifts * 2 params)
        with tf.GradientTape(persistent=True) as t:
            res = cost(a, cache=None)
        t.jacobian(res, a, experimental_use_pfor=False)
        assert dev.num_executions == 9

        # With caching, and non-vectorized, 5 evaluations are required to compute
        # the Jacobian: 1 (forward pass) + (2 shifts * 2 params)
        dev._num_executions = 0
        with tf.GradientTape(persistent=True) as t:
            res = cost(a, cache=True)
        t.jacobian(res, a)
        assert dev.num_executions == 5

        # In vectorized mode, 5 evaluations are required to compute
        # the Jacobian regardless of caching: 1 (forward pass) + (2 shifts * 2 params)
        dev._num_executions = 0
        with tf.GradientTape() as t:
            res = cost(a, cache=None)
        t.jacobian(res, a)
        assert dev.num_executions == 5

    @pytest.mark.parametrize("num_params", [2, 3])
    def test_caching_param_shift_hessian(self, num_params, tol):
        """Test that, when using parameter-shift transform,
        caching reduces the number of evaluations to their optimum
        when computing Hessians."""
        dev = qml.device("default.qubit", wires=2)
        params = tf.Variable(np.arange(1, num_params + 1) / 10, dtype=tf.float64)

        N = params.shape[0]

        def cost(x, cache):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x[0], wires=[0])
                qml.RY(x[1], wires=[1])

                for i in range(2, num_params):
                    qml.RZ(x[i], wires=[i % 2])

                qml.CNOT(wires=[0, 1])
                qml.var(qml.PauliZ(0) @ qml.PauliX(1))

            return execute(
                [tape], dev, gradient_fn=param_shift, cache=cache, interface="tf", max_diff=2
            )[0]

        # No caching: number of executions is not ideal
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                res = cost(params, cache=False)
            grad = t1.gradient(res, params)
        hess1 = t2.jacobian(grad, params)

        if num_params == 2:
            # compare to theoretical result
            x, y, *_ = params * 1.0
            expected = np.array(
                [
                    [2 * np.cos(2 * x) * np.sin(y) ** 2, np.sin(2 * x) * np.sin(2 * y)],
                    [np.sin(2 * x) * np.sin(2 * y), -2 * np.cos(x) ** 2 * np.cos(2 * y)],
                ]
            )
            assert np.allclose(expected, hess1, atol=tol, rtol=0)

        nonideal_runs = dev.num_executions

        # Use caching: number of executions is ideal
        dev._num_executions = 0
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                res = cost(params, cache=True)
            grad = t1.gradient(res, params)
        hess2 = t2.jacobian(grad, params)

        assert np.allclose(hess1, hess2, atol=tol, rtol=0)

        expected_runs_ideal = 1  # forward pass
        expected_runs_ideal += 2 * N  # Jacobian
        expected_runs_ideal += N + 1  # Hessian diagonal
        expected_runs_ideal += 4 * N * (N - 1) // 2  # Hessian off-diagonal
        assert dev.num_executions == expected_runs_ideal
        assert expected_runs_ideal < nonideal_runs


execute_kwargs = [
    {"gradient_fn": param_shift, "interface": "tf"},
    {
        "gradient_fn": "device",
        "mode": "forward",
        "gradient_kwargs": {"method": "adjoint_jacobian", "use_device_state": True},
        "interface": "tf",
    },
    {
        "gradient_fn": "device",
        "mode": "backward",
        "gradient_kwargs": {"method": "adjoint_jacobian"},
        "interface": "tf",
    },
]


@pytest.mark.parametrize("execute_kwargs", execute_kwargs)
class TestTensorFlowExecuteIntegration:
    """Test the TensorFlow interface execute function
    integrates well for both forward and backward execution"""

    def test_execution(self, execute_kwargs):
        """Test execution"""
        dev = qml.device("default.qubit", wires=1)
        a = tf.Variable(0.1)
        b = tf.Variable(0.2)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape1:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            with qml.tape.QuantumTape() as tape2:
                qml.RY(a, wires=0)
                qml.RX(b, wires=0)
                qml.expval(qml.PauliZ(0))

            res = execute([tape1, tape2], dev, **execute_kwargs)

        assert len(res) == 2
        assert res[0].shape == (1,)
        assert res[1].shape == (1,)

    def test_scalar_jacobian(self, execute_kwargs, tol):
        """Test scalar jacobian calculation"""
        a = tf.Variable(0.1, dtype=tf.float64)
        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.expval(qml.PauliZ(0))
            res = execute([tape], dev, **execute_kwargs)[0]

        res = t.jacobian(res, a)
        assert res.shape == (1,)

        # compare to standard tape jacobian
        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.expval(qml.PauliZ(0))

        tape.trainable_params = [0]
        tapes, fn = param_shift(tape)
        expected = fn(dev.batch_execute(tapes))

        assert expected.shape == (1, 1)
        assert np.allclose(res, np.squeeze(expected), atol=tol, rtol=0)

    def test_jacobian(self, execute_kwargs, tol):
        """Test jacobian calculation"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)
        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliY(1))
            res = execute([tape], dev, max_diff=2, **execute_kwargs)[0]

        expected = [np.cos(a), -np.cos(a) * np.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        (agrad, bgrad) = t.jacobian(res, [a, b])
        assert agrad.shape == (2,)
        assert bgrad.shape == (2,)

        expected = [[-np.sin(a), np.sin(a) * np.sin(b)], [0, -np.cos(a) * np.cos(b)]]
        assert np.allclose(expected, [agrad, bgrad], atol=tol, rtol=0)

    def test_tape_no_parameters(self, execute_kwargs, tol):
        """Test that a tape with no parameters is correctly
        ignored during the gradient computation"""
        dev = qml.device("default.qubit", wires=1)
        params = tf.Variable([0.1, 0.2], dtype=tf.float64)
        x, y = 1.0 * params

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape1:
                qml.Hadamard(0)
                qml.expval(qml.PauliX(0))

            with qml.tape.QuantumTape() as tape2:
                qml.RY(0.5, wires=0)
                qml.expval(qml.PauliZ(0))

            with qml.tape.QuantumTape() as tape3:
                qml.RY(params[0], wires=0)
                qml.RX(params[1], wires=0)
                qml.expval(qml.PauliZ(0))

            res = sum(execute([tape1, tape2, tape3], dev, **execute_kwargs))

        expected = 1 + np.cos(0.5) + np.cos(x) * np.cos(y)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = t.gradient(res, params)
        expected = [-np.cos(y) * np.sin(x), -np.cos(x) * np.sin(y)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_reusing_quantum_tape(self, execute_kwargs, tol):
        """Test re-using a quantum tape by passing new parameters"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(a, wires=0)
                qml.RX(b, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliY(1))

            assert tape.trainable_params == [0, 1]
            res = execute([tape], dev, **execute_kwargs)[0]

        jac = t.jacobian(res, [a, b])

        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.Variable(0.8, dtype=tf.float64)

        # check that the cost function continues to depend on the
        # values of the parameters for subsequent calls
        with tf.GradientTape() as t:
            tape.set_parameters([2 * a, b])
            res2 = execute([tape], dev, **execute_kwargs)[0]

        expected = [tf.cos(2 * a), -tf.cos(2 * a) * tf.sin(b)]
        assert np.allclose(res2, expected, atol=tol, rtol=0)

        jac2 = t.jacobian(res2, [a, b])
        expected = [
            [-2 * tf.sin(2 * a), 2 * tf.sin(2 * a) * tf.sin(b)],
            [0, -tf.cos(2 * a) * tf.cos(b)],
        ]
        assert np.allclose(jac2, expected, atol=tol, rtol=0)

    def test_reusing_pre_constructed_quantum_tape(self, execute_kwargs, tol):
        """Test re-using a quantum tape that was previously constructed
        *outside of* a gradient tape, by passing new parameters"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliY(1))

        with tf.GradientTape() as t:
            tape.set_parameters([a, b])
            assert tape.trainable_params == [0, 1]
            res = execute([tape], dev, **execute_kwargs)[0]

        jac = t.jacobian(res, [a, b])

        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.Variable(0.8, dtype=tf.float64)

        with tf.GradientTape() as t:
            tape.set_parameters([2 * a, b])
            res2 = execute([tape], dev, **execute_kwargs)[0]

        expected = [tf.cos(2 * a), -tf.cos(2 * a) * tf.sin(b)]
        assert np.allclose(res2, expected, atol=tol, rtol=0)

        jac2 = t.jacobian(res2, [a, b])
        expected = [
            [-2 * tf.sin(2 * a), 2 * tf.sin(2 * a) * tf.sin(b)],
            [0, -tf.cos(2 * a) * tf.cos(b)],
        ]
        assert np.allclose(jac2, expected, atol=tol, rtol=0)

    def test_classical_processing(self, execute_kwargs, tol):
        """Test classical processing within the quantum tape"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.constant(0.2, dtype=tf.float64)
        c = tf.Variable(0.3, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=1)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(a * c, wires=0)
                qml.RZ(b, wires=0)
                qml.RX(c + c**2 + tf.sin(a), wires=0)
                qml.expval(qml.PauliZ(0))

            res = execute([tape], dev, **execute_kwargs)[0]
            assert tape.trainable_params == [0, 2]
            assert tape.get_parameters() == [a * c, c + c**2 + tf.sin(a)]

        res = t.jacobian(res, [a, b, c])
        assert isinstance(res[0], tf.Tensor)
        assert res[1] is None
        assert isinstance(res[2], tf.Tensor)

    def test_no_trainable_parameters(self, execute_kwargs, tol):
        """Test evaluation and Jacobian if there are no trainable parameters"""
        b = tf.constant(0.2, dtype=tf.float64)
        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RY(0.2, wires=0)
                qml.RX(b, wires=0)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            res = execute([tape], dev, **execute_kwargs)[0]
            loss = tf.reduce_sum(res)

        assert res.shape == (2,)
        assert isinstance(res, tf.Tensor)

        res = t.jacobian(res, b)
        assert res is None

    @pytest.mark.parametrize("U", [tf.constant([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])])
    def test_matrix_parameter(self, execute_kwargs, U, tol):
        """Test that the TF interface works correctly
        with a matrix parameter"""
        a = tf.Variable(0.1, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as t:

            with qml.tape.QuantumTape() as tape:
                qml.QubitUnitary(U, wires=0)
                qml.RY(a, wires=0)
                qml.expval(qml.PauliZ(0))

            res = execute([tape], dev, **execute_kwargs)[0]
            assert tape.trainable_params == [1]

        assert np.allclose(res, -tf.cos(a), atol=tol, rtol=0)

        res = t.jacobian(res, a)
        assert np.allclose(res, tf.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, execute_kwargs, tol):
        """Test that operation and nested tape expansion
        is differentiable"""

        class U3(qml.U3):
            def expand(self):
                tape = qml.tape.QuantumTape()
                theta, phi, lam = self.data
                wires = self.wires
                tape._ops += [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]
                return tape

        qtape = qml.tape.QuantumTape()

        dev = qml.device("default.qubit", wires=1)
        a = np.array(0.1)
        p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)

        with tf.GradientTape() as tape:

            with qtape:
                qml.RX(a, wires=0)
                U3(p[0], p[1], p[2], wires=0)
                qml.expval(qml.PauliX(0))

            res = execute([qtape], dev, **execute_kwargs)[0]

        expected = tf.cos(a) * tf.cos(p[1]) * tf.sin(p[0]) + tf.sin(a) * (
            tf.cos(p[2]) * tf.sin(p[1]) + tf.cos(p[0]) * tf.cos(p[1]) * tf.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, p)
        expected = np.array(
            [
                tf.cos(p[1]) * (tf.cos(a) * tf.cos(p[0]) - tf.sin(a) * tf.sin(p[0]) * tf.sin(p[2])),
                tf.cos(p[1]) * tf.cos(p[2]) * tf.sin(a)
                - tf.sin(p[1])
                * (tf.cos(a) * tf.sin(p[0]) + tf.cos(p[0]) * tf.sin(a) * tf.sin(p[2])),
                tf.sin(a)
                * (tf.cos(p[0]) * tf.cos(p[1]) * tf.cos(p[2]) - tf.sin(p[1]) * tf.sin(p[2])),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_probability_differentiation(self, execute_kwargs, tol):
        """Tests correct output shape and evaluation for a tape
        with prob outputs"""

        if execute_kwargs["gradient_fn"] == "device":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        dev = qml.device("default.qubit", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.probs(wires=[0])
                qml.probs(wires=[1])

            res = execute([tape], dev, **execute_kwargs)[0]

        expected = np.array(
            [
                [tf.cos(x / 2) ** 2, tf.sin(x / 2) ** 2],
                [(1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = t.jacobian(res, [x, y])
        expected = np.array(
            [
                [
                    [-tf.sin(x) / 2, tf.sin(x) / 2],
                    [-tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                ],
                [
                    [0, 0],
                    [-tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
                ],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_ragged_differentiation(self, execute_kwargs, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if execute_kwargs["gradient_fn"] == "device":
            pytest.skip("Adjoint differentiation does not yet support probabilities")

        dev = qml.device("default.qubit", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.probs(wires=[1])

            res = execute([tape], dev, **execute_kwargs)[0]

        expected = np.array(
            [tf.cos(x), (1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = t.jacobian(res, [x, y])
        expected = np.array(
            [
                [-tf.sin(x), -tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                [0, -tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_sampling(self, execute_kwargs):
        """Test sampling works as expected"""
        if execute_kwargs["gradient_fn"] == "device" and execute_kwargs["mode"] == "forward":
            pytest.skip("Adjoint differentiation does not support samples")

        dev = qml.device("default.qubit", wires=2, shots=10)

        with tf.GradientTape() as t:
            with qml.tape.QuantumTape() as tape:
                qml.Hadamard(wires=[0])
                qml.CNOT(wires=[0, 1])
                qml.sample(qml.PauliZ(0))
                qml.sample(qml.PauliX(1))

            res = execute([tape], dev, **execute_kwargs)[0]

        assert res.shape == (2, 10)
        assert isinstance(res, tf.Tensor)


class TestHigherOrderDerivatives:
    """Test that the TensorFlow execute function can be differentiated"""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "params",
        [
            tf.Variable([0.543, -0.654], dtype=tf.float64),
            tf.Variable([0, -0.654], dtype=tf.float64),
            tf.Variable([-2.0, 0], dtype=tf.float64),
        ],
    )
    def test_parameter_shift_hessian(self, params, tol):
        """Tests that the output of the parameter-shift transform
        can be differentiated using tensorflow, yielding second derivatives."""
        dev = qml.device("default.qubit.tf", wires=2)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)
        x, y = params * 1.0

        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                with qml.tape.QuantumTape() as tape1:
                    qml.RX(params[0], wires=[0])
                    qml.RY(params[1], wires=[1])
                    qml.CNOT(wires=[0, 1])
                    qml.var(qml.PauliZ(0) @ qml.PauliX(1))

                with qml.tape.QuantumTape() as tape2:
                    qml.RX(params[0], wires=0)
                    qml.RY(params[0], wires=1)
                    qml.CNOT(wires=[0, 1])
                    qml.probs(wires=1)

                result = execute(
                    [tape1, tape2], dev, gradient_fn=param_shift, interface="tf", max_diff=2
                )
                res = result[0][0] + result[1][0, 0]

            expected = 0.5 * (3 + np.cos(x) ** 2 * np.cos(2 * y))
            assert np.allclose(res, expected, atol=tol, rtol=0)

            grad = t1.gradient(res, params)
            expected = np.array(
                [-np.cos(x) * np.cos(2 * y) * np.sin(x), -np.cos(x) ** 2 * np.sin(2 * y)]
            )
            assert np.allclose(grad, expected, atol=tol, rtol=0)

        hess = t2.jacobian(grad, params)
        expected = np.array(
            [
                [-np.cos(2 * x) * np.cos(2 * y), np.sin(2 * x) * np.sin(2 * y)],
                [np.sin(2 * x) * np.sin(2 * y), -2 * np.cos(x) ** 2 * np.cos(2 * y)],
            ]
        )
        assert np.allclose(hess, expected, atol=tol, rtol=0)

    def test_hessian_vector_valued(self, tol):
        """Test hessian calculation of a vector valued QNode"""
        dev = qml.device("default.qubit.tf", wires=1)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)

        with tf.GradientTape() as t2:
            with tf.GradientTape(persistent=True) as t1:
                with qml.tape.QuantumTape() as tape:
                    qml.RY(params[0], wires=0)
                    qml.RX(params[1], wires=0)
                    qml.probs(wires=0)

                res = execute([tape], dev, gradient_fn=param_shift, interface="tf", max_diff=2)
                res = tf.stack(res)

            g = t1.jacobian(res, params, experimental_use_pfor=False)

        hess = tf.squeeze(t2.jacobian(g, params))

        a, b = params * 1.0

        expected_res = [
            0.5 + 0.5 * tf.cos(a) * tf.cos(b),
            0.5 - 0.5 * tf.cos(a) * tf.cos(b),
        ]
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [
            [-0.5 * tf.sin(a) * tf.cos(b), -0.5 * tf.cos(a) * tf.sin(b)],
            [0.5 * tf.sin(a) * tf.cos(b), 0.5 * tf.cos(a) * tf.sin(b)],
        ]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        expected_hess = [
            [
                [-0.5 * tf.cos(a) * tf.cos(b), 0.5 * tf.sin(a) * tf.sin(b)],
                [0.5 * tf.sin(a) * tf.sin(b), -0.5 * tf.cos(a) * tf.cos(b)],
            ],
            [
                [0.5 * tf.cos(a) * tf.cos(b), -0.5 * tf.sin(a) * tf.sin(b)],
                [-0.5 * tf.sin(a) * tf.sin(b), 0.5 * tf.cos(a) * tf.cos(b)],
            ],
        ]

        np.testing.assert_allclose(hess, expected_hess, atol=tol, rtol=0, verbose=True)

    def test_adjoint_hessian(self, tol):
        """Since the adjoint hessian is not a differentiable transform,
        higher-order derivatives are not supported."""
        dev = qml.device("default.qubit", wires=2)
        params = tf.Variable([0.543, -0.654])

        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                with qml.tape.QuantumTape() as tape:
                    qml.RX(params[0], wires=[0])
                    qml.RY(params[1], wires=[1])
                    qml.CNOT(wires=[0, 1])
                    qml.expval(qml.PauliZ(0))

                res = execute(
                    [tape],
                    dev,
                    gradient_fn="device",
                    gradient_kwargs={"method": "adjoint_jacobian", "use_device_state": True},
                    interface="tf",
                )[0]

            grad = t1.gradient(res, params)
            assert grad is not None

        hess = t2.jacobian(grad, params)
        assert hess is None

    def test_max_diff(self, tol):
        """Test that setting the max_diff parameter blocks higher-order
        derivatives"""
        dev = qml.device("default.qubit.tf", wires=2)
        params = tf.Variable([0.543, -0.654], dtype=tf.float64)
        x, y = params * 1.0

        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                with qml.tape.QuantumTape() as tape1:
                    qml.RX(params[0], wires=[0])
                    qml.RY(params[1], wires=[1])
                    qml.CNOT(wires=[0, 1])
                    qml.var(qml.PauliZ(0) @ qml.PauliX(1))

                with qml.tape.QuantumTape() as tape2:
                    qml.RX(params[0], wires=0)
                    qml.RY(params[0], wires=1)
                    qml.CNOT(wires=[0, 1])
                    qml.probs(wires=1)

                result = execute(
                    [tape1, tape2], dev, gradient_fn=param_shift, max_diff=1, interface="tf"
                )
                res = result[0][0] + result[1][0, 0]

                expected = 0.5 * (3 + np.cos(x) ** 2 * np.cos(2 * y))
                assert np.allclose(res, expected, atol=tol, rtol=0)

            grad = t1.gradient(res, params)

            expected = np.array(
                [-np.cos(x) * np.cos(2 * y) * np.sin(x), -np.cos(x) ** 2 * np.sin(2 * y)]
            )
            assert np.allclose(grad, expected, atol=tol, rtol=0)

        hess = t2.jacobian(grad, params)
        assert hess is None


execute_kwargs = [
    {"gradient_fn": param_shift, "interface": "tensorflow"},
    {"gradient_fn": finite_diff, "interface": "tensorflow"},
]


@pytest.mark.parametrize("execute_kwargs", execute_kwargs)
class TestHamiltonianWorkflows:
    """Test that tapes ending with expectations
    of Hamiltonians provide correct results and gradients"""

    @pytest.fixture
    def cost_fn(self, execute_kwargs):
        """Cost function for gradient tests"""

        def _cost_fn(weights, coeffs1, coeffs2, dev=None):
            obs1 = [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliY(0)]
            H1 = qml.Hamiltonian(coeffs1, obs1)

            obs2 = [qml.PauliZ(0)]
            H2 = qml.Hamiltonian(coeffs2, obs2)

            with qml.tape.QuantumTape() as tape:
                qml.RX(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.expval(H1)
                qml.expval(H2)

            return execute([tape], dev, **execute_kwargs)[0]

        return _cost_fn

    @staticmethod
    def cost_fn_expected(weights, coeffs1, coeffs2):
        """Analytic value of cost_fn above"""
        a, b, c = coeffs1.numpy()
        d = coeffs2.numpy()[0]
        x, y = weights.numpy()
        return [-c * np.sin(x) * np.sin(y) + np.cos(x) * (a + b * np.sin(y)), d * np.cos(x)]

    @staticmethod
    def cost_fn_jacobian(weights, coeffs1, coeffs2):
        """Analytic jacobian of cost_fn above"""
        a, b, c = coeffs1.numpy()
        d = coeffs2.numpy()[0]
        x, y = weights.numpy()
        return np.array(
            [
                [
                    -c * np.cos(x) * np.sin(y) - np.sin(x) * (a + b * np.sin(y)),
                    b * np.cos(x) * np.cos(y) - c * np.cos(y) * np.sin(x),
                    np.cos(x),
                    np.cos(x) * np.sin(y),
                    -(np.sin(x) * np.sin(y)),
                    0,
                ],
                [-d * np.sin(x), 0, 0, 0, 0, np.cos(x)],
            ]
        )

    def test_multiple_hamiltonians_not_trainable(self, cost_fn, execute_kwargs, tol):
        coeffs1 = tf.constant([0.1, 0.2, 0.3], dtype=tf.float64)
        coeffs2 = tf.constant([0.7], dtype=tf.float64)
        weights = tf.Variable([0.4, 0.5], dtype=tf.float64)
        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as tape:
            res = cost_fn(weights, coeffs1, coeffs2, dev=dev)

        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [weights, coeffs1, coeffs2])
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        assert np.allclose(res[0], expected[:, :2], atol=tol, rtol=0)
        assert res[1] is None
        assert res[2] is None

    def test_multiple_hamiltonians_trainable(self, cost_fn, execute_kwargs, tol):
        coeffs1 = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)
        coeffs2 = tf.Variable([0.7], dtype=tf.float64)
        weights = tf.Variable([0.4, 0.5], dtype=tf.float64)
        dev = qml.device("default.qubit", wires=2)

        with tf.GradientTape() as tape:
            res = cost_fn(weights, coeffs1, coeffs2, dev=dev)

        expected = self.cost_fn_expected(weights, coeffs1, coeffs2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [weights, coeffs1, coeffs2])
        expected = self.cost_fn_jacobian(weights, coeffs1, coeffs2)
        assert np.allclose(res[0], expected[:, :2], atol=tol, rtol=0)
        assert np.allclose(res[1], expected[:, 2:5], atol=tol, rtol=0)
        assert np.allclose(res[2], expected[:, 5:], atol=tol, rtol=0)
