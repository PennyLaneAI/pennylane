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
"""
Unit tests for the ``GradientDescentOptimizer``.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

univariate = [
    (qml.math.sin, qml.math.cos),
    (lambda x: qml.math.exp(x / 10.0), lambda x: qml.math.exp(x / 10.0) / 10.0),
    (lambda x: x**2, lambda x: 2 * x),
]

multivariate = [
    (lambda x: qml.math.sin(x[0]) + qml.math.cos(x[1]), lambda x: (qml.math.array([qml.math.cos(x[0]), -qml.math.sin(x[1])]),)),
    (
        lambda x: qml.math.exp(x[0] / 3) * qml.math.tanh(x[1]),
        lambda x: (
            qml.math.array(
                [
                    qml.math.exp(x[0] / 3) / 3 * qml.math.tanh(x[1]),
                    qml.math.exp(x[0] / 3) * (1 - qml.math.tanh(x[1]) ** 2),
                ]
            ),
        ),
    ),
    (lambda x: qml.math.sum([x_**2 for x_ in x]), lambda x: (qml.math.array([2 * x_ for x_ in x]),)),
]

multivariate_jit = [
    (lambda x: qml.math.sin(x[0]) + qml.math.cos(x[1]), lambda x: (qml.math.array([qml.math.cos(x[0]), -qml.math.sin(x[1])], like="jax"),)),
    (
        lambda x: qml.math.exp(x[0] / 3) * qml.math.tanh(x[1]),
        lambda x: (
            qml.math.array(
                [
                    qml.math.exp(x[0] / 3) / 3 * qml.math.tanh(x[1]),
                    qml.math.exp(x[0] / 3) * (1 - qml.math.tanh(x[1]) ** 2),
                ], like="jax"
            ),
        ),
    ),
    (lambda x: qml.math.sum([x_**2 for x_ in x]), lambda x: (qml.math.array([2 * x_ for x_ in x], like="jax"),)),
]

class TestGradientDescentOptimizer:
    """Test the Gradient Descent optimizer"""

    @pytest.mark.parametrize(
        "grad,args",
        [
            ([40, -4, 12, -17, 400], [0, 30, 6, -7, 800]),
            ([0.00033, 0.45e-5, 0.0], [1.3, -0.5, 8e3]),
            ([43], [0.8]),
        ],
    )
    def test_apply_grad(self, grad, args, tol):
        """
        Test that a gradient step can be applied correctly to a set of parameters.
        """
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(None, stepsize=stepsize)
        grad, args = np.array(grad), np.array(args, requires_grad=True)

        res = sgd_opt.apply_grad(grad, args)
        expected = args - stepsize * grad
        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "grad,args",
        [
            ([40, -4, 12, -17, 400], [0, 30, 6, -7, 800]),
            ([0.00033, 0.45e-5, 0.0], [1.3, -0.5, 8e3]),
            ([43], [0.8]),
        ],
    )
    def test_apply_grad_jit(self, grad, args, tol):
        """
        Test that a gradient step can be applied correctly to a set of parameters.
        """

        import jax

        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(None, stepsize=stepsize)
        grad, args = qml.math.array(grad), qml.math.array(args)

        res = jax.jit(sgd_opt.apply_grad)(grad, args)
        expected = args - stepsize * grad
        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f, df", univariate)
    def test_step_and_cost_supplied_grad(self, args, f, df):
        """Test that returned cost is correct if gradient function is supplied"""
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(f, grad_fn=df, stepsize=stepsize)

        _, res = sgd_opt.step_and_cost(args)
        expected = f(args)
        assert np.all(res == expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f, df", univariate)
    def test_step_and_cost_supplied_grad_jit(self, args, f, df):
        """Test that returned cost is correct if gradient function is supplied"""
        import jax
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(f, grad_fn=df, stepsize=stepsize)

        _, res = jax.jit(sgd_opt.step_and_cost)(args)
        expected = f(args)
        assert np.all(res == expected)

    def test_step_and_cost_autograd_sgd_multiple_inputs(self):
        """Test that the correct cost is returned via the step_and_cost method for the
        gradient-descent optimizer"""
        stepsize = 0.1

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(*variables):
            qml.RX(variables[0][1], wires=[0])
            qml.RY(variables[1][2], wires=[0])
            qml.RY(variables[2], wires=[0])
            return qml.expval(qml.PauliZ(0))

        sgd_opt = GradientDescentOptimizer(quant_fun, stepsize=stepsize)

        inputs = [
            np.array((0.2, 0.3), requires_grad=True),
            np.array([0.4, 0.2, 0.4], requires_grad=False),
            np.array(0.1, requires_grad=True),
        ]

        _, res = sgd_opt.step_and_cost(*inputs)
        expected = quant_fun(*inputs)

        assert np.all(res == expected)

    def test_step_and_cost_autograd_sgd_single_multid_input(self):
        """Test that the correct cost is returned via the step_and_cost method for the
        gradient-descent optimizer"""
        stepsize = 0.1
        multid_array = np.array([[0.1, 0.2], [-0.1, -0.4]])

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun_mdarr(var):
            qml.RX(var[0, 1], wires=[0])
            qml.RY(var[1, 0], wires=[0])
            qml.RY(var[1, 1], wires=[0])
            return qml.expval(qml.PauliZ(0))

        sgd_opt = GradientDescentOptimizer(quant_fun_mdarr, stepsize=stepsize)
        _, res = sgd_opt.step_and_cost(multid_array)
        expected = quant_fun_mdarr(multid_array)

        assert np.all(res == expected)

    @pytest.mark.parametrize("x_start", np.linspace(-10, 10, 16, endpoint=False))
    @pytest.mark.parametrize("f, df", univariate)
    def test_gradient_descent_optimizer_univar(self, x_start, f, df, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for univariate functions."""
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(f, grad_fn=df, stepsize=stepsize)

        x_new = sgd_opt.step(x_start)
        x_correct = x_start - df(x_start) * stepsize
        assert np.allclose(x_new, x_correct, atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("x_start", np.linspace(-10, 10, 16, endpoint=False))
    @pytest.mark.parametrize("f, df", univariate)
    def test_gradient_descent_optimizer_univar_jit(self, x_start, f, df, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for univariate functions."""
        import jax

        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(f, grad_fn=df, stepsize=stepsize)

        x_new = jax.jit(sgd_opt.step)(x_start)
        x_correct = x_start - df(x_start) * stepsize
        assert qml.math.allclose(x_new, x_correct, atol=tol)

    @pytest.mark.parametrize("f, df", multivariate)
    def test_gradient_descent_optimizer_multivar(self, f, df, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for multivariate functions."""
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(f, grad_fn=df, stepsize=stepsize)

        x_vals = np.linspace(-10, 10, 16, endpoint=False)

        for jdx in range(len(x_vals[:-1])):
            x_vec = x_vals[jdx : jdx + 2]
            x_new = sgd_opt.step(x_vec)
            x_correct = x_vec - df(x_vec)[0] * stepsize
            assert np.allclose(x_new, x_correct, atol=tol)

    @pytest.mark.parametrize("f, df", multivariate_jit)
    def test_gradient_descent_optimizer_multivar_jit(self, f, df, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for multivariate functions."""
        import jax

        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(f, grad_fn=df, stepsize=stepsize)

        x_vals = qml.math.linspace(-10, 10, 16, endpoint=False)

        for jdx in range(len(x_vals[:-1])):
            x_vec = qml.math.array(x_vals[jdx : jdx + 2])
            x_new = jax.jit(sgd_opt.step)(x_vec)
            x_correct = x_vec - df(x_vec)[0] * stepsize
            assert qml.math.allclose(x_new, x_correct, atol=tol)

    def test_gradient_descent_optimizer_multivar_multidim(self, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for multivariate functions and with higher dimensional inputs."""
        stepsize = 0.1

        mvar_mdim_funcs = [
            lambda x: np.sin(x[0, 0]) + np.cos(x[1, 0]) - np.sin(x[0, 1]) + x[1, 1],
            lambda x: np.exp(x[0, 0] / 3) * np.tanh(x[0, 1]),
            lambda x: np.sum([x_[0] ** 2 for x_ in x]),
        ]
        grad_mvar_mdim_funcs = [
            lambda x: (np.array([[np.cos(x[0, 0]), -np.cos(x[0, 1])], [-np.sin(x[1, 0]), 1.0]]),),
            lambda x: (
                np.array(
                    [
                        [
                            np.exp(x[0, 0] / 3) / 3 * np.tanh(x[0, 1]),
                            np.exp(x[0, 0] / 3) * (1 - np.tanh(x[0, 1]) ** 2),
                        ],
                        [0.0, 0.0],
                    ]
                ),
            ),
            lambda x: (np.array([[2 * x_[0], 0.0] for x_ in x]),),
        ]

        x_vals = np.linspace(-10, 10, 16, endpoint=False)

        for gradf, f in zip(grad_mvar_mdim_funcs, mvar_mdim_funcs):
            sgd_opt = GradientDescentOptimizer(f, stepsize=stepsize)
            for jdx in range(len(x_vals[:-3])):
                x_vec = x_vals[jdx : jdx + 4]
                x_vec_multidim = np.reshape(x_vec, (2, 2))
                x_new = sgd_opt.step(x_vec_multidim)
                x_correct = x_vec_multidim - gradf(x_vec_multidim)[0] * stepsize
                x_new_flat = x_new.flatten()
                x_correct_flat = x_correct.flatten()
                assert np.allclose(x_new_flat, x_correct_flat, atol=tol)

    @pytest.mark.parametrize("x_start", np.linspace(-10, 10, 16, endpoint=False))
    @pytest.mark.parametrize("f, df", univariate)
    def test_gradient_descent_optimizer_usergrad(self, x_start, f, df, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        using user-provided gradients."""
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(f, grad_fn=df, stepsize=stepsize)

        x_new = sgd_opt.step(x_start)
        x_correct = x_start - df(x_start) * stepsize
        assert np.allclose(x_new, x_correct, atol=tol)
