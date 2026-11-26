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

import jax
import pytest
from legacy_optimizer_native import native_gradient

import pennylane as qp
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

univariate = [
    (np.sin, np.cos),
    (lambda x: np.exp(x / 10.0), lambda x: np.exp(x / 10.0) / 10.0),
    (lambda x: x**2, lambda x: 2 * x),
]

multivariate = [
    (lambda x: np.sin(x[0]) + np.cos(x[1]), lambda x: (np.array([np.cos(x[0]), -np.sin(x[1])]),)),
    (
        lambda x: np.exp(x[0] / 3) * np.tanh(x[1]),
        lambda x: (
            np.array(
                [
                    np.exp(x[0] / 3) / 3 * np.tanh(x[1]),
                    np.exp(x[0] / 3) * (1 - np.tanh(x[1]) ** 2),
                ]
            ),
        ),
    ),
    (lambda x: np.sum([x_**2 for x_ in x]), lambda x: (np.array([2 * x_ for x_ in x]),)),
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
        sgd_opt = GradientDescentOptimizer(stepsize)
        grad, args = np.array(grad), np.array(args, requires_grad=True)

        res = sgd_opt.apply_grad(grad, args)
        expected = args - stepsize * grad
        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f, df", univariate)
    def test_step_and_cost_supplied_grad(self, args, f, df):
        """Test that returned cost is correct if gradient function is supplied"""
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(stepsize)

        _, res = sgd_opt.step_and_cost(f, args, grad_fn=df)
        expected = f(args)
        assert np.all(res == expected)

    def test_step_and_cost_supplied_grad_sgd_multiple_inputs(self):
        """Test that the correct cost is returned via the step_and_cost method for the
        gradient-descent optimizer"""
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(stepsize)

        @qp.transforms.convert_to_numpy_parameters
        @qp.qnode(qp.device("default.qubit", wires=1), interface="numpy")
        def quant_fun(*variables):
            qp.RX(variables[0][1], wires=[0])
            qp.RY(variables[1][2], wires=[0])
            qp.RY(variables[2], wires=[0])
            return qp.expval(qp.PauliZ(0))

        inputs = [
            np.array((0.2, 0.3), requires_grad=True),
            np.array([0.4, 0.2, 0.4], requires_grad=False),
            np.array(0.1, requires_grad=True),
        ]

        _, res = sgd_opt.step_and_cost(quant_fun, *inputs, grad_fn=native_gradient(quant_fun))
        expected = quant_fun(*inputs)

        assert np.all(res == expected)

    def test_step_and_cost_supplied_grad_sgd_single_multid_input(self):
        """Test that the correct cost is returned via the step_and_cost method for the
        gradient-descent optimizer"""
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(stepsize)
        multid_array = np.array([[0.1, 0.2], [-0.1, -0.4]])

        @qp.transforms.convert_to_numpy_parameters
        @qp.qnode(qp.device("default.qubit", wires=1), interface="numpy")
        def quant_fun_mdarr(var):
            qp.RX(var[0, 1], wires=[0])
            qp.RY(var[1, 0], wires=[0])
            qp.RY(var[1, 1], wires=[0])
            return qp.expval(qp.PauliZ(0))

        _, res = sgd_opt.step_and_cost(
            quant_fun_mdarr, multid_array, grad_fn=native_gradient(quant_fun_mdarr)
        )
        expected = quant_fun_mdarr(multid_array)

        assert np.all(res == expected)

    @pytest.mark.parametrize("x_start", np.linspace(-10, 10, 16, endpoint=False))
    @pytest.mark.parametrize("f, df", univariate)
    def test_gradient_descent_optimizer_univar(self, x_start, f, df, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for univariate functions."""
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(stepsize)

        x_new = sgd_opt.step(f, x_start)
        x_correct = x_start - df(x_start) * stepsize
        assert np.allclose(x_new, x_correct, atol=tol)

    @pytest.mark.parametrize("f, df", multivariate)
    def test_gradient_descent_optimizer_multivar(self, f, df, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for multivariate functions."""
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(stepsize)

        x_vals = np.linspace(-10, 10, 16, endpoint=False)

        for jdx in range(len(x_vals[:-1])):
            x_vec = x_vals[jdx : jdx + 2]
            x_new = sgd_opt.step(f, x_vec)
            x_correct = x_vec - df(x_vec)[0] * stepsize
            assert np.allclose(x_new, x_correct, atol=tol)

    def test_gradient_descent_optimizer_multivar_multidim(self, tol):
        """Tests that basic stochastic gradient descent takes gradient-descent steps correctly
        for multivariate functions and with higher dimensional inputs."""
        stepsize = 0.1
        sgd_opt = GradientDescentOptimizer(stepsize)

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
            for jdx in range(len(x_vals[:-3])):
                x_vec = x_vals[jdx : jdx + 4]
                x_vec_multidim = np.reshape(x_vec, (2, 2))
                x_new = sgd_opt.step(f, x_vec_multidim)
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
        sgd_opt = GradientDescentOptimizer(stepsize)

        x_new = sgd_opt.step(f, x_start, grad_fn=df)
        x_correct = x_start - df(x_start) * stepsize
        assert np.allclose(x_new, x_correct, atol=tol)


@pytest.mark.jax
@pytest.mark.parametrize("grad_fn", [None, lambda x: 2 * x])
def test_optimizer_rejects_jax_parameters(grad_fn):
    """JAX parameters require an optimizer that recognizes their trainability."""
    with pytest.raises(ValueError, match="Optax"):
        qp.GradientDescentOptimizer().step(lambda x: x**2, jax.numpy.array(2.0), grad_fn=grad_fn)


@pytest.mark.jax
def test_optimizer_with_jax_auxiliary_data():
    """Mixed inputs require an explicit gradient function instead of automatic dispatch."""
    optimizer = qp.GradientDescentOptimizer(stepsize=0.1)
    value = qp.numpy.array(2.0)
    data = jax.numpy.array([1.0])
    objective = lambda parameter, aux: parameter**2
    with pytest.raises(ValueError, match="Autograd and JAX inputs cannot be mixed"):
        optimizer.step(objective, value, data)
    updated, unchanged = optimizer.step(objective, value, data, grad_fn=lambda x, aux: 2 * x)
    assert qp.math.allclose(updated, 1.6)
    assert unchanged is data


@pytest.mark.jax
@pytest.mark.parametrize("grad_fn", [None, lambda params: {"value": 2 * params["value"]}])
def test_optimizer_rejects_nested_jax_parameters(grad_fn):
    """Nested JAX parameters cannot silently bypass optimizer validation."""
    params = {"value": jax.numpy.array(2.0)}
    with pytest.raises(ValueError, match="Optax"):
        qp.GradientDescentOptimizer().step(
            lambda argument: argument["value"] ** 2, params, grad_fn=grad_fn
        )
