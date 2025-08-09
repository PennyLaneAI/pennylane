# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Sanity checks for classical automatic gradient formulas (without QNodes).
"""
import autograd
import pytest

import pennylane as qml
from pennylane import numpy as np


def test_grad_no_ints():
    """Test that grad raises a `ValueError` if the trainable parameter is an int."""

    x = qml.numpy.array(2)

    def f(x):
        return x**2

    with pytest.raises(ValueError, match="Autograd does not support differentiation of ints."):
        qml.grad(f)(x)

    y = qml.numpy.array([2, 2])
    with pytest.raises(ValueError, match="Autograd does not support differentiation of ints."):
        qml.jacobian(f)(y)


class TestGradientUnivar:
    """Tests gradients of univariate unidimensional functions."""

    def test_sin(self, tol):
        """Tests with sin function."""
        x_vals = np.linspace(-10, 10, 16, endpoint=False)
        g = qml.grad(np.sin, 0)
        auto_grad = [g(x) for x in x_vals]
        correct_grad = np.cos(x_vals)

        assert np.allclose(auto_grad, correct_grad, atol=tol, rtol=0)

    def test_exp(self, tol):
        """Tests exp function."""
        x_vals = np.linspace(-10, 10, 16, endpoint=False)
        func = lambda x: np.exp(x / 10.0) / 10.0
        g = qml.grad(func, 0)
        auto_grad = [g(x) for x in x_vals]
        correct_grad = np.exp(x_vals / 10.0) / 100.0

        assert np.allclose(auto_grad, correct_grad, atol=tol, rtol=0)

    def test_poly(self, tol):
        """Tests a polynomial function."""
        x_vals = np.linspace(-10, 10, 16, endpoint=False)
        func = lambda x: 2 * x**2 + 3 * x + 4
        g = qml.grad(func, 0)
        auto_grad = [g(x) for x in x_vals]
        correct_grad = 4 * x_vals + 3

        assert np.allclose(auto_grad, correct_grad, atol=tol, rtol=0)


class TestGradientMultiVar:
    """Tests gradients of multivariate unidimensional functions."""

    def test_sin(self, tol):
        """Tests gradients with multivariate sin and cosine."""
        multi_var = lambda x: np.sin(x[0]) + np.cos(x[1])
        grad_multi_var = lambda x: np.array([np.cos(x[0]), -np.sin(x[1])])

        x_vec = [1.5, -2.5]
        g = qml.grad(multi_var, 0)
        auto_grad = g(x_vec)
        correct_grad = grad_multi_var(x_vec)

        assert np.allclose(auto_grad, correct_grad, atol=tol, rtol=0)

    def test_exp(self, tol):
        """Tests gradients with a multivariate exp and tanh."""
        multi_var = lambda x: np.exp(x[0] / 3) * np.tanh(x[1])
        grad_multi_var = lambda x: np.array(
            [
                np.exp(x[0] / 3) / 3 * np.tanh(x[1]),
                np.exp(x[0] / 3) * (1 - np.tanh(x[1]) ** 2),
            ]
        )
        x_vec = np.random.uniform(-5, 5, size=2)
        g = qml.grad(multi_var, 0)
        auto_grad = g(x_vec)
        correct_grad = grad_multi_var(x_vec)

        assert np.allclose(auto_grad, correct_grad, atol=tol, rtol=0)

    def test_quadratic(self, tol):
        """Tests gradients with a quadratic function."""
        multi_var = lambda x: np.sum([x_**2 for x_ in x])
        grad_multi_var = lambda x: np.array([2 * x_ for x_ in x])
        x_vec = np.random.uniform(-5, 5, size=2)
        g = qml.grad(multi_var, 0)
        auto_grad = g(x_vec)
        correct_grad = grad_multi_var(x_vec)

        assert np.allclose(auto_grad, correct_grad, atol=tol, rtol=0)


class TestGradientMultiargs:
    """Tests gradients of univariate functions with multiple arguments in signature."""

    def test_sin(self, tol):
        """Tests multiarg gradients with sin and cos functions."""
        x = -2.5
        y = 1.5
        gradf = lambda x, y: (np.cos(x), -np.sin(y))
        f = lambda x, y: np.sin(x) + np.cos(y)

        # gradient wrt first argument
        gx = qml.grad(f, 0)

        auto_gradx = gx(x, y)
        correct_gradx = gradf(x, y)[0]
        assert np.allclose(auto_gradx, correct_gradx, atol=tol, rtol=0)

        # gradient wrt second argument
        gy = qml.grad(f, 1)
        auto_grady = gy(x, y)
        correct_grady = gradf(x, y)[1]
        assert np.allclose(auto_grady, correct_grady, atol=tol, rtol=0)

        # gradient wrt both arguments
        gxy = qml.grad(f, [0, 1])
        auto_gradxy = gxy(x, y)
        correct_gradxy = gradf(x, y)
        assert np.allclose(auto_gradxy, correct_gradxy, atol=tol, rtol=0)

    def test_exp(self, tol):
        """Tests multiarg gradients with exp and tanh functions."""
        x = -2.5
        y = 1.5
        gradf = lambda x, y: (
            np.exp(x / 3) / 3 * np.tanh(y),
            np.exp(x / 3) * (1 - np.tanh(y) ** 2),
        )
        f = lambda x, y: np.exp(x / 3) * np.tanh(y)

        # gradient wrt first argument
        gx = qml.grad(f, 0)
        auto_gradx = gx(x, y)
        correct_gradx = gradf(x, y)[0]
        assert np.allclose(auto_gradx, correct_gradx, atol=tol, rtol=0)

        # gradient wrt second argument
        gy = qml.grad(f, 1)
        auto_grady = gy(x, y)
        correct_grady = gradf(x, y)[1]
        assert np.allclose(auto_grady, correct_grady, atol=tol, rtol=0)

        # gradient wrt both arguments
        gxy = qml.grad(f, [0, 1])
        auto_gradxy = gxy(x, y)
        correct_gradxy = gradf(x, y)
        assert np.allclose(auto_gradxy, correct_gradxy, atol=tol, rtol=0)

    def test_linear(self, tol):
        """Tests multiarg gradients with a linear function."""
        x = -2.5
        y = 1.5
        gradf = lambda x, y: (2 * x, 2 * y)
        f = lambda x, y: np.sum([x_**2 for x_ in [x, y]])

        # gradient wrt first argument
        gx = qml.grad(f, 0)
        auto_gradx = gx(x, y)
        correct_gradx = gradf(x, y)[0]
        assert np.allclose(auto_gradx, correct_gradx, atol=tol, rtol=0)

        # gradient wrt second argument
        gy = qml.grad(f, 1)
        auto_grady = gy(x, y)
        correct_grady = gradf(x, y)[1]
        assert np.allclose(auto_grady, correct_grady, atol=tol, rtol=0)

        # gradient wrt both arguments
        gxy = qml.grad(f, [0, 1])
        auto_gradxy = gxy(x, y)
        correct_gradxy = gradf(x, y)
        assert np.allclose(auto_gradxy, correct_gradxy, atol=tol, rtol=0)


class TestGradientMultivarMultidim:
    """Tests gradients of multivariate multidimensional functions."""

    def test_sin(self, tol):
        """Tests gradients with multivariate multidimensional sin and cos."""
        x_vec = np.random.uniform(-5, 5, size=2)
        x_vec_multidim = np.expand_dims(x_vec, axis=1)

        gradf = lambda x: ([[np.cos(x[0, 0])], [-np.sin(x[[1]])]])
        f = lambda x: np.sin(x[0, 0]) + np.cos(x[1, 0])

        g = qml.grad(f, 0)
        auto_grad = g(x_vec_multidim)
        correct_grad = gradf(x_vec_multidim)
        assert np.allclose(auto_grad[0], correct_grad[0], atol=tol, rtol=0)
        assert np.allclose(auto_grad[1], correct_grad[1], atol=tol, rtol=0)

    def test_exp(self, tol):
        """Tests gradients with multivariate multidimensional exp and tanh."""
        x_vec = np.random.uniform(-5, 5, size=2)
        x_vec_multidim = np.expand_dims(x_vec, axis=1)

        gradf = lambda x: np.array(
            [
                [np.exp(x[0, 0] / 3) / 3 * np.tanh(x[1, 0])],
                [np.exp(x[0, 0] / 3) * (1 - np.tanh(x[1, 0]) ** 2)],
            ]
        )
        f = lambda x: np.exp(x[0, 0] / 3) * np.tanh(x[1, 0])

        g = qml.grad(f, 0)
        auto_grad = g(x_vec_multidim)
        correct_grad = gradf(x_vec_multidim)
        assert np.allclose(auto_grad, correct_grad, atol=tol, rtol=0)

    def test_linear(self, tol):
        """Tests gradients with multivariate multidimensional linear func."""
        x_vec = np.random.uniform(-5, 5, size=2)
        x_vec_multidim = np.expand_dims(x_vec, axis=1)

        gradf = lambda x: np.array([[2 * x_[0]] for x_ in x])
        f = lambda x: np.sum([x_[0] ** 2 for x_ in x])

        g = qml.grad(f, 0)
        auto_grad = g(x_vec_multidim)
        correct_grad = gradf(x_vec_multidim)
        assert np.allclose(auto_grad, correct_grad, atol=tol, rtol=0)


class TestGrad:
    """Unit tests for the gradient function"""

    def test_non_scalar_cost_gradient(self):
        """Test gradient computation with a non-scalar cost function raises an error"""

        def cost(x):
            return np.sin(x)

        grad_fn = qml.grad(cost, argnum=[0])
        arr1 = np.array([0.0, 1.0, 2.0], requires_grad=True)

        with pytest.raises(TypeError, match="only applies to real scalar-output functions"):
            grad_fn(arr1)

    # pylint: disable=no-value-for-parameter
    def test_agrees_with_autograd(self, tol):
        """Test that the grad function agrees with autograd"""

        def cost(x):
            return np.sum(np.sin(x) * x[0] ** 3)

        grad_fn = qml.grad(cost)
        params = np.array([0.5, 1.0, 2.0], requires_grad=True)
        res = grad_fn(params)
        expected = autograd.grad(cost)(params)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_forward_pass_value_storing(self, tol):
        """Test that the intermediate forward pass value is accessible and correct"""

        def cost(x):
            return np.sum(np.sin(x) * x[0] ** 3)

        grad_fn = qml.grad(cost)
        params = np.array([-0.654, 1.0, 2.0], requires_grad=True)

        assert grad_fn.forward is None

        grad_fn(params)

        res = grad_fn.forward
        expected = cost(params)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # change the parameters
        params2 = np.array([1.4, 1.0, 2.0], requires_grad=True)
        grad_fn(params2)

        res = grad_fn.forward
        expected = cost(params2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_no_argnum_grad(self, mocker, tol):
        """Test the qml.grad function for inferred argnums"""
        cost_fn = lambda x, y: np.sin(x) * np.cos(y) + x * y**2

        x = np.array(0.5, requires_grad=True)
        y = np.array(0.2, requires_grad=True)

        grad_fn = qml.grad(cost_fn)
        spy = mocker.spy(grad_fn, "_grad_with_forward")

        res = grad_fn(x, y)
        expected = np.array([np.cos(x) * np.cos(y) + y**2, -np.sin(x) * np.sin(y) + 2 * x * y])
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert spy.call_args_list[0][1]["argnum"] == [0, 1]

        x = np.array(0.5, requires_grad=True)
        y = np.array(0.2, requires_grad=False)
        spy.call_args_list = []

        res = grad_fn(x, y)
        expected = np.array([np.cos(x) * np.cos(y) + y**2])
        assert np.allclose(res, expected, atol=tol, rtol=0)
        assert spy.call_args_list[0][1]["argnum"] == 0


class TestJacobian:
    """Tests for the jacobian function"""

    def test_single_argnum_jacobian(self, tol):
        """Test the qml.jacobian function for a single argnum"""
        cost_fn = lambda x, y: np.array([np.sin(x) * np.cos(y), x * y**2])

        x = np.array(0.5, requires_grad=True)
        y = np.array(0.2, requires_grad=True)

        jac_fn = qml.jacobian(cost_fn, argnum=0)
        res = jac_fn(x, y)
        expected = np.array([np.cos(x) * np.cos(y), y**2])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_argnum_jacobian(self, tol):
        """Test the qml.jacobian function for multiple argnums"""
        cost_fn = lambda x, y: np.array([np.sin(x) * np.cos(y), x * y**2])

        x = np.array(0.5, requires_grad=True)
        y = np.array(0.2, requires_grad=True)

        jac_fn = qml.jacobian(cost_fn, argnum=[0, 1])
        res = jac_fn(x, y)
        expected = (
            np.array([np.cos(x) * np.cos(y), y**2]),
            np.array([-np.sin(x) * np.sin(y), 2 * x * y]),
        )
        assert all(np.allclose(_r, _e, atol=tol, rtol=0) for _r, _e in zip(res, expected))

    def test_no_argnum_jacobian(self, tol):
        """Test the qml.jacobian function for inferred argnums"""
        cost_fn = lambda x, y: np.array([np.sin(x) * np.cos(y), x * y**2])

        x = np.array(0.5, requires_grad=True)
        y = np.array(0.2, requires_grad=True)

        jac_fn = qml.jacobian(cost_fn)
        res = jac_fn(x, y)
        expected = (
            np.array([np.cos(x) * np.cos(y), y**2]),
            np.array([-np.sin(x) * np.sin(y), 2 * x * y]),
        )
        assert all(np.allclose(_r, _e, atol=tol, rtol=0) for _r, _e in zip(res, expected))

        x = np.array(0.5, requires_grad=False)
        y = np.array(0.2, requires_grad=True)

        res = jac_fn(x, y)
        assert np.allclose(res, expected[1], atol=tol, rtol=0)
