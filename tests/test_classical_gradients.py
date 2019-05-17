# Copyright 2018 Xanadu Quantum Technologies Inc.

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

import pytest

from defaults import pennylane as qml, BaseTest
from pennylane import numpy as np


def test_gradient_univar():
    """Tests gradients of univariate unidimensional functions."""
    # sin
    x_vals = np.linspace(-10, 10, 16, endpoint=False)

    g = qml.grad(np.sin, 0)
    auto_grad = [g(x) for x in x_vals]
    correct_grad = np.cos(x_vals)
    np.allclose(auto_grad, correct_grad)

    # Custom function 1
    func = lambda x: np.exp(x / 10.0) / 10.0
    g = qml.grad(func, 0)
    auto_grad = [g(x) for x in x_vals]
    correct_grad = np.exp(x_vals / 10.0)
    np.allclose(auto_grad, correct_grad)

    # Custom function 2
    func = lambda x: 2 * x
    g = qml.grad(func, 0)
    auto_grad = [g(x) for x in x_vals]
    correct_grad = x_vals ** 2
    np.allclose(auto_grad, correct_grad)


def test_gradient_multivar():
    """Tests gradients of multivariate unidimensional functions."""
    # function 1
    multi_var = lambda x: np.sin(x[0]) + np.cos(x[1])
    grad_multi_var = lambda x: np.array([np.cos(x[0]), -np.sin(x[1])])

    x_vec = np.random.uniform(-5, 5, size=(2))
    g = qml.grad(multi_var, 0)
    auto_grad = g(x_vec)
    correct_grad = grad_multi_var(x_vec)
    np.allclose(auto_grad, correct_grad)

    # function 2
    multi_var = lambda x: np.exp(x[0] / 3) * np.tanh(x[1])
    grad_multi_var = lambda x: np.array(
        [
            np.exp(x[0] / 3) / 3 * np.tanh(x[1]),
            np.exp(x[0] / 3) * (1 - np.tanh(x[1]) ** 2),
        ]
    )

    x_vec = np.random.uniform(-5, 5, size=(2))
    g = qml.grad(multi_var, 0)
    auto_grad = g(x_vec)
    correct_grad = grad_multi_var(x_vec)
    np.allclose(auto_grad, correct_grad)

    # function 3
    multi_var = lambda x: np.sum([x_ ** 2 for x_ in x])
    grad_multi_var = lambda x: np.array([2 * x_ for x_ in x])

    x_vec = np.random.uniform(-5, 5, size=(2))
    g = qml.grad(multi_var, 0)
    auto_grad = g(x_vec)
    correct_grad = grad_multi_var(x_vec)
    np.allclose(auto_grad, correct_grad)


def test_gradient_multiargs():
    """Tests gradients of univariate functions with multiple arguments in signature."""

    x = np.random.random()
    y = np.random.random()

    # ----------
    # function 1
    # ----------
    gradf = lambda x, y: (np.cos(x), -np.sin(y))
    f = lambda x, y: np.sin(x) + np.cos(y)

    # gradient wrt first argument
    gx = qml.grad(f, 0)
    auto_gradx = gx(x, y)
    correct_gradx = gradf(x, y)[0]
    np.allclose(auto_gradx, correct_gradx)

    # gradient wrt second argument
    gy = qml.grad(f, 1)
    auto_grady = gy(x, y)
    correct_grady = gradf(x, y)[1]
    np.allclose(auto_grady, correct_grady)

    # gradient wrt both arguments
    gxy = qml.grad(f, [0, 1])
    auto_gradxy = gxy(x, y)
    correct_gradxy = gradf(x, y)
    np.allclose(auto_gradxy, correct_gradxy)

    # ----------
    # function 2
    # ----------
    gradf = lambda x, y: (
        np.exp(x / 3) / 3 * np.tanh(y),
        np.exp(x / 3) * (1 - np.tanh(y) ** 2),
    )
    f = lambda x, y: np.exp(x / 3) * np.tanh(y)

    # gradient wrt first argument
    gx = qml.grad(f, 0)
    auto_gradx = gx(x, y)
    correct_gradx = gradf(x, y)[0]
    np.allclose(auto_gradx, correct_gradx)

    # gradient wrt second argument
    gy = qml.grad(f, 1)
    auto_grady = gy(x, y)
    correct_grady = gradf(x, y)[1]
    np.allclose(auto_grady, correct_grady)

    # gradient wrt both arguments
    gxy = qml.grad(f, [0, 1])
    auto_gradxy = gxy(x, y)
    correct_gradxy = gradf(x, y)
    np.allclose(auto_gradxy, correct_gradxy)

    # ----------
    # function 3
    # ----------
    gradf = lambda x, y: (2 * x, 2 * y)
    f = lambda x, y: np.sum([x_ ** 2 for x_ in [x, y]])

    # gradient wrt first argument
    gx = qml.grad(f, 0)
    auto_gradx = gx(x, y)
    correct_gradx = gradf(x, y)[0]
    np.allclose(auto_gradx, correct_gradx)

    # gradient wrt second argument
    gy = qml.grad(f, 1)
    auto_grady = gy(x, y)
    correct_grady = gradf(x, y)[1]
    np.allclose(auto_grady, correct_grady)

    # gradient wrt both arguments
    gxy = qml.grad(f, [0, 1])
    auto_gradxy = gxy(x, y)
    correct_gradxy = gradf(x, y)
    np.allclose(auto_gradxy, correct_gradxy)


def test_gradient_multivar_multidim():
    """Tests gradients of multivariate multidimensional functions."""
    # ----------
    # function 1
    # ----------
    gradf = lambda x: np.array([[np.cos(x[0, 0])], [-np.sin(x[[1]])]])
    f = lambda x: np.sin(x[0, 0]) + np.cos(x[1, 0])

    x_vals = np.linspace(-10, 10, 16, endpoint=False)

    x_vec = np.random.uniform(-5, 5, size=(2))
    x_vec_multidim = np.expand_dims(x_vec, axis=1)

    g = qml.grad(f, 0)
    auto_grad = g(x_vec_multidim)
    correct_grad = gradf(x_vec_multidim)
    np.allclose(auto_grad, correct_grad)

    # ----------
    # function 2
    # ----------
    gradf = lambda x: np.array(
        [
            [np.exp(x[0, 0] / 3) / 3 * np.tanh(x[1, 0])],
            [np.exp(x[0, 0] / 3) * (1 - np.tanh(x[1, 0]) ** 2)],
        ]
    )
    f = lambda x: np.exp(x[0, 0] / 3) * np.tanh(x[1, 0])

    x_vals = np.linspace(-10, 10, 16, endpoint=False)

    x_vec = np.random.uniform(-5, 5, size=(2))
    x_vec_multidim = np.expand_dims(x_vec, axis=1)

    g = qml.grad(f, 0)
    auto_grad = g(x_vec_multidim)
    correct_grad = gradf(x_vec_multidim)
    np.allclose(auto_grad, correct_grad)

    # ----------
    # function 3
    # ----------
    gradf = lambda x: np.array([[2 * x_[0]] for x_ in x])
    f = lambda x: np.sum([x_[0] ** 2 for x_ in x])

    x_vals = np.linspace(-10, 10, 16, endpoint=False)

    x_vec = np.random.uniform(-5, 5, size=(2))
    x_vec_multidim = np.expand_dims(x_vec, axis=1)

    g = qml.grad(f, 0)
    auto_grad = g(x_vec_multidim)
    correct_grad = gradf(x_vec_multidim)
    np.allclose(auto_grad, correct_grad)
