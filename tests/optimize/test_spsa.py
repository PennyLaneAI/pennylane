# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the SPSA optimizer"""
import pytest

import pennylane as qml
from pennylane import numpy as np


univariate = [(np.sin), (lambda x: np.exp(x / 10.0)), (lambda x: x**2)]

multivariate = [
    (lambda x: np.sin(x[0]) + np.cos(x[1])),
    (lambda x: np.exp(x[0] / 3) * np.tanh(x[1])),
    (lambda x: np.sum([x_**2 for x_ in x])),
]


class TestSPSAOptimizer:
    """Test the SPSA optimizer"""

    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f", univariate)
    def test_apply_grad(self, args, f):
        """
        Test that a gradient step can be applied correctly with a univariate
        function.
        """
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        args = np.array([args], requires_grad=True)

        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 20.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / (k + 1.0) ** gamma
        ak = a / (A + k + 1.0) ** alpha
        tol = np.maximum(np.abs(f(args - ck)), np.abs(f(args + ck)))

        y = f(args)
        grad = (y) / (2 * ck)
        spsa_opt.increment_k()

        res = spsa_opt.apply_grad(grad, args)
        expected = args - ak * grad
        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("f", multivariate)
    def test_apply_grad_multivar(self, f):
        """
        Test that a gradient step can be applied correctly to a multivariate
        function.
        """
        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 20.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / (k + 1.0) ** gamma
        ak = a / (A + k + 1.0) ** alpha
        spsa_opt = qml.SPSAOptimizer(maxiter=10)

        x_vals = np.linspace(-10, 10, 16, endpoint=False)

        for jdx in range(len(x_vals[:-1])):
            x_vec = x_vals[jdx : jdx + 2]
            y = f(x_vec)
            grad = (y) / (2 * ck * np.ones((2)))
            spsa_opt.increment_k()
            x_new = spsa_opt.apply_grad(grad, x_vec)
            x_al = x_vec - ak * grad
            tol = np.maximum(np.abs(f(x_vec - ck)), np.abs(f(x_vec + ck)))
            assert np.allclose(x_new, x_al, atol=tol)

    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f", univariate)
    def test_step_and_cost_supplied_cost(self, args, f):
        """Test that returned cost is correct"""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        args = np.array(args, requires_grad=True)

        _, res = spsa_opt.step_and_cost(f, args)
        expected = f(args)
        assert np.all(res == expected)

    def test_step_and_cost_supplied_cost2(self):
        """Test that the correct cost is returned via the step_and_cost method
        for the SPSA optimizer"""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(variables):
            qml.RX(variables[0], wires=[0])
            qml.RY(variables[1], wires=[0])
            qml.RY(variables[2], wires=[0])
            return qml.expval(qml.PauliZ(0))

        inputs = np.array([0.4, 0.2, 0.4], requires_grad=True)

        expected = quant_fun(inputs)
        _, res = spsa_opt.step_and_cost(quant_fun, inputs)

        assert np.all(res == expected)

    def test_step_spsa2(self):
        """Test that the correct param is returned via the step method"""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(variables):
            qml.RX(variables[0], wires=[0])
            qml.RY(variables[1], wires=[0])
            qml.RY(variables[2], wires=[0])
            return qml.expval(qml.PauliZ(0))

        inputs = np.array([0.4, 0.2, 0.4], requires_grad=True)

        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 20.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / (k + 1.0) ** gamma
        ak = a / (A + k + 1.0) ** alpha
        tol = np.maximum(np.abs(quant_fun(inputs - ck)), np.abs(quant_fun(inputs + ck)))

        y = quant_fun(inputs)
        grad = (y) / (2 * ck)

        expected = inputs - ak * grad

        res = spsa_opt.step(quant_fun, inputs)

        assert np.allclose(res, expected, atol=tol)

    def test_step_and_cost_spsa_single_multid_input(self):
        """Test that the correct cost is returned via the step_and_cost method
        with a multidimensional input"""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        multid_array = np.array([[0.1, 0.2], [-0.1, -0.4]])

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun_mdarr(var):
            qml.RX(var[0, 1], wires=[0])
            qml.RY(var[1, 0], wires=[0])
            qml.RY(var[1, 1], wires=[0])
            return qml.expval(qml.PauliZ(0))

        _, res = spsa_opt.step_and_cost(quant_fun_mdarr, multid_array)
        expected = quant_fun_mdarr(multid_array)

        assert np.all(res == expected)

    def test_step_spsa_single_multid_input(self):
        """Test that the correct param is returned via the step method
        with a multidimensional input"""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        multid_array = np.array([[0.1, 0.2], [-0.1, -0.4]])

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun_mdarr(var):
            qml.RX(var[0, 1], wires=[0])
            qml.RY(var[1, 0], wires=[0])
            qml.RY(var[1, 1], wires=[0])
            return qml.expval(qml.PauliZ(0))

        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 20.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / (k + 1.0) ** gamma
        ak = a / (A + k + 1.0) ** alpha
        tol = np.maximum(
            np.abs(quant_fun_mdarr(multid_array - ck)), np.abs(quant_fun_mdarr(multid_array + ck))
        )

        y = quant_fun_mdarr(multid_array)
        grad = (y) / (2 * ck)

        expected = multid_array - ak * grad

        res = spsa_opt.step(quant_fun_mdarr, multid_array)

        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f", univariate)
    def test_step(self, args, f):
        """
        Test that a gradient step can be applied correctly with a univariate
        function.
        """
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        args = np.array([args], requires_grad=True)

        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 20.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / (k + 1.0) ** gamma
        ak = a / (A + k + 1.0) ** alpha
        tol = np.maximum(np.abs(f(args - ck)), np.abs(f(args + ck)))

        y = f(args)
        grad = (y) / (2 * ck)

        res = spsa_opt.step(f, args)
        expected = args - ak * grad
        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("args", [0, -3, 42])
    @pytest.mark.parametrize("f", univariate)
    def test_step_and_cost(self, args, f):
        """
        Test that a gradient step can be applied correctly with a univariate
        function.
        """
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        args = np.array([args], requires_grad=True)

        alpha = 0.602
        gamma = 0.101
        c = 0.2
        A = 20.0
        a = 0.05 * (A + 1) ** alpha
        k = 1
        ck = c / (k + 1.0) ** gamma
        ak = a / (A + k + 1.0) ** alpha
        tol = np.maximum(np.abs(f(args - ck)), np.abs(f(args + ck)))

        y = f(args)
        grad = (y) / (2 * ck)

        res, rescost = spsa_opt.step_and_cost(f, args)
        expected = args - ak * grad
        assert np.allclose(res, expected, atol=tol)
        assert np.allclose(y, rescost, atol=tol)

    def test_parameters_not_a_tensor_and_not_all_require_grad(self):
        """Test execution of list of parameters of different sizes
        and not all require grad"""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(*variables):
            qml.RX(variables[0][1], wires=[0])
            qml.RY(variables[1][2], wires=[0])
            qml.RY(variables[2], wires=[0])
            return qml.expval(qml.PauliZ(0))

        inputs = [
            np.array((0.2, 0.3), requires_grad=True),
            np.array([0.4, 0.2, 0.4], requires_grad=False),
            np.array(0.1, requires_grad=True),
        ]

        res, cost = spsa_opt.step_and_cost(quant_fun, *inputs)
        assert isinstance(res, list)
        assert np.all(res[1] == inputs[1])
        assert np.all(res[0] != inputs[0])

    def test_parameters_in_step(self):
        """Test execution of list of parameters of different sizes
        and not all require grad"""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def quant_fun(*variables):
            qml.RX(variables[0][1], wires=[0])
            qml.RY(variables[1][2], wires=[0])
            qml.RY(variables[2], wires=[0])
            return qml.expval(qml.PauliZ(0))

        inputs = [
            np.array((0.2, 0.3), requires_grad=True),
            np.array([0.4, 0.2, 0.4], requires_grad=False),
            np.array(0.1, requires_grad=True),
        ]

        res = spsa_opt.step(quant_fun, *inputs)
        assert isinstance(res, list)
        assert np.all(res[1] == inputs[1])
        assert np.all(res[0] != inputs[0])

    def test_parameter_not_an_array(self):
        """Test function when there is only one float parameter that doesn't
        require grad"""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost(a):
            return circuit(a)

        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        params = 0.5

        res = spsa_opt.step(cost, params)

        assert isinstance(res, float)
        assert res == params

    def test_increment(self):
        """Test that increment works"""
        spsa_opt = qml.SPSAOptimizer(maxiter=10)
        spsa_opt.increment_k()
        spsa_opt.increment_k()

        assert spsa_opt.k == 2
