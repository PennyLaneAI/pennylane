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
Unit tests for the ``SPSAOptimizer``.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import SPSAOptimizer

univariate = [np.sin, lambda x: np.exp(x / 10.0), lambda x: x ** 2]

multivariate = [
    lambda x: np.sin(x[0]) + np.cos(x[1]),
    lambda x: np.exp(x[0] / 3) * np.tanh(x[1]),
    lambda x: np.sum([x_ ** 2 for x_ in x]),
]


class TestSPSAOptimizer:
    """Test the SPSA optimizer"""

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
        spsa_opt = SPSAOptimizer()
        grad, args = np.array(grad), np.array(args, requires_grad=True)

        A = 10
        alpha = 0.602
        a = (A + 1) ** alpha * 0.1 / (np.abs(grad).max() + 1)
        k = 1

        res = spsa_opt.apply_grad(grad, args, a, A, k, alpha)

        ak = a / (k + A + 1) ** alpha

        expected = args - ak * grad
        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("args", [-13, 0, 37])
    @pytest.mark.parametrize("func", univariate)
    def test_step_and_cost_univariate(self, args, func, tol):
        expected = func(args)
        spsa_opt = SPSAOptimizer()
        args = np.array(args, requires_grad=True)
        _, res = spsa_opt.step_and_cost(func, args)

        assert np.allclose(res, expected, atol=tol)

    @pytest.mark.parametrize("args", [[-10, 10], [0, 0], [13, 37]])
    @pytest.mark.parametrize("func", multivariate)
    def test_step_and_cost_multivariate(self, args, func, tol):
        expected = func(args)
        spsa_opt = SPSAOptimizer()
        args = np.array(args, requires_grad=True)
        _, res = spsa_opt.step_and_cost(func, args)

        assert np.allclose(res, expected, atol=tol)

    def test_step_and_cost_spsa_multiple_inputs(self):
        """Test that the correct cost is returned via the step_and_cost method for the
        SPSA optimizer"""
        spsa_opt = SPSAOptimizer()

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

        _, res = spsa_opt.step_and_cost(quant_fun, *inputs)
        expected = quant_fun(*inputs)

        assert np.all(res == expected)
