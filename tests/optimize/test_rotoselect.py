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
Unit tests for the ``RotoselectOptimizer``.
"""
import itertools as it

import numpy as onp
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import RotoselectOptimizer


class TestRotoselectOptimizer:
    """Test the Rotoselect (gradient-free) optimizer"""

    @pytest.mark.parametrize("params", [[1.7, 2.2], [-1.42, 0.1], [0.05, -0.8]])
    def test_step_and_cost_autograd_rotoselect(self, params):
        """Test that the correct cost is returned via the step_and_cost method for the
        Rotoselect momentum optimizer"""
        rotoselect_opt = RotoselectOptimizer()
        generators = [qml.RY, qml.RX]
        possible_generators = [qml.RX, qml.RY, qml.RZ]
        rotoselect_opt.possible_generators = possible_generators

        dev = qml.device("default.qubit", wires=2)

        def ansatz(params, generators):
            generators[0](params[0], wires=0)
            generators[1](params[1], wires=1)
            qml.CNOT(wires=[0, 1])

        @qml.qnode(dev)
        def circuit_1(params, generators=None):  # generators will be passed as a keyword arg
            ansatz(params, generators)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        @qml.qnode(dev)
        def circuit_2(params, generators=None):  # generators will be passed as a keyword arg
            ansatz(params, generators)
            return qml.expval(qml.PauliX(0))

        def cost_fn(params, generators):
            Z_1, Y_2 = circuit_1(params, generators=generators)
            X_1 = circuit_2(params, generators=generators)
            return 0.5 * Y_2 + 0.8 * Z_1 - 0.2 * X_1

        _, _, res = rotoselect_opt.step_and_cost(cost_fn, params, generators)
        expected = cost_fn(params, generators)

        assert np.all(res == expected)

    @pytest.mark.slow
    @pytest.mark.parametrize("x_start", [[1.2, 0.2], [-0.62, -2.1], [0.05, 0.8]])
    @pytest.mark.parametrize(
        "generators", [list(tup) for tup in it.product([qml.RX, qml.RY, qml.RZ], repeat=2)]
    )
    def test_rotoselect_optimizer(self, x_start, generators, tol):
        """Tests that rotoselect optimizer finds the optimal generators and parameters for the
        VQE circuit defined in `this rotoselect tutorial
        <https://pennylane.ai/qml/demos/tutorial_rotoselect>`_."""

        # the optimal generators for the 2-qubit VQE circuit
        # H = 0.5 * Y_2 + 0.8 * Z_1 - 0.2 * X_1
        rotoselect_opt = RotoselectOptimizer()
        optimal_generators = [qml.RY, qml.RX]
        possible_generators = [qml.RX, qml.RY, qml.RZ]
        rotoselect_opt.possible_generators = possible_generators

        dev = qml.device("default.qubit", wires=2)

        def ansatz(params, generators):
            generators[0](params[0], wires=0)
            generators[1](params[1], wires=1)
            qml.CNOT(wires=[0, 1])

        @qml.qnode(dev)
        def circuit_1(params, generators=None):  # generators will be passed as a keyword arg
            ansatz(params, generators)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        @qml.qnode(dev)
        def circuit_2(params, generators=None):  # generators will be passed as a keyword arg
            ansatz(params, generators)
            return qml.expval(qml.PauliX(0))

        def cost_fn(params, generators):
            Z_1, Y_2 = circuit_1(params, generators=generators)
            X_1 = circuit_2(params, generators=generators)
            return 0.5 * Y_2 + 0.8 * Z_1 - 0.2 * X_1

        def f_best_gen(x):
            return cost_fn(x, optimal_generators)

        optimal_x_start = x_start.copy()

        # after four steps the optimzer should find the optimal generators/x_start values
        for _ in range(4):
            x_start, generators = rotoselect_opt.step(cost_fn, x_start, generators)
            optimal_x_start = self.rotosolve_step(f_best_gen, optimal_x_start)

        assert np.allclose(x_start, optimal_x_start, atol=tol)
        assert generators == optimal_generators

    def test_rotoselect_optimizer_raises(self):
        """Tests that step my raise an error."""
        rotoselect_opt = RotoselectOptimizer()

        def cost_fn():
            return None

        with pytest.raises(ValueError, match="must be equal to the number of generators"):
            rotoselect_opt.step(cost_fn, [0.2], [qml.PauliX, qml.PauliZ])

    @pytest.mark.slow
    @pytest.mark.parametrize("x_start", [[1.2, 0.2], [-0.62, -2.1], [0.05, 0.8]])
    def test_keywords_rotoselect(self, x_start, tol):
        """test rotoselect accepts keywords"""
        rotoselect_opt = RotoselectOptimizer()
        generators = [qml.RY, qml.RX]
        possible_generators = [qml.RX, qml.RY, qml.RZ]
        rotoselect_opt.possible_generators = possible_generators

        dev = qml.device("default.qubit", wires=2)

        def ansatz(params, generators):
            generators[0](params[0], wires=0)
            generators[1](params[1], wires=1)
            qml.CNOT(wires=[0, 1])

        @qml.qnode(dev)
        def circuit_1(params, generators=None):  # generators will be passed as a keyword arg
            ansatz(params, generators)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        @qml.qnode(dev)
        def circuit_2(params, generators=None):  # generators will be passed as a keyword arg
            ansatz(params, generators)
            return qml.expval(qml.PauliX(0))

        def cost_fn(params, generators, shift=0.0):
            Z_1, Y_2 = circuit_1(params, generators=generators)
            X_1 = circuit_2(params, generators=generators)
            return 0.5 * (Y_2 - shift) ** 2 + 0.8 * (Z_1 - shift) ** 2 - 0.2 * (X_1 - shift) ** 2

        params_new, *_ = rotoselect_opt.step_and_cost(cost_fn, x_start, generators, shift=0.0)
        params_new2, _, res_new2 = rotoselect_opt.step_and_cost(
            cost_fn, x_start, generators, shift=1.0
        )

        assert not np.allclose(params_new, params_new2, atol=tol)
        assert np.allclose(res_new2, cost_fn(x_start, generators, shift=1.0), atol=tol)

    @staticmethod
    def rotosolve_step(f, x):
        """Helper function to test the Rotosolve and Rotoselect optimizers"""
        # make sure that x is an array
        if np.ndim(x) == 0:
            x = np.array([x])

        # helper function for x[d] = theta
        def insert(xf, d, theta):
            xf[d] = theta
            return xf

        for d, _ in enumerate(x):
            H_0 = float(f(insert(x, d, 0)))
            H_p = float(f(insert(x, d, np.pi / 2)))
            H_m = float(f(insert(x, d, -np.pi / 2)))
            a = onp.arctan2(2 * H_0 - H_p - H_m, H_p - H_m)

            x[d] = -np.pi / 2 - a

            if x[d] <= -np.pi:
                x[d] += 2 * np.pi
        return x
