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
"""Tests for the QNG optimizer"""
# pylint: disable=too-few-public-methods
import pytest
import scipy as sp

import pennylane as qml
from pennylane import numpy as np


class TestExceptions:
    """Test exceptions are raised for incorrect usage"""

    def test_obj_func_not_a_qnode(self):
        """Test that if the objective function is not a
        QNode, an error is raised."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost(a):
            return circuit(a)

        opt = qml.QNGOptimizer()
        params = 0.5

        with pytest.raises(
            ValueError,
            match="The objective function must either be encoded as a single QNode or an ExpvalCost object",
        ):
            opt.step(cost, params)


class TestOptimize:
    """Test basic optimization integration"""

    def test_step_and_cost_autograd(self):
        """Test that the correct cost and step is returned via the
        step_and_cost method for the QNG optimizer"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        var = np.array([0.011, 0.012])
        opt = qml.QNGOptimizer(stepsize=0.01)

        step1, res = opt.step_and_cost(circuit, var)
        step2 = opt.step(circuit, var)

        expected = circuit(var)
        expected_step = var - opt.stepsize * 4 * qml.grad(circuit)(var)
        assert np.all(res == expected)
        assert np.allclose(step1, expected_step)
        assert np.allclose(step2, expected_step)

    def test_step_and_cost_autograd_with_gen_hamiltonian(self):
        """Test that the correct cost and step is returned via the
        step_and_cost method for the QNG optimizer when the generator
        of an operator is a Hamiltonian"""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        var = np.array([0.011, 0.012])
        opt = qml.QNGOptimizer(stepsize=0.01)

        step1, res = opt.step_and_cost(circuit, var)
        step2 = opt.step(circuit, var)

        expected = circuit(var)
        expected_step = var - opt.stepsize * 4 * qml.grad(circuit)(var)
        assert np.all(res == expected)
        assert np.allclose(step1, expected_step)
        assert np.allclose(step2, expected_step)

    def test_step_and_cost_with_grad_fn_grouped_input(self):
        """Test that the correct cost and update is returned via the step_and_cost
        method for the QNG optimizer when providing an explicit grad_fn.
        Using a circuit with a single input containing all parameters."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        var = np.array([0.011, 0.012])
        opt = qml.QNGOptimizer(stepsize=0.01)

        # With autograd gradient function
        grad_fn = qml.grad(circuit)
        step1, cost1 = opt.step_and_cost(circuit, var, grad_fn=grad_fn)
        step2 = opt.step(circuit, var, grad_fn=grad_fn)

        # With more custom gradient function, forward has to be computed explicitly.
        grad_fn = lambda param: np.array(qml.grad(circuit)(param))
        step3, cost2 = opt.step_and_cost(circuit, var, grad_fn=grad_fn)
        opt.step(circuit, var, grad_fn=grad_fn)
        expected_step = var - opt.stepsize * 4 * grad_fn(var)
        expected_cost = circuit(var)

        for step in [step1, step2, step3, step3]:
            assert np.allclose(step, expected_step)
        assert np.isclose(cost1, expected_cost)
        assert np.isclose(cost2, expected_cost)

    @pytest.mark.skip("QNGOptimizer is not yet implemented for split inputs.")
    def test_step_and_cost_with_grad_fn_split_input(self):
        """Test that the correct cost and update is returned via the step_and_cost
        method for the QNG optimizer when providing an explicit grad_fn.
        Using a circuit with multiple inputs containing the parameters."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(params_0, params_1):
            qml.RX(params_0, wires=0)
            qml.RY(params_1, wires=0)
            return qml.expval(qml.PauliZ(0))

        var = np.array([0.011, 0.012])
        opt = qml.QNGOptimizer(stepsize=0.01)

        # With autograd gradient function
        grad_fn = qml.grad(circuit)
        step1, cost1 = opt.step_and_cost(circuit, *var, grad_fn=grad_fn)
        step2 = opt.step(circuit, *var, grad_fn=grad_fn)

        # With more custom gradient function, forward has to be computed explicitly.
        grad_fn = lambda params_0, params_1: np.array(qml.grad(circuit)(params_0, params_1))
        step3, cost2 = opt.step_and_cost(circuit, *var, grad_fn=grad_fn)
        opt.step(circuit, *var, grad_fn=grad_fn)
        expected_step = var - opt.stepsize * 4 * grad_fn(*var)
        expected_cost = circuit(*var)

        for step in [step1, step2, step3, step3]:
            assert np.allclose(step, expected_step)
        assert np.isclose(cost1, expected_cost)
        assert np.isclose(cost2, expected_cost)

    def test_qubit_rotation(self, tol):
        """Test qubit rotation has the correct QNG value
        every step, the correct parameter updates,
        and correct cost after 200 steps"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        def gradient(params):
            """Returns the gradient of the above circuit"""
            da = -np.sin(params[0]) * np.cos(params[1])
            db = -np.cos(params[0]) * np.sin(params[1])
            return np.array([da, db])

        eta = 0.01
        init_params = np.array([0.011, 0.012])
        num_steps = 200

        opt = qml.QNGOptimizer(eta)
        theta = init_params

        # optimization for 200 steps total
        for _ in range(num_steps):
            theta_new = opt.step(circuit, theta)

            # check metric tensor
            res = opt.metric_tensor
            exp = np.diag([0.25, (np.cos(theta[0]) ** 2) / 4])
            assert np.allclose(res, exp, atol=tol, rtol=0)

            # check parameter update
            dtheta = eta * sp.linalg.pinvh(exp) @ gradient(theta)
            assert np.allclose(dtheta, theta - theta_new, atol=tol, rtol=0)

            theta = theta_new

        # check final cost
        assert np.allclose(circuit(theta), -0.9963791, atol=tol, rtol=0)

    @pytest.mark.skip("QNGOptimizer is not yet implemented for split inputs.")
    def test_single_qubit_vqe_using_expval_h_multiple_input_params(self, tol, recwarn):
        """Test single-qubit VQE by returning qml.expval(H) in the QNode and
        check for the correct QNG value every step, the correct parameter updates, and
        correct cost after 200 steps"""
        dev = qml.device("default.qubit", wires=1)
        coeffs = [1, 1]
        obs_list = [qml.PauliX(0), qml.PauliZ(0)]

        H = qml.Hamiltonian(coeffs=coeffs, observables=obs_list)

        @qml.qnode(dev)
        def circuit(x, y, wires=0):
            qml.RX(x, wires=wires)
            qml.RY(y, wires=wires)
            return qml.expval(H)

        eta = 0.01
        x = np.array(0.011, requires_grad=True)
        y = np.array(0.022, requires_grad=True)

        def gradient(params):
            """Returns the gradient"""
            da = -np.sin(params[0]) * (np.cos(params[1]) + np.sin(params[1]))
            db = np.cos(params[0]) * (np.cos(params[1]) - np.sin(params[1]))
            return np.array([da, db])

        eta = 0.01
        num_steps = 200

        opt = qml.QNGOptimizer(eta)

        # optimization for 200 steps total
        for _ in range(num_steps):
            theta = np.array([x, y])
            x, y = opt.step(circuit, x, y)

            # check metric tensor
            res = opt.metric_tensor
            exp = np.diag([0.25, (np.cos(x) ** 2) / 4])
            assert np.allclose(res, exp, atol=0.00001, rtol=0)

            # check parameter update
            theta_new = np.array([x, y])
            dtheta = eta * sp.linalg.pinvh(exp) @ gradient(theta)
            assert np.allclose(dtheta, theta - theta_new, atol=0.000001, rtol=0)

        # check final cost
        assert np.allclose(circuit(x, y), -1.41421356, atol=tol, rtol=0)
        assert len(recwarn) == 0
