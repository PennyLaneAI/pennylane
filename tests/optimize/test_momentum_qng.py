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
"""Tests for the Momentum-QNG optimizer"""
# pylint: disable=too-few-public-methods
import pytest
import scipy as sp

import pennylane as qml
from pennylane import numpy as np


class TestBasics:
    """Test basic properties of the MomentumQNGOptimizer."""

    def test_initialization_default(self):
        """Test that initializing MomentumQNGOptimizer with default values works."""
        opt = qml.MomentumQNGOptimizer()
        assert opt.stepsize == 0.01
        assert opt.approx == "block-diag"
        assert opt.lam == 0
        assert opt.momentum == 0.9
        assert opt.accumulation is None
        assert opt.metric_tensor is None

    def test_initialization_custom_values(self):
        """Test that initializing MomentumQNGOptimizer with custom values works."""
        opt = qml.MomentumQNGOptimizer(stepsize=0.05, momentum=0.8, approx="diag", lam=1e-9)
        assert opt.stepsize == 0.05
        assert opt.approx == "diag"
        assert opt.lam == 1e-9
        assert opt.momentum == 0.8
        assert opt.accumulation is None
        assert opt.metric_tensor is None


class TestOptimize:
    """Test basic optimization integration"""

    @pytest.mark.parametrize("rho", [0.9, 0.0])
    def test_step_and_cost(self, rho):
        """Test that the correct cost and step is returned after 8 optimization steps via the
        step_and_cost method for the MomentumQNG optimizer"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        var = np.array([0.11, 0.412])
        stepsize = 0.01
        # Create two optimizers so that the opt.accumulation state does not
        # interact between tests for step_and_cost and for step.
        opt1 = qml.MomentumQNGOptimizer(stepsize=stepsize, momentum=rho)
        opt2 = qml.MomentumQNGOptimizer(stepsize=stepsize, momentum=rho)

        var1 = var2 = var
        accum = np.zeros_like(var)

        for _ in range(4):
            var1, res = opt1.step_and_cost(circuit, var1)
            var2 = opt2.step(circuit, var2)
            # Analytic expressions
            mt = np.array([0.25, (np.cos(2 * var[0]) + 1) / 8])
            assert np.allclose(circuit(var), res)
            assert np.allclose(opt1.metric_tensor, np.diag(mt))
            assert np.allclose(opt2.metric_tensor, np.diag(mt))
            accum = rho * accum + stepsize * qml.grad(circuit)(var) / mt
            var -= accum
            assert np.allclose([var1, var2], var)

    def test_step_and_cost_autograd_with_gen_hamiltonian(self):
        """Test that the correct cost and step is returned after 8 optimization steps via the
        step_and_cost method for the MomentumQNG optimizer when the generator
        of an operator is a Hamiltonian"""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        var = np.array([0.311, -0.52])
        stepsize = 0.05
        momentum = 0.7
        # Create two optimizers so that the opt.accumulation state does not
        # interact between tests for step_and_cost and for step.
        opt1 = qml.MomentumQNGOptimizer(stepsize=stepsize, momentum=momentum)
        opt2 = qml.MomentumQNGOptimizer(stepsize=stepsize, momentum=momentum)

        var1 = var2 = var
        accum = np.zeros_like(var)
        expected_mt = np.array([1 / 16, 1 / 4])

        for _ in range(4):
            var1, res = opt1.step_and_cost(circuit, var1)
            var2 = opt2.step(circuit, var2)
            # Analytic expressions
            assert np.allclose(circuit(var), res)
            assert np.allclose(opt1.metric_tensor, np.diag(expected_mt))
            assert np.allclose(opt2.metric_tensor, np.diag(expected_mt))
            accum = momentum * accum + stepsize * qml.grad(circuit)(var) / expected_mt
            var -= accum
            assert np.allclose([var1, var2], var)

    @pytest.mark.parametrize("split_input", [False, True])
    def test_step_and_cost_with_grad_fn_grouped_and_split(self, split_input):
        """Test that the correct cost and update is returned after 8 optimization steps via the step_and_cost
        method for the MomentumQNG optimizer when providing an explicit grad_fn.
        Using a circuit with a single input containing all parameters."""
        dev = qml.device("default.qubit", wires=1)

        # Flat variables used for split_input=True and False
        var = np.array([0.911, 0.512])
        accum = np.zeros_like(var)

        if split_input:

            @qml.qnode(dev)
            def circuit(params_0, params_1):
                qml.RX(params_0, wires=0)
                qml.RY(params_1, wires=0)
                return qml.expval(qml.PauliZ(0))

            args = var
        else:

            @qml.qnode(dev)
            def circuit(params):
                qml.RX(params[0], wires=0)
                qml.RY(params[1], wires=0)
                return qml.expval(qml.PauliZ(0))

            args = (var,)

        stepsize = 0.05
        momentum = 0.7
        # Create multiple optimizers so that the opt.accumulation state does not
        # interact between tests for step_and_cost and for step.
        opt1 = qml.MomentumQNGOptimizer(stepsize=stepsize, momentum=momentum)
        opt2 = qml.MomentumQNGOptimizer(stepsize=stepsize, momentum=momentum)
        opt3 = qml.MomentumQNGOptimizer(stepsize=stepsize, momentum=momentum)
        opt4 = qml.MomentumQNGOptimizer(stepsize=stepsize, momentum=momentum)

        args1 = args2 = args3 = args4 = args

        # With autograd gradient function
        grad_fn1 = qml.grad(circuit)
        for _ in range(4):
            args1, cost1 = opt1.step_and_cost(circuit, *args1, grad_fn=grad_fn1)
            args2 = opt2.step(circuit, *args2, grad_fn=grad_fn1)
            if not split_input:
                args1 = (args1,)
                args2 = (args2,)

        mt1 = opt1.metric_tensor
        mt2 = opt2.metric_tensor

        # With more custom gradient function, forward has to be computed explicitly.
        def grad_fn2(*args):
            return np.array(qml.grad(circuit)(*args))

        for _ in range(4):
            args3, cost2 = opt3.step_and_cost(circuit, *args3, grad_fn=grad_fn2)
            args4 = opt4.step(circuit, *args4, grad_fn=grad_fn2)

            # Compute expected metric tensor, update step etc
            expected_mt_diag = np.array([0.25, (np.cos(var[0]) ** 2) / 4])
            accum = momentum * accum + stepsize * grad_fn2(*args) / expected_mt_diag
            expected_cost = circuit(*args)
            # Update var and args
            var -= accum
            args = var if split_input else (var,)
            if not split_input:
                args3 = (args3,)
                args4 = (args4,)

        mt3 = opt3.metric_tensor
        mt4 = opt4.metric_tensor

        expected_mt = np.diag(expected_mt_diag)
        if split_input:
            expected_mt = (expected_mt[:1, :1], expected_mt[1:, 1:])

        for a in [args1, args2, args3, args4]:
            assert np.allclose(a, args)
        for mt in [mt1, mt2, mt3, mt4]:
            assert np.allclose(mt, expected_mt)
        assert np.isclose(cost1, expected_cost)
        assert np.isclose(cost2, expected_cost)

    @pytest.mark.parametrize("trainable_idx", [0, 1])
    def test_step_and_cost_split_input_one_trainable(self, trainable_idx):
        """Test that the correct cost and update is returned after 8 optimization steps via the step_and_cost
        method for the MomentumQNG optimizer when providing an explicit grad_fn or not.
        Using a circuit with multiple inputs, one of which is trainable."""

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(x, y):
            """A cost function with two arguments."""
            qml.RX(x, 0)
            qml.RY(-y, 0)
            return qml.expval(qml.Z(0))

        grad_fn = qml.grad(circuit)

        params = np.array(0.2, requires_grad=False), np.array(-0.8, requires_grad=False)
        params[trainable_idx].requires_grad = True
        opt = qml.MomentumQNGOptimizer(stepsize=0.01, momentum=0.9)

        params1 = params2 = params3 = params

        # Without manually provided functions
        for _ in range(4):
            params1, cost1 = opt.step_and_cost(circuit, *params1)

        opt = qml.MomentumQNGOptimizer(stepsize=0.01, momentum=0.9)
        for _ in range(4):
            params2 = opt.step(circuit, *params2)

        accum = np.zeros_like(params[0]), np.zeros_like(params[1])

        # Expectations
        def mt_inv(tr_idx, par):
            if tr_idx == 1:
                mt_i = 1 / (np.cos(2 * par[0]) + 1) * 8
            else:
                mt_i = 4
            return mt_i

        for _ in range(4):
            inv_mt = mt_inv(trainable_idx, params3)
            expected_cost = circuit(*params3)
            if trainable_idx == 0:
                accum = (opt.momentum * accum[0] + opt.stepsize * grad_fn(*params3) * inv_mt, 0.0)
                params3 = (params3[0] - accum[0], params3[1])
            elif trainable_idx == 1:
                accum = (0.0, opt.momentum * accum[1] + opt.stepsize * grad_fn(*params3) * inv_mt)
                params3 = (params3[0], params3[1] - accum[1])

        for p in [params1, params2]:
            assert np.allclose(params3, p)
        assert np.isclose(cost1, expected_cost)

    def test_qubit_rotation(self, tol):
        """Test qubit rotation has the correct Momentum-QNG value
        every step, the correct parameter updates,
        and correct cost after a few steps"""
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

        eta = 0.05
        rho = 0.9
        init_params = np.array([0.011, 0.012])
        num_steps = 80

        opt = qml.MomentumQNGOptimizer(stepsize=eta, momentum=rho)
        theta = init_params
        dtheta = np.zeros_like(init_params)

        for _ in range(num_steps):
            theta_new = opt.step(circuit, theta)

            # check metric tensor
            res = opt.metric_tensor
            exp = np.diag([0.25, (np.cos(theta[0]) ** 2) / 4])
            assert np.allclose(res, exp, atol=tol, rtol=0)

            # check parameter update
            dtheta *= rho
            dtheta += eta * sp.linalg.pinvh(exp) @ gradient(theta)
            assert np.allclose(dtheta, theta - theta_new, atol=tol, rtol=0)

            theta = theta_new

        # check final cost
        assert np.allclose(circuit(theta), -1, atol=1e-4)

    def test_single_qubit_vqe_using_expval_h_multiple_input_params(self, tol):
        """Test single-qubit VQE by returning qml.expval(H) in the QNode and
        check for the correct MomentumQNG value every step, the correct parameter updates, and
        correct cost after a few steps"""
        dev = qml.device("default.qubit", wires=1)
        coeffs = [1, 1]
        obs_list = [qml.PauliX(0), qml.PauliZ(0)]

        H = qml.Hamiltonian(coeffs=coeffs, observables=obs_list)

        @qml.qnode(dev)
        def circuit(x, y, wires=0):
            qml.RX(x, wires=wires)
            qml.RY(y, wires=wires)
            return qml.expval(H)

        eta = 0.02
        rho = 0.7
        x = np.array(0.011, requires_grad=True)
        y = np.array(0.022, requires_grad=True)

        def gradient(params):
            """Returns the gradient"""
            da = -np.sin(params[0]) * (np.cos(params[1]) + np.sin(params[1]))
            db = np.cos(params[0]) * (np.cos(params[1]) - np.sin(params[1]))
            return np.array([da, db])

        num_steps = 30

        opt = qml.MomentumQNGOptimizer(stepsize=eta, momentum=rho)

        theta = np.array([x, y])
        dtheta = np.zeros_like(theta)

        for _ in range(num_steps):
            theta = np.array([x, y])
            x, y = opt.step(circuit, x, y)

            # check metric tensor
            res = opt.metric_tensor
            exp = (np.array([[0.25]]), np.array([[(np.cos(2 * theta[0]) + 1) / 8]]))
            assert np.allclose(res, exp)

            # check parameter update
            theta_new = (x, y)
            grad = gradient(theta)
            dtheta *= rho
            dtheta += tuple(eta * g / e[0, 0] for e, g in zip(exp, grad))
            assert np.allclose(dtheta, theta - theta_new)

        # check final cost
        assert np.allclose(circuit(x, y), qml.eigvals(H).min(), atol=tol, rtol=0)
