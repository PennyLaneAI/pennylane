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

        opt = qml.MomentumQNGOptimizer()
        params = 0.5

        with pytest.raises(
            ValueError,
            match="The objective function must be encoded as a single QNode.",
        ):
            opt.step(cost, params)


class TestOptimize:
    """Test basic optimization integration"""

    def test_step_and_cost_autograd(self):
        """Test that the correct cost and step is returned after 8 optimization steps via the
        step_and_cost method for the MomentumQNG optimizer"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        var = np.array([0.011, 0.012])
        opt = qml.MomentumQNGOptimizer(stepsize=0.01, momentum=0.9)
        
        var1 = var2 = var
        accum = [np.zeros_like(v) for v in var]
        
        for i in range(0, 8):
            var1, res = opt.step_and_cost(circuit, var1)
            var2 = opt.step(circuit, var2)
            accum = opt.momentum * accum + opt.stepsize * 4 * qml.grad(circuit)(var)
            var -= accum
            
        expected = circuit(var)
        
        assert np.all(res == expected)
        assert np.allclose(var1, var)
        assert np.allclose(var2, var)

    @pytest.mark.usefixtures("use_legacy_opmath")
    def test_step_and_cost_autograd_with_gen_hamiltonian_legacy_opmath(self):
        """Test that the correct cost and step is returned after 8 optimization steps via the
        step_and_cost method for the MomentumQNG optimizer when the generator
        of an operator is a Hamiltonian"""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit(params):
            qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        var = np.array([0.011, 0.012])
        opt = qml.MomentumQNGOptimizer(stepsize=0.01, momentum=0.9)

        var1 = var2 = var
        accum = [np.zeros_like(v) for v in var]
        
        for i in range(0, 8):
            var1, res = opt.step_and_cost(circuit, var1)
            var2 = opt.step(circuit, var2)
            accum = opt.momentum * accum + opt.stepsize * 4 * qml.grad(circuit)(var)
            var -= accum
            
        expected = circuit(var)
        
        assert np.all(res == expected)
        assert np.allclose(var1, var)
        assert np.allclose(var2, var)

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

        var = np.array([0.011, 0.012])
        opt = qml.MomentumQNGOptimizer(stepsize=0.01, momentum=0.9)

        var1 = var2 = var
        accum = [np.zeros_like(v) for v in var]
        
        for i in range(0, 8):
            var1, res = opt.step_and_cost(circuit, var1)
            var2 = opt.step(circuit, var2)
            accum = opt.momentum * accum + opt.stepsize * 4 * qml.grad(circuit)(var)
            var -= accum
            
        expected = circuit(var)
        
        assert np.all(res == expected)
        assert np.allclose(var1, var)
        assert np.allclose(var2, var)

    def test_step_and_cost_with_grad_fn_grouped_input(self):
        """Test that the correct cost and update is returned after 8 optimization steps via the step_and_cost
        method for the MomentumQNG optimizer when providing an explicit grad_fn.
        Using a circuit with a single input containing all parameters."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        var = np.array([0.011, 0.012])
        opt = qml.MomentumQNGOptimizer(stepsize=0.01, momentum=0.9)
        
        var1 = var2 = var3 = var4 = var
        accum = [np.zeros_like(v) for v in var]

        # With autograd gradient function
        for i in range(0, 8):
            grad_fn1 = qml.grad(circuit)(var1)
            var1, cost1 = opt.step_and_cost(circuit, var1, grad_fn=grad_fn1)
            grad_fn1 = qml.grad(circuit)(var2)
            var2 = opt.step(circuit, var2, grad_fn=grad_fn1)

        # With more custom gradient function, forward has to be computed explicitly.
        def grad_fn2(param):
            return np.array(qml.grad(circuit)(param))

        # grad_fn = lambda param: np.array(qml.grad(circuit)(param))
        for i in range(0, 8):
            grad_fn2 = grad_fn2(var3)
            var3, cost2 = opt.step_and_cost(circuit, var3, grad_fn=grad_fn2)
            grad_fn2 = grad_fn2(var4)
            var4 = opt.step(circuit, var4, grad_fn=grad_fn2)
            accum = opt.momentum * accum + opt.stepsize * 4 * grad_fn2(var)
            var -= accum
            
        expected_cost = circuit(var)

        for v in [var1, var2, var3, var4]:
            assert np.allclose(v, var)
        assert np.isclose(cost1, expected_cost)
        assert np.isclose(cost2, expected_cost)

    def test_step_and_cost_with_grad_fn_split_input(self):
        """Test that the correct cost and update is returned after 8 optimization steps via the step_and_cost
        method for the MomentumQNG optimizer when providing an explicit grad_fn.
        Using a circuit with multiple inputs containing the parameters."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(params_0, params_1):
            qml.RX(params_0, wires=0)
            qml.RY(params_1, wires=0)
            return qml.expval(qml.PauliZ(0))

        var = np.array([0.011, 0.012])
        opt = qml.MomentumQNGOptimizer(stepsize=0.01, momentum=0.9)

        # With autograd gradient function
        grad_fn1 = qml.grad(circuit)

        var1 = var2 = var3 = var4 = var
        accum = [np.zeros_like(v) for v in var]

        # With autograd gradient function
        for i in range(0, 8):
            grad_fn1 = qml.grad(circuit)(*var1)
            var1, cost1 = opt.step_and_cost(circuit, *var1, grad_fn=grad_fn1)
            grad_fn1 = qml.grad(circuit)(*var2)
            var2 = opt.step(circuit, *var2, grad_fn=grad_fn1)
        

        # With more custom gradient function, forward has to be computed explicitly.
        def grad_fn2(params_0, params_1):
            return np.array(qml.grad(circuit)(params_0, params_1))

        for i in range(0, 8):
            grad_fn2 = grad_fn2(*var3)
            var3, cost2 = opt.step_and_cost(circuit, *var3, grad_fn=grad_fn2)
            grad_fn2 = grad_fn2(*var4)
            var4 = opt.step(circuit, *var4, grad_fn=grad_fn2)
            accum = opt.momentum * accum + opt.stepsize * 4 * grad_fn2(*var)
            var -= accum
            
        expected_cost = circuit(*var)

        for v in [var1, var2, var3, var4]:
            assert np.allclose(v, var)
        
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
        mt_fn = qml.metric_tensor(circuit)

        params = np.array(0.2, requires_grad=False), np.array(-0.8, requires_grad=False)
        params[trainable_idx].requires_grad = True
        opt = qml.MomentumQNGOptimizer(stepsize=0.01, momentum=0.9)

        params1 = params2 = params3 = params4 = params5 = params6 = params
        
        # Without manually provided functions
        for i in range(0, 8):
            params1, cost1 = opt.step_and_cost(circuit, *params1)
            params2 = opt.step(circuit, *params2)
        
        # With modified autograd gradient function
        fake_grad_fn = lambda *args, **kwargs: grad_fn(*args, **kwargs) * 2
        
        for i in range(0, 8):
            params3, cost2 = opt.step_and_cost(circuit, *params3, grad_fn=fake_grad_fn)
            params4 = opt.step(circuit, *params4, grad_fn=fake_grad_fn)
        
        # With modified metric tensor function
        fake_mt_fn = lambda *args, **kwargs: mt_fn(*args, **kwargs) * 4

        for i in range(0, 8):
            params5 = opt.step(circuit, *params5, metric_tensor_fn=fake_mt_fn)    
        
        accum = np.zeros_like(params[0]), np.zeros_like(params[1])
        # Expectations
        def mt_inv(tr_idx, par):
            if tr_idx == 1:
                mt_i = 1 / (np.cos(2 * par[0]) + 1) * 8
            else:
                mt_i = 4
            return mt_i
        
        for i in range(0, 8):
            accum = opt.momentum * accum + opt.stepsize * grad_fn(*params6) * mt_inv(trainable_idx, *params6)
            params6 -= accum
            
        exact_update = params6 - params
        
        factors = [1.0, 1.0, 2.0, 2.0, 0.25]
        

        for factor, step in zip(factors, [params1, params2, params3, params4, params5]):
            expected_step = tuple(
                par + exact_update * factor if i == trainable_idx else par
                for i, par in enumerate(params)
            )
            assert np.allclose(step, expected_step)
        expected_cost = circuit(*params)
        assert np.isclose(cost1, expected_cost)
        assert np.isclose(cost2, expected_cost)

    @pytest.mark.slow
    def test_qubit_rotation(self, tol):
        """Test qubit rotation has the correct MomentumQNG value
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
        rho = 0.9
        init_params = np.array([0.011, 0.012])
        num_steps = 200

        opt = qml.MomentumQNGOptimizer(stepsize=eta, momentum=rho)
        theta = init_params
        dtheta = np.zeros_like(init_params)
        # optimization for 200 steps total
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
        assert np.allclose(circuit(theta), -0.9963791, atol=tol, rtol=0)

    def test_single_qubit_vqe_using_expval_h_multiple_input_params(self, tol, recwarn):
        """Test single-qubit VQE by returning qml.expval(H) in the QNode and
        check for the correct MomentumQNG value every step, the correct parameter updates, and
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
        rho = 0.9
        x = np.array(0.011, requires_grad=True)
        y = np.array(0.022, requires_grad=True)

        def gradient(params):
            """Returns the gradient"""
            da = -np.sin(params[0]) * (np.cos(params[1]) + np.sin(params[1]))
            db = np.cos(params[0]) * (np.cos(params[1]) - np.sin(params[1]))
            return np.array([da, db])

        num_steps = 200

        opt = qml.MomentumQNGOptimizer(stepsize=eta, momentum=rho)

        theta = np.array([x, y])
        dtheta = np.zeros_like(theta)

        # optimization for 200 steps total
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
        assert np.allclose(circuit(x, y), -1.41421356, atol=tol, rtol=0)
        assert len(recwarn) == 0
