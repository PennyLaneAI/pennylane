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
from pennylane.optimize.qng import _flatten_np, _unflatten_np


class TestBasics:
    """Test basic properties of the QNGOptimizer."""

    def test_initialization_default(self):
        """Test that initializing QNGOptimizer with default values works."""
        opt = qml.QNGOptimizer()
        assert opt.stepsize == 0.01
        assert opt.approx == "block-diag"
        assert opt.lam == 0
        assert opt.metric_tensor is None

    def test_initialization_custom_values(self):
        """Test that initializing QNGOptimizer with custom values works."""
        opt = qml.QNGOptimizer(stepsize=0.05, approx="diag", lam=1e-9)
        assert opt.stepsize == 0.05
        assert opt.approx == "diag"
        assert opt.lam == 1e-9
        assert opt.metric_tensor is None


class TestAttrsAffectingMetricTensor:
    """Test that the attributes `approx` and `lam`, which affect the metric tensor
    and its inversion, are used correctly."""

    def test_no_approx(self):
        """Test that the full metric tensor is used correctly for ``approx=None``."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(params):
            qml.RY(eta, wires=0)
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        opt = qml.QNGOptimizer(approx=None)
        eta = 0.7
        params = np.array([0.11, 0.412])
        new_params_no_approx = opt.step(circuit, params)
        opt_with_approx = qml.QNGOptimizer()
        new_params_block_approx = opt_with_approx.step(circuit, params)
        # Expected result, requires some manual calculation, compare analytic test cases page
        x = params[0]
        first_term = np.eye(2) / 4
        vec_potential = np.array([-0.5j * np.sin(eta), 0.5j * np.sin(x) * np.cos(eta)])
        second_term = np.real(np.outer(vec_potential.conj(), vec_potential))
        exp_mt = first_term - second_term

        assert np.allclose(opt.metric_tensor, exp_mt)
        assert np.allclose(opt_with_approx.metric_tensor, np.diag(np.diag(exp_mt)))
        assert not np.allclose(new_params_no_approx, new_params_block_approx)

    def test_lam(self):
        """Test that the regularization ``lam`` is used correctly."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(params):
            qml.RY(eta, wires=0)
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        lam = 1e-9
        opt = qml.QNGOptimizer(lam=lam, stepsize=1.0)
        eta = np.pi
        params = np.array([np.pi / 2, 0.412])
        new_params_with_lam = opt.step(circuit, params)
        opt_without_lam = qml.QNGOptimizer(stepsize=1.0)
        new_params_without_lam = opt_without_lam.step(circuit, params)
        # Expected result, requires some manual calculation, compare analytic test cases page
        x, y = params
        first_term = np.eye(2) / 4
        vec_potential = np.array([-0.5j * np.sin(eta), 0.5j * np.sin(x) * np.cos(eta)])
        second_term = np.real(np.outer(vec_potential.conj(), vec_potential))
        exp_mt = first_term - second_term

        assert np.allclose(opt.metric_tensor, exp_mt + np.eye(2) * lam)
        assert np.allclose(opt_without_lam.metric_tensor, np.diag(np.diag(exp_mt)))
        # With regularization, y can be updated. Without regularization it can not.
        assert np.isclose(new_params_without_lam[1], y)
        assert not np.isclose(new_params_with_lam[1], y, atol=1e-11, rtol=0.0)


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
            match="The objective function must be encoded as a single QNode.",
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

        var = np.array([0.31, 0.842])
        opt = qml.QNGOptimizer(stepsize=0.05)

        expected_mt_diag = np.array([0.25, (np.cos(var[0]) ** 2) / 4])
        expected_res = circuit(var)
        expected_step = var - opt.stepsize * qml.grad(circuit)(var) / expected_mt_diag

        step1, res = opt.step_and_cost(circuit, var)
        assert np.allclose(opt.metric_tensor, np.diag(expected_mt_diag))
        step2 = opt.step(circuit, var)
        assert np.allclose(opt.metric_tensor, np.diag(expected_mt_diag))

        assert np.allclose(res, expected_res)
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

        var = np.array([0.311, -0.52])
        opt = qml.QNGOptimizer(stepsize=0.05)

        expected_mt = np.diag([1 / 16, 1 / 4])
        expected_res = circuit(var)
        expected_step = var - opt.stepsize * np.linalg.pinv(expected_mt) @ qml.grad(circuit)(var)

        step1, res = opt.step_and_cost(circuit, var)
        assert np.allclose(opt.metric_tensor, expected_mt)
        step2 = opt.step(circuit, var)
        assert np.allclose(opt.metric_tensor, expected_mt)

        assert np.allclose(res, expected_res)
        assert np.allclose(step1, expected_step)
        assert np.allclose(step2, expected_step)

    @pytest.mark.parametrize("split_input", [False, True])
    def test_step_and_cost_with_grad_fn_grouped_and_split(self, split_input):
        """Test that the correct cost and update is returned via the step_and_cost
        method for the QNG optimizer when providing an explicit grad_fn."""
        dev = qml.device("default.qubit", wires=1)

        var = np.array([0.911, 0.512])
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

        opt = qml.QNGOptimizer(stepsize=0.04)

        # With autograd gradient function
        grad_fn1 = qml.grad(circuit)
        step1, cost1 = opt.step_and_cost(circuit, *args, grad_fn=grad_fn1)
        mt1 = opt.metric_tensor
        step2 = opt.step(circuit, *args, grad_fn=grad_fn1)
        mt2 = opt.metric_tensor

        # With more custom gradient function, forward has to be computed explicitly.
        def grad_fn2(*args):
            return np.array(qml.grad(circuit)(*args))

        step3, cost2 = opt.step_and_cost(circuit, *args, grad_fn=grad_fn2)
        mt3 = opt.metric_tensor
        step4 = opt.step(circuit, *args, grad_fn=grad_fn2)
        mt4 = opt.metric_tensor

        expected_mt_diag = np.array([0.25, (np.cos(var[0]) ** 2) / 4])
        expected_step = var - opt.stepsize * grad_fn2(*args) / expected_mt_diag
        expected_mt = np.diag(expected_mt_diag)
        if split_input:
            expected_mt = (expected_mt[:1, :1], expected_mt[1:, 1:])
        expected_cost = circuit(*args)

        for step in [step1, step2, step3, step4]:
            assert np.allclose(step, expected_step)
        for mt in [mt1, mt2, mt3, mt4]:
            assert np.allclose(mt, expected_mt)
        assert np.isclose(cost1, expected_cost)
        assert np.isclose(cost2, expected_cost)

    @pytest.mark.parametrize("trainable_idx", [0, 1])
    def test_step_and_cost_split_input_one_trainable(self, trainable_idx):
        """Test that the correct cost and update is returned via the step_and_cost
        method for the QNG optimizer when providing an explicit grad_fn or not.
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
        opt = qml.QNGOptimizer(stepsize=0.01)

        # Without manually provided functions
        step1, cost1 = opt.step_and_cost(circuit, *params)
        step2 = opt.step(circuit, *params)

        # With modified autograd gradient function
        fake_grad_fn = lambda *args, **kwargs: grad_fn(*args, **kwargs) * 2
        step3, cost2 = opt.step_and_cost(circuit, *params, grad_fn=fake_grad_fn)
        step4 = opt.step(circuit, *params, grad_fn=fake_grad_fn)

        # With modified metric tensor function
        fake_mt_fn = lambda *args, **kwargs: mt_fn(*args, **kwargs) * 4
        step5 = opt.step(circuit, *params, metric_tensor_fn=fake_mt_fn)

        # Expectations
        if trainable_idx == 1:
            mt_inv = 1 / (np.cos(2 * params[0]) + 1) * 8
        else:
            mt_inv = 4
        exact_update = -opt.stepsize * grad_fn(*params) * mt_inv
        factors = [1.0, 1.0, 2.0, 2.0, 0.25]
        expected_cost = circuit(*params)

        for factor, step in zip(factors, [step1, step2, step3, step4, step5]):
            expected_step = tuple(
                par + exact_update * factor if i == trainable_idx else par
                for i, par in enumerate(params)
            )
            assert np.allclose(step, expected_step)
        assert np.isclose(cost1, expected_cost)
        assert np.isclose(cost2, expected_cost)

    def test_qubit_rotation(self, tol):
        """Test qubit rotation has the correct QNG value
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

        eta = 0.2
        init_params = np.array([0.011, 0.012])
        num_steps = 15

        opt = qml.QNGOptimizer(eta)
        theta = init_params

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
        assert np.allclose(circuit(theta), -1)

    def test_single_qubit_vqe_using_expval_h_multiple_input_params(self, tol, recwarn):
        """Test single-qubit VQE by returning qml.expval(H) in the QNode and
        check for the correct QNG value every step, the correct parameter updates, and
        correct cost after a few steps"""
        dev = qml.device("default.qubit", wires=1)
        coeffs = [1, 1]
        obs_list = [qml.PauliX(0), qml.PauliZ(0)]

        H = qml.ops.LinearCombination(coeffs=coeffs, observables=obs_list)

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

        eta = 0.2
        num_steps = 10

        opt = qml.QNGOptimizer(eta)

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
            dtheta = tuple(eta * g / e[0, 0] for e, g in zip(exp, grad))
            assert np.allclose(dtheta, theta - theta_new)

        # check final cost
        assert np.allclose(circuit(x, y), qml.eigvals(H).min(), atol=tol, rtol=0)
        assert len(recwarn) == 0


flat_dummy_array = np.linspace(-1, 1, 64)
test_shapes = [
    (64,),
    (64, 1),
    (32, 2),
    (16, 4),
    (8, 8),
    (16, 2, 2),
    (8, 2, 2, 2),
    (4, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
]


class TestFlatten:
    """Tests the flatten and unflatten functions"""

    @pytest.mark.parametrize("shape", test_shapes)
    def test_flatten(self, shape):
        """Tests that _flatten successfully flattens multidimensional arrays."""

        reshaped = np.reshape(flat_dummy_array, shape)
        flattened = np.array(list(_flatten_np(reshaped)))

        assert flattened.shape == flat_dummy_array.shape
        assert np.array_equal(flattened, flat_dummy_array)

    @pytest.mark.parametrize("shape", test_shapes)
    def test_unflatten(self, shape):
        """Tests that _unflatten successfully unflattens multidimensional arrays."""

        reshaped = np.reshape(flat_dummy_array, shape)
        unflattened = np.array(list(_unflatten_np(flat_dummy_array, reshaped)))

        assert unflattened.shape == reshaped.shape
        assert np.array_equal(unflattened, reshaped)

    def test_unflatten_error_unsupported_model(self):
        """Tests that unflatten raises an error if the given model is not supported"""

        with pytest.raises(TypeError, match="Unsupported type in the model"):
            model = lambda x: x  # not a valid model for unflatten
            _unflatten_np(flat_dummy_array, model)

    def test_unflatten_error_too_many_elements(self):
        """Tests that unflatten raises an error if the given iterable has
        more elements than the model"""

        reshaped = np.reshape(flat_dummy_array, (16, 2, 2))

        with pytest.raises(ValueError, match="Flattened iterable has more elements than the model"):
            _unflatten_np(np.concatenate([flat_dummy_array, flat_dummy_array]), reshaped)

    def test_flatten_wires(self):
        """Tests flattening a Wires object."""
        wires = qml.wires.Wires([3, 4])
        wires_int = [3, 4]

        wires = _flatten_np(wires)
        for i, wire in enumerate(wires):
            assert wires_int[i] == wire
