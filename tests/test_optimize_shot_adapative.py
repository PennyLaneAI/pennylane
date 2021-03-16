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
"""Tests for the shot adaptive optimizer"""
import pytest

import pennylane as qml
from pennylane import numpy as np


@pytest.fixture(autouse=True)
def tape_mode_only():
    """Run the test in tape mode"""
    qml.enable_tape()
    yield
    qml.disable_tape()


class TestExceptions:
    """Test exceptions are raised for incorrect usage"""

    def test_analytic_device_error(self):
        """Test that an exception is raised if an analytic device is used"""
        H = qml.Hamiltonian([0.3, 0.1], [qml.PauliX(0), qml.PauliZ(0)])
        dev = qml.device("default.qubit", wires=1, analytic=True)
        expval_cost = qml.ExpvalCost(lambda x, **kwargs: qml.RX(x, wires=0), H, dev)

        opt = qml.ShotAdaptiveOptimizer(min_shots=10)

        # test expval cost
        with pytest.raises(ValueError, match="can only be used with non-analytic"):
            opt.step(expval_cost, 0.5)

        # test qnode cost
        with pytest.raises(ValueError, match="can only be used with non-analytic"):
            opt.step(expval_cost.qnodes[0], 0.5)

    def test_learning_error(self):
        """Test that an exception is raised if the learning rate is beyond the
        lipschitz bound"""
        coeffs = [0.3, 0.1]
        H = qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)])
        dev = qml.device("default.qubit", wires=1, analytic=False)
        expval_cost = qml.ExpvalCost(lambda x, **kwargs: qml.RX(x, wires=0), H, dev)

        opt = qml.ShotAdaptiveOptimizer(min_shots=10, stepsize=100.)

        # lipschitz constant is given by sum(|coeffs|)
        lipschitz = np.sum(np.abs(coeffs))

        assert opt._stepsize > 2 / lipschitz

        with pytest.raises(ValueError, match=f"The learning rate must be less than {2 / lipschitz}"):
            opt.step(expval_cost, 0.5)

        # for a single QNode, the lipschitz constant is simply 1
        opt = qml.ShotAdaptiveOptimizer(min_shots=10, stepsize=100.)
        with pytest.raises(ValueError, match=f"The learning rate must be less than {2 / 1}"):
            opt.step(expval_cost.qnodes[0], 0.5)

    def test_unknown_objective_function(self):
        """Test that an exception is raised if an unknown objective function is passed"""
        dev = qml.device("default.qubit", wires=1, analytic=False)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost(x):
            return np.sin(circuit(x))

        opt = qml.ShotAdaptiveOptimizer(min_shots=10)

        # test expval cost
        with pytest.raises(ValueError, match="The objective function must either be encoded as a single QNode"):
            opt.step(cost, 0.5)

        # defining the device attribute allows it to proceed
        cost.device = circuit.device
        new_x = opt.step(cost, 0.5)

        assert isinstance(new_x, float)


class TestSingleShotGradientIntegration:
    """Integration tests to ensure that the single shot gradient is correctly computed
    for a variety of argument types."""

    dev = qml.device("default.qubit", wires=1, analytic=False)
    H = qml.Hamiltonian([1.0], [qml.PauliZ(0)])

    ansatz = lambda x, **kwargs: qml.RX(x, wires=0)
    expval_cost = qml.ExpvalCost(ansatz, H, dev)

    @qml.qnode(dev)
    def qnode(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    @pytest.mark.parametrize("cost_fn", [qnode, expval_cost])
    def test_single_argument_step(self, cost_fn, mocker, monkeypatch):
        """Test that a simple QNode with a single argument correctly performs an optimization step,
        and that the single-shot gradients generated have the correct shape"""

        opt = qml.ShotAdaptiveOptimizer(min_shots=10)
        spy_single_shot_expval = mocker.spy(opt, "_single_shot_expval_gradients")
        spy_single_shot_qnodes = mocker.spy(opt, "_single_shot_qnode_gradients")
        spy_grad = mocker.spy(opt, "compute_grad")

        x_init = 0.5
        new_x = opt.step(cost_fn, x_init)

        assert isinstance(new_x, float)
        assert new_x != x_init

        spy_grad.assert_called_once()

        if isinstance(cost_fn, qml.ExpvalCost):
            spy_single_shot_expval.assert_called_once()
            single_shot_grads = opt._single_shot_expval_gradients(cost_fn, [x_init], {})
        else:
            spy_single_shot_qnodes.assert_called_once()
            single_shot_grads = opt._single_shot_qnode_gradients(cost_fn, [x_init], {})

        # assert single shot gradients are computed correctly
        assert len(single_shot_grads) == 1
        assert single_shot_grads[0].shape == (10,)

        # monkeypatch the optimizer to use the same single shot gradients
        # as previously
        monkeypatch.setattr(opt, "_single_shot_qnode_gradients", lambda *args, **kwargs: single_shot_grads)
        monkeypatch.setattr(opt, "_single_shot_expval_gradients", lambda *args, **kwargs: single_shot_grads)

        # reset the shot budget
        opt.s = [np.array(10)]

        # check that the gradient and variance are computed correctly
        grad, grad_variance = opt.compute_grad(cost_fn, [x_init], {})
        assert len(grad) == 1
        assert len(grad_variance) == 1
        assert np.allclose(grad, np.mean(single_shot_grads))
        assert np.allclose(grad_variance, np.var(single_shot_grads, ddof=1))

        # check that the gradient and variance are computed correctly
        # with a different shot budget
        opt.s = [np.array(5)]
        grad, grad_variance = opt.compute_grad(cost_fn, [x_init], {})
        assert len(grad) == 1
        assert len(grad_variance) == 1
        assert np.allclose(grad, np.mean(single_shot_grads[0][:5]))
        assert np.allclose(grad_variance, np.var(single_shot_grads[0][:5], ddof=1))

    def ansatz(x, **kwargs):
        qml.RX(x[0, 0], wires=0)
        qml.RY(x[0, 1], wires=0)
        qml.RZ(x[0, 2], wires=0)
        qml.RX(x[1, 0], wires=0)
        qml.RY(x[1, 1], wires=0)
        qml.RZ(x[1, 2], wires=0)

    expval_cost = qml.ExpvalCost(ansatz, H, dev)
    qnode = expval_cost.qnodes[0]

    @pytest.mark.parametrize("cost_fn", [qnode, expval_cost])
    def test_single_array_argument_step(self, cost_fn, mocker, monkeypatch):
        """Test that a simple QNode with a single array argument correctly performs an optimization step,
        and that the single-shot gradients generated have the correct shape"""
        opt = qml.ShotAdaptiveOptimizer(min_shots=10)
        spy_single_shot_expval = mocker.spy(opt, "_single_shot_expval_gradients")
        spy_single_shot_qnodes = mocker.spy(opt, "_single_shot_qnode_gradients")
        spy_grad = mocker.spy(opt, "compute_grad")

        x_init = np.array([[1., 2., 3.], [4., 5., 6.]])
        new_x = opt.step(cost_fn, x_init)

        assert isinstance(new_x, np.ndarray)
        assert not np.allclose(new_x, x_init)

        if isinstance(cost_fn, qml.ExpvalCost):
            spy_single_shot_expval.assert_called_once()
            single_shot_grads = opt._single_shot_expval_gradients(cost_fn, [x_init], {})
        else:
            spy_single_shot_qnodes.assert_called_once()
            single_shot_grads = opt._single_shot_qnode_gradients(cost_fn, [x_init], {})

        spy_grad.assert_called_once()

        # assert single shot gradients are computed correctly
        assert len(single_shot_grads) == 1
        assert single_shot_grads[0].shape == (10, 2, 3)

        # monkeypatch the optimizer to use the same single shot gradients
        # as previously
        monkeypatch.setattr(opt, "_single_shot_qnode_gradients", lambda *args, **kwargs: single_shot_grads)
        monkeypatch.setattr(opt, "_single_shot_expval_gradients", lambda *args, **kwargs: single_shot_grads)

        # reset the shot budget
        opt.s = [10 * np.ones([2, 3], dtype=np.int64)]

        # check that the gradient and variance are computed correctly
        grad, grad_variance = opt.compute_grad(cost_fn, [x_init], {})
        assert len(grad) == 1
        assert len(grad_variance) == 1
        assert grad[0].shape == x_init.shape
        assert grad_variance[0].shape == x_init.shape

        assert np.allclose(grad, np.mean(single_shot_grads, axis=1))
        assert np.allclose(grad_variance, np.var(single_shot_grads, ddof=1, axis=1))

        # check that the gradient and variance are computed correctly
        # with a different shot budget
        opt.s[0] = opt.s[0] // 2  # all array elements have a shot budget of 5
        opt.s[0][0, 0] = 8    # set the shot budget of the zeroth element to 8

        grad, grad_variance = opt.compute_grad(cost_fn, [x_init], {})
        assert len(grad) == 1
        assert len(grad_variance) == 1

        # check zeroth element
        assert np.allclose(grad[0][0, 0], np.mean(single_shot_grads[0][:8, 0, 0]))
        assert np.allclose(grad_variance[0][0, 0], np.var(single_shot_grads[0][:8, 0, 0], ddof=1))

        # check other elements
        assert np.allclose(grad[0][0, 1], np.mean(single_shot_grads[0][:5, 0, 1]))
        assert np.allclose(grad_variance[0][0, 1], np.var(single_shot_grads[0][:5, 0, 1], ddof=1))

    def test_multiple_argument_step(self, mocker, monkeypatch):
        """Test that a simple QNode with multiple scalar arguments correctly performs an optimization step,
        and that the single-shot gradients generated have the correct shape"""
        dev = qml.device("default.qubit", wires=1, analytic=False)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        opt = qml.ShotAdaptiveOptimizer(min_shots=10)
        spy_single_shot = mocker.spy(opt, "_single_shot_qnode_gradients")
        spy_grad = mocker.spy(opt, "compute_grad")

        args = [0.1, 0.2]
        new_x = opt.step(circuit, *args)

        assert isinstance(new_x, list)
        assert len(new_x) == 2

        spy_single_shot.assert_called_once()
        spy_grad.assert_called_once()

        # assert single shot gradients are computed correctly
        single_shot_grads = opt._single_shot_qnode_gradients(circuit, args, {})
        assert len(single_shot_grads) == 2
        assert single_shot_grads[0].shape == (10,)

        # monkeypatch the optimizer to use the same single shot gradients
        # as previously
        monkeypatch.setattr(opt, "_single_shot_qnode_gradients", lambda *args, **kwargs: single_shot_grads)

        # reset the shot budget
        opt.s = [np.array(10), np.array(10)]

        # check that the gradient and variance are computed correctly
        grad, grad_variance = opt.compute_grad(circuit, args, {})
        assert len(grad) == 2
        assert len(grad_variance) == 2
        assert np.allclose(grad, np.mean(single_shot_grads, axis=1))
        assert np.allclose(grad_variance, np.var(single_shot_grads, ddof=1, axis=1))

        # check that the gradient and variance are computed correctly
        # with a different shot budget
        opt.s = [np.array(5), np.array(7)]
        grad, grad_variance = opt.compute_grad(circuit, args, {})
        assert len(grad) == 2
        assert len(grad_variance) == 2

        for p, s in zip(range(2), opt.s):
            assert np.allclose(grad[p], np.mean(single_shot_grads[p][:s]))
            assert np.allclose(grad_variance[p], np.var(single_shot_grads[p][:s], ddof=1))

    def test_multiple_array_argument_step(self, mocker, monkeypatch):
        """Test that a simple QNode with multiple array arguments correctly performs an optimization step,
        and that the single-shot gradients generated have the correct shape"""
        dev = qml.device("default.qubit", wires=1, analytic=False)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x[0, 0], wires=0)
            qml.RY(x[0, 1], wires=0)
            qml.RZ(x[0, 2], wires=0)
            qml.RX(x[1, 0], wires=0)
            qml.RY(x[1, 1], wires=0)
            qml.RZ(x[1, 2], wires=0)
            qml.RX(y[0], wires=0)
            qml.RY(y[1], wires=0)
            qml.RZ(y[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        opt = qml.ShotAdaptiveOptimizer(min_shots=10)
        spy_single_shot = mocker.spy(opt, "_single_shot_qnode_gradients")
        spy_grad = mocker.spy(opt, "compute_grad")

        args = [np.array([[1., 2., 3.], [4., 5., 6.]]), np.array([1., 2., 3.])]
        new_x = opt.step(circuit, *args)

        assert isinstance(new_x, list)
        assert len(new_x) == 2

        spy_single_shot.assert_called_once()
        spy_grad.assert_called_once()

        # assert single shot gradients are computed correctly
        single_shot_grads = opt._single_shot_qnode_gradients(circuit, args, {})
        assert len(single_shot_grads) == 2
        assert single_shot_grads[0].shape == (10, 2, 3)
        assert single_shot_grads[1].shape == (10, 3)

        # monkeypatch the optimizer to use the same single shot gradients
        # as previously
        monkeypatch.setattr(opt, "_single_shot_qnode_gradients", lambda *args, **kwargs: single_shot_grads)

        # reset the shot budget
        opt.s = [10 * np.ones([2, 3], dtype=np.int64), 10 * np.ones([3], dtype=np.int64)]

        # check that the gradient and variance are computed correctly
        grad, grad_variance = opt.compute_grad(circuit, args, {})
        assert len(grad) == 2
        assert len(grad_variance) == 2

        for i in range(2):
            assert grad[i].shape == args[i].shape
            assert grad_variance[i].shape == args[i].shape

        assert np.allclose(grad[0], np.mean(single_shot_grads[0], axis=0))
        assert np.allclose(grad_variance[0], np.var(single_shot_grads[0], ddof=1, axis=0))

        assert np.allclose(grad[1], np.mean(single_shot_grads[1], axis=0))
        assert np.allclose(grad_variance[1], np.var(single_shot_grads[1], ddof=1, axis=0))

        # check that the gradient and variance are computed correctly
        # with a different shot budget
        opt.s[0] = opt.s[0] // 2  # all array elements have a shot budget of 5
        opt.s[0][0, 0] = 8    # set the shot budget of the zeroth element to 8

        opt.s[1] = opt.s[1] // 5  # all array elements have a shot budget of 2
        opt.s[1][0] = 7    # set the shot budget of the zeroth element to 7

        grad, grad_variance = opt.compute_grad(circuit, args, {})
        assert len(grad) == 2
        assert len(grad_variance) == 2

        # check zeroth element of arg 0
        assert np.allclose(grad[0][0, 0], np.mean(single_shot_grads[0][:8, 0, 0]))
        assert np.allclose(grad_variance[0][0, 0], np.var(single_shot_grads[0][:8, 0, 0], ddof=1))

        # check other elements of arg 0
        assert np.allclose(grad[0][0, 1], np.mean(single_shot_grads[0][:5, 0, 1]))
        assert np.allclose(grad_variance[0][0, 1], np.var(single_shot_grads[0][:5, 0, 1], ddof=1))

        # check zeroth element of arg 1
        assert np.allclose(grad[1][0], np.mean(single_shot_grads[1][:7, 0]))
        assert np.allclose(grad_variance[1][0], np.var(single_shot_grads[1][:7, 0], ddof=1))

        # check other elements of arg 1
        assert np.allclose(grad[1][1], np.mean(single_shot_grads[1][:2, 1]))
        assert np.allclose(grad_variance[1][1], np.var(single_shot_grads[1][:2, 1], ddof=1))


class TestWeightedRandomSampling:
    """Tests for weighted random Hamiltonian term sampling"""

    def test_wrs_expval_cost(self, mocker):
        """Checks that cost functions that are expval costs can
        make use of, and turn off, weighted random sampling"""
        coeffs = [0.2, 0.1]
        dev = qml.device("default.qubit", wires=2, analytic=False)
        H = qml.Hamiltonian(coeffs, [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)])

        expval_cost = qml.ExpvalCost(qml.templates.StronglyEntanglingLayers, H, dev)
        weights = qml.init.strong_ent_layers_normal(n_layers=3, n_wires=2)

        opt = qml.ShotAdaptiveOptimizer(min_shots=100)
        spy = mocker.spy(opt, "weighted_random_sampling")

        new_weights = opt.step(expval_cost, weights)
        spy.assert_called_once()

        grads = opt.weighted_random_sampling(expval_cost.qnodes, coeffs, 10, 0, weights)
        assert len(grads) == 1
        assert grad[0].shape == (10,)
        assert dev.num_executions == len(coeffs)
