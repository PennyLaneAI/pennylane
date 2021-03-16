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


class TestQNodeIntegration:
    """Integration tests to ensure that the optimizer works correctly with QNodes"""

    def test_single_argument_step(self, mocker, monkeypatch):
        """Test that a simple QNode with a single argument correctly performs an optimization step,
        and that the single-shot gradients generated have the correct shape"""
        dev = qml.device("default.qubit", wires=1, analytic=False)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        opt = qml.ShotAdaptiveOptimizer(min_shots=10)
        spy_single_shot = mocker.spy(opt, "_single_shot_qnode_gradients")
        spy_grad = mocker.spy(opt, "compute_grad")

        x_init = 0.5
        new_x = opt.step(circuit, x_init)

        assert isinstance(new_x, float)
        assert new_x != x_init

        spy_single_shot.assert_called_once()
        spy_grad.assert_called_once()

        # assert single shot gradients are computed correctly
        single_shot_grads = opt._single_shot_qnode_gradients(circuit, [x_init], {})
        assert len(single_shot_grads) == 1
        assert single_shot_grads[0].shape == (10,)

        # monkeypatch the optimizer to use the same single shot gradients
        # as previously
        monkeypatch.setattr(opt, "_single_shot_qnode_gradients", lambda *args, **kwargs: single_shot_grads)

        # reset the shot budget
        opt.s = [np.array(10)]

        # check that the gradient and variance are computed correctly
        grad, grad_variance = opt.compute_grad(circuit, [x_init], {})
        assert len(grad) == 1
        assert len(grad_variance) == 1
        assert np.allclose(grad, np.mean(single_shot_grads))
        assert np.allclose(grad_variance, np.var(single_shot_grads, ddof=1))

        # check that the gradient and variance are computed correctly
        # with a different shot budget
        opt.s = [np.array(5)]
        grad, grad_variance = opt.compute_grad(circuit, [x_init], {})
        assert len(grad) == 1
        assert len(grad_variance) == 1
        assert np.allclose(grad, np.mean(single_shot_grads[0][:5]))
        assert np.allclose(grad_variance, np.var(single_shot_grads[0][:5], ddof=1))
