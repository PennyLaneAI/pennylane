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
Unit tests for the ``QNSPSAOptimizer``
"""
# pylint: disable=protected-access
from copy import deepcopy

import pytest
from scipy.linalg import sqrtm

import pennylane as qml
from pennylane import numpy as np


def get_single_input_qnode():
    """Prepare qnode with a single tensor as input."""
    dev = qml.device("default.qubit", wires=2)

    # the analytical expression of the qnode goes as:
    # np.cos(params[0][0] / 2) ** 2 - np.sin(params[0][0] / 2) ** 2 * np.cos(params[0][1])
    @qml.qnode(dev)
    def loss_fn(params):
        qml.RY(params[0][0], wires=0)
        qml.CRX(params[0][1], wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return loss_fn, (1, 2)  # returns the qnode and the input param shape


def get_multi_input_qnode():
    """Prepare qnode with two separate tensors as input."""
    dev = qml.device("default.qubit", wires=2)

    # the analytical expression of the qnode goes as:
    # np.cos(x1 / 2) ** 2 - np.sin(x1 / 2) ** 2 * np.cos(x2)
    @qml.qnode(dev)
    def loss_fn(x1, x2):
        qml.RY(x1, wires=0)
        qml.CRX(x2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return loss_fn


def get_grad_finite_diff(params, finite_diff_step, grad_dirs):
    """Helper function computing the qnode finite difference for computing the gradient analytically.
    One can expand the following expression to get the qnode_finite_diff expression:
    qnode(params + finite_diff_step * grad_dirs) - qnode(params - finite_diff_step * grad_dirs)
    """
    qnode_finite_diff = (
        -np.sin(params[0]) * np.sin(finite_diff_step * grad_dirs[0])
        + np.sin(params[1]) * np.sin(finite_diff_step * grad_dirs[1])
        + (
            np.cos(
                params[0]
                + params[1]
                + finite_diff_step * grad_dirs[0]
                + finite_diff_step * grad_dirs[1]
            )
            + np.cos(
                params[0]
                + finite_diff_step * grad_dirs[0]
                - params[1]
                - finite_diff_step * grad_dirs[1]
            )
            - np.cos(
                params[0]
                + params[1]
                - finite_diff_step * grad_dirs[0]
                - finite_diff_step * grad_dirs[1]
            )
            - np.cos(
                params[0]
                - finite_diff_step * grad_dirs[0]
                - params[1]
                + finite_diff_step * grad_dirs[1]
            )
        )
        / 4
    )
    return qnode_finite_diff


def get_metric_from_single_input_qnode(params, finite_diff_step, tensor_dirs):
    """Compute the expected raw metric tensor from analytical state overlap expression."""
    dir_vec1 = tensor_dirs[0]
    dir_vec2 = tensor_dirs[1]
    dir1 = dir_vec1.reshape(params.shape)
    dir2 = dir_vec2.reshape(params.shape)
    perturb1 = dir1 * finite_diff_step
    perturb2 = dir2 * finite_diff_step

    def get_state_overlap(params1, params2):
        # analytically computed state overlap between two parametrized ansatzes
        # with input params1 and params2
        return (
            np.cos(params1[0][0] / 2) * np.cos(params2[0][0] / 2)
            + np.sin(params1[0][0] / 2)
            * np.sin(params2[0][0] / 2)
            * (
                np.sin(params1[0][1] / 2) * np.sin(params2[0][1] / 2)
                + np.cos(params1[0][1] / 2) * np.cos(params2[0][1] / 2)
            )
        ) ** 2

    tensor_finite_diff = (
        get_state_overlap(params, params + perturb1 + perturb2)
        - get_state_overlap(params, params + perturb1)
        - get_state_overlap(params, params - perturb1 + perturb2)
        + get_state_overlap(params, params - perturb1)
    )
    metric_tensor_expected = (
        -(np.tensordot(dir_vec1, dir_vec2, axes=0) + np.tensordot(dir_vec2, dir_vec1, axes=0))
        * tensor_finite_diff
        / (8 * finite_diff_step * finite_diff_step)
    )
    return metric_tensor_expected


@pytest.mark.parametrize("finite_diff_step", [1e-3, 1e-2, 1e-1])
class TestQNSPSAOptimizer:
    def test_gradient_from_single_input(self, finite_diff_step, seed):
        """Test that the QNSPSA gradient estimation is correct by comparing the optimizer result
        to the analytical result."""
        opt = qml.QNSPSAOptimizer(
            stepsize=1e-3,
            regularization=1e-3,
            finite_diff_step=finite_diff_step,
            resamplings=1,
            blocking=True,
            history_length=5,
            seed=seed,
        )
        qnode, params_shape = get_single_input_qnode()
        params = np.random.rand(*params_shape)

        # gradient result from QNSPSAOptimizer
        grad_tapes, grad_dirs = opt._get_spsa_grad_tapes(qnode, [params], {})
        raw_results = qml.execute(grad_tapes, qnode.device, None)
        grad_res = opt._post_process_grad(raw_results, grad_dirs)[0]

        # gradient computed analytically
        qnode_finite_diff = get_grad_finite_diff(params[0], finite_diff_step, grad_dirs[0][0])
        grad_expected = qnode_finite_diff / (2 * finite_diff_step) * grad_dirs[0]
        assert np.allclose(grad_res, grad_expected)

    def test_raw_metric_tensor(self, finite_diff_step, seed):
        """Test that the QNSPSA metric tensor estimation(before regularization) is correct by
        comparing the optimizer result to the analytical result."""
        opt = qml.QNSPSAOptimizer(
            stepsize=1e-3,
            regularization=1e-3,
            finite_diff_step=finite_diff_step,
            resamplings=1,
            blocking=True,
            history_length=5,
            seed=seed,
        )

        qnode, params_shape = get_single_input_qnode()
        params = np.random.rand(*params_shape)

        # raw metric tensor result from QNSPSAOptimizer
        metric_tapes, tensor_dirs = opt._get_tensor_tapes(qnode, [params], {})
        raw_results = qml.execute(metric_tapes, qnode.device, None)
        metric_tensor_res = opt._post_process_tensor(raw_results, tensor_dirs)

        # expected raw metric tensor from analytical state overlap
        metric_tensor_expected = get_metric_from_single_input_qnode(
            params, finite_diff_step, tensor_dirs
        )
        assert np.allclose(metric_tensor_res, metric_tensor_expected)

    def test_gradient_from_multi_input(self, finite_diff_step, seed):
        """Test that the QNSPSA gradient estimation is correct by comparing the optimizer result
        to the analytical result."""
        opt = qml.QNSPSAOptimizer(
            stepsize=1e-3,
            regularization=1e-3,
            finite_diff_step=finite_diff_step,
            resamplings=1,
            blocking=True,
            history_length=5,
            seed=seed,
        )
        qnode = get_multi_input_qnode()
        params = [np.random.rand(1) for _ in range(2)]
        # gradient result from QNSPSAOptimizer
        grad_tapes, grad_dirs = opt._get_spsa_grad_tapes(qnode, params, {})
        raw_results = qml.execute(grad_tapes, qnode.device, None)
        grad_res = opt._post_process_grad(raw_results, grad_dirs)

        # gradient computed analytically
        qnode_finite_diff = get_grad_finite_diff(params, finite_diff_step, grad_dirs)
        grad_expected = [
            qnode_finite_diff / (2 * finite_diff_step) * grad_dir for grad_dir in grad_dirs
        ]
        assert np.allclose(grad_res, grad_expected)

    def test_step_from_single_input(self, finite_diff_step, seed):
        """Test step() function with the single-input qnode."""
        regularization = 1e-3
        stepsize = 1e-2
        opt = qml.QNSPSAOptimizer(
            stepsize=stepsize,
            regularization=regularization,
            finite_diff_step=finite_diff_step,
            resamplings=1,
            blocking=False,
            history_length=5,
            seed=seed,
        )
        # target opt is used to reproduce the random sampling result
        target_opt = deepcopy(opt)

        qnode, params_shape = get_single_input_qnode()
        params = np.random.rand(*params_shape)

        new_params_res = opt.step(qnode, params)

        _, grad_dirs = target_opt._get_spsa_grad_tapes(qnode, [params], {})
        _, tensor_dirs = target_opt._get_tensor_tapes(qnode, [params], {})

        qnode_finite_diff = get_grad_finite_diff(params[0], finite_diff_step, grad_dirs[0][0])
        grad_expected = (qnode_finite_diff / (2 * finite_diff_step) * grad_dirs[0])[0]

        metric_tensor_expected = get_metric_from_single_input_qnode(
            params, finite_diff_step, tensor_dirs
        )
        # regularize raw metric tensor
        identity = np.identity(metric_tensor_expected.shape[0])
        avg_metric_tensor = 0.5 * (identity + metric_tensor_expected)
        tensor_reg = np.real(sqrtm(np.matmul(avg_metric_tensor, avg_metric_tensor)))
        tensor_reg = (tensor_reg + regularization * identity) / (1 + regularization)

        inv_metric_tensor = np.linalg.inv(tensor_reg)
        new_params_expected = params - stepsize * np.matmul(inv_metric_tensor, grad_expected)
        assert np.allclose(new_params_res, new_params_expected)

    def test_step_and_cost_from_single_input(self, finite_diff_step, seed):
        """Test step_and_cost() function with the single-input qnode. Both blocking settings
        (on/off) are tested.
        """
        regularization = 1e-3
        stepsize = 1e-2
        opt_blocking = qml.QNSPSAOptimizer(
            stepsize=stepsize,
            regularization=regularization,
            finite_diff_step=finite_diff_step,
            resamplings=1,
            blocking=True,
            history_length=5,
            seed=seed,
        )
        opt_no_blocking = deepcopy(opt_blocking)
        opt_no_blocking.blocking = False
        # target opt is used to reproduce the result with step()
        target_opt = deepcopy(opt_blocking)

        qnode, params_shape = get_single_input_qnode()
        params = np.random.rand(*params_shape)

        new_params_blocking_res, qnode_blocking_res = opt_blocking.step_and_cost(qnode, params)
        with pytest.warns(UserWarning):
            new_params_expected = target_opt.step(qnode, params)
        # analytical expression of the qnode
        qnode_expected = np.cos(params[0][0] / 2) ** 2 - np.sin(params[0][0] / 2) ** 2 * np.cos(
            params[0][1]
        )
        assert np.allclose(new_params_blocking_res, new_params_expected)
        assert np.allclose(qnode_blocking_res, qnode_expected)

        new_params_no_blocking_res, qnode_no_blocking_res = opt_no_blocking.step_and_cost(
            qnode, params
        )
        assert np.allclose(new_params_no_blocking_res, new_params_expected)
        assert np.allclose(qnode_no_blocking_res, qnode_expected)

    def test_step_and_cost_from_multi_input(self, finite_diff_step, seed):
        """Test step_and_cost() function with the multi-input qnode."""
        regularization = 1e-3
        stepsize = 1e-2
        opt = qml.QNSPSAOptimizer(
            stepsize=stepsize,
            regularization=regularization,
            finite_diff_step=finite_diff_step,
            resamplings=1,
            blocking=True,
            history_length=5,
            seed=seed,
        )
        # target opt is used to reproduce the random sampling result
        target_opt = deepcopy(opt)

        qnode = get_multi_input_qnode()
        params = [np.array(1.0) for _ in range(2)]
        # this single-step result will be different from the one from the single-input qnode, due to the
        # different order in sampling perturbation directions.
        new_params_res, qnode_res = opt.step_and_cost(qnode, *params)

        # test the expectation value
        qnode_expected = np.cos(params[0] / 2) ** 2 - np.sin(params[0] / 2) ** 2 * np.cos(params[1])
        assert np.allclose(qnode_res, qnode_expected)

        # test the next-step parameter
        _, grad_dirs = target_opt._get_spsa_grad_tapes(qnode, params, {})
        _, tensor_dirs = target_opt._get_tensor_tapes(qnode, params, {})
        qnode_finite_diff = get_grad_finite_diff(params, finite_diff_step, grad_dirs)
        grad_expected = [
            qnode_finite_diff / (2 * finite_diff_step) * grad_dir for grad_dir in grad_dirs
        ]
        # reshape the params list into a tensor to reuse the
        # get_metric_from_single_input_qnode helper function
        params_tensor = np.array(params).reshape(1, len(params))
        metric_tensor_expected = get_metric_from_single_input_qnode(
            params_tensor, finite_diff_step, tensor_dirs
        )

        # regularize raw metric tensor
        identity = np.identity(metric_tensor_expected.shape[0])
        avg_metric_tensor = 0.5 * (identity + metric_tensor_expected)
        tensor_reg = np.real(sqrtm(np.matmul(avg_metric_tensor, avg_metric_tensor)))
        tensor_reg = (tensor_reg + regularization * identity) / (1 + regularization)

        inv_metric_tensor = np.linalg.inv(tensor_reg)
        grad_tensor = np.array(grad_expected).reshape(
            inv_metric_tensor.shape[0],
        )
        new_params_tensor_expected = params_tensor - stepsize * np.matmul(
            inv_metric_tensor, grad_tensor
        )

        assert np.allclose(
            np.array(new_params_res).reshape(new_params_tensor_expected.shape),
            new_params_tensor_expected,
        )

    def test_step_and_cost_with_non_trainable_input(self, finite_diff_step, seed):
        """
        Test step_and_cost() function with the qnode with non-trainable input,
        both using the `default.qubit` device.
        """
        regularization = 1e-3
        stepsize = 1e-2
        opt = qml.QNSPSAOptimizer(
            stepsize=stepsize,
            regularization=regularization,
            finite_diff_step=finite_diff_step,
            resamplings=1,
            blocking=True,
            history_length=5,
            seed=seed,
        )
        # a deep copy of the same opt, to be applied to qnode_reduced
        target_opt = deepcopy(opt)
        dev = qml.device("default.qubit", wires=2)
        non_trainable_param = np.random.rand(1)
        non_trainable_param.requires_grad = False

        trainable_param = np.random.rand(1)

        @qml.qnode(dev)
        def qnode_with_non_trainable(trainable, non_trainable):
            qml.RY(trainable, wires=0)
            qml.CRX(non_trainable, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        # reduced qnode where non-trainable param value is hard coded
        @qml.qnode(dev)
        def qnode_reduced(trainable):
            qml.RY(trainable, wires=0)
            qml.CRX(non_trainable_param, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        new_params_res, qnode_res = opt.step_and_cost(
            qnode_with_non_trainable, trainable_param, non_trainable_param
        )
        new_trainable_res, new_non_trianable_res = new_params_res

        new_trainable_expected, qnode_expected = target_opt.step_and_cost(
            qnode_reduced, trainable_param
        )

        assert np.allclose(qnode_res, qnode_expected)
        assert np.allclose(new_non_trianable_res, non_trainable_param)
        assert np.allclose(new_trainable_res, new_trainable_expected)

    def test_blocking(self, finite_diff_step, seed):
        """Test blocking setting of the optimizer."""
        regularization = 1e-3
        stepsize = 1.0
        history_length = 5
        opt = qml.QNSPSAOptimizer(
            stepsize=stepsize,
            regularization=regularization,
            finite_diff_step=finite_diff_step,
            resamplings=1,
            blocking=True,
            history_length=history_length,
            seed=seed,
        )
        qnode, params_shape = get_single_input_qnode()
        # params minimizes the qnode
        params = np.tensor([3.1415, 0]).reshape(params_shape)

        # fill opt.last_n_steps array with a minimum expectation value
        for _ in range(history_length):
            opt.step_and_cost(qnode, params)
        # blocking should stop params from updating from this minimum
        new_params, _ = opt.step_and_cost(qnode, params)
        assert np.allclose(new_params, params)


def test_template_no_adjoint(seed):
    """Test that qnspsa iterates when the operations do not have a custom adjoint."""

    num_qubits = 2
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def cost(params):
        qml.RandomLayers(weights=params, wires=range(num_qubits), seed=seed)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    params = np.random.normal(0, np.pi, (2, 4))
    opt = qml.QNSPSAOptimizer(stepsize=5e-2)
    assert opt.step_and_cost(cost, params)  # just checking it runs without error
    assert not qml.RandomLayers.has_adjoint


def test_workflow_integration():
    """Test that the optimizer can optimize a workflow."""

    num_qubits = 2
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def cost(params):
        qml.RX(params[0], wires=0)
        qml.CRY(params[1], wires=[0, 1])
        return qml.expval(qml.Z(0) @ qml.Z(1))

    params = qml.numpy.array([0.5, 0.5], requires_grad=True)
    opt = qml.optimize.QNSPSAOptimizer(stepsize=5e-2)
    for _ in range(101):
        params, loss = opt.step_and_cost(cost, params)

    assert qml.math.allclose(loss, -1, atol=1e-3)
    # compare sine of params and target params as could converge to params + 2* np.pi
    target_params = np.array([np.pi, 0])
    assert qml.math.allclose(np.sin(params), np.sin(target_params), atol=1e-2)
