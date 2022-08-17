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
import pytest

import pennylane as qml
from pennylane import numpy as np


def get_single_input_qnode():
    dev = qml.device("default.qubit", wires=2)
    # the analytical expression of the qnode goes as:
    # np.cos(params[0][0] / 2) ** 2 - np.sin(params[0][0] / 2) ** 2 * np.cos(params[0][1])
    @qml.qnode(dev)
    def loss_fn(params):
        qml.RY(params[0][0], wires=0)
        qml.CRX(params[0][1], wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return loss_fn, (2, 2)  # returns the qnode and the input param shape


def get_multi_input_qnode():
    dev = qml.device("default.qubit", wires=2)
    # the analytical expression of the qnode goes as:
    # np.cos(x1 / 2) ** 2 - np.sin(x1 / 2) ** 2 * np.cos(x2)
    @qml.qnode(dev)
    def loss_fn(x1, x2):
        qml.RY(x1, wires=0)
        qml.CRX(x2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return loss_fn


class TestSPSAOptimizer:
    @pytest.mark.parametrize("seed", [1, 151, 1231])
    @pytest.mark.parametrize("finite_diff_step", [1e-3, 1e-2, 1e-1])
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

        # gradient computed analytically. One can expand the following expression
        # to get the qnode_finite_diff expression:
        # qnode(params + finite_diff_step * direction) - qnode(params - finite_diff_step * direction)

        direction = grad_dirs[0]
        qnode_finite_diff = (
            -np.sin(params[0][0]) * np.sin(finite_diff_step * direction[0][0])
            + np.sin(params[0][1]) * np.sin(finite_diff_step * direction[0][1])
            + (
                np.cos(
                    params[0][0]
                    + params[0][1]
                    + finite_diff_step * direction[0][0]
                    + finite_diff_step * direction[0][1]
                )
                + np.cos(
                    params[0][0]
                    + finite_diff_step * direction[0][0]
                    - params[0][1]
                    - finite_diff_step * direction[0][1]
                )
                - np.cos(
                    params[0][0]
                    + params[0][1]
                    - finite_diff_step * direction[0][0]
                    - finite_diff_step * direction[0][1]
                )
                - np.cos(
                    params[0][0]
                    - finite_diff_step * direction[0][0]
                    - params[0][1]
                    + finite_diff_step * direction[0][1]
                )
            )
            / 4
        )
        grad_expected = qnode_finite_diff / (2 * finite_diff_step) * direction
        assert np.allclose(grad_res, grad_expected)

    @pytest.mark.parametrize("seed", [1, 151, 1231])
    @pytest.mark.parametrize("finite_diff_step", [1e-3, 1e-2, 1e-1])
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
        rng = np.random.default_rng(seed)
        dir1 = rng.choice([-1, 1], size=params.shape)
        dir2 = rng.choice([-1, 1], size=params.shape)
        dir_vec1 = dir1.reshape(-1)
        dir_vec2 = dir2.reshape(-1)
        perturb1 = dir1 * finite_diff_step
        perturb2 = dir2 * finite_diff_step

        def get_state_overlap(params1, params2):
            # analytically computed state overlap between two parameterized ansatzes
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

        assert np.allclose(metric_tensor_res, metric_tensor_expected)

    @pytest.mark.parametrize("seed", [1, 151, 1231])
    @pytest.mark.parametrize("finite_diff_step", [1e-3, 1e-2, 1e-1])
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

        # gradient computed analytically. One can expand the following expression
        # to get the qnode_finite_diff expression:
        # qnode(params + finite_diff_step * direction) - qnode(params - finite_diff_step * direction)
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
        grad_expected = [
            qnode_finite_diff / (2 * finite_diff_step) * grad_dir for grad_dir in grad_dirs
        ]
        assert np.allclose(grad_res, grad_expected)
