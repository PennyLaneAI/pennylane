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

    @qml.qnode(dev)
    def loss_fn(params):
        qml.RX(params[0][0], wires=0)
        qml.CRY(params[0][1], wires=[0, 1])
        qml.RY(params[1][0], wires=0)
        qml.CRX(params[1][1], wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return loss_fn, (2, 2)  # returns the qnode and the input param shape


def get_multi_input_qnode():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def loss_fn(x1, x2):
        qml.RX(x1, wires=0)
        qml.CRY(x2, wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return loss_fn


class TestSPSAOptimizer:
    @pytest.mark.parametrize("seed", [1, 151, 1231])
    @pytest.mark.parametrize("finite_diff_step", [1e-3, 1e-2, 1e-1])
    def test_qnspsa_grad(self, finite_diff_step, seed):
        """Test that the QNSPSA gradient estimation is correct by comparing the optimizer result
        to a straightforward implementation."""
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

        # gradient expected from a straightforward implementation
        rng = np.random.default_rng(seed)
        direction = rng.choice([-1, 1], size=params.shape)
        loss_plus = qnode(params + direction * finite_diff_step)
        loss_minus = qnode(params - direction * finite_diff_step)
        grad_expected = (loss_plus - loss_minus) / (2 * finite_diff_step) * direction
        assert np.allclose(grad_res, grad_expected)

    @pytest.mark.parametrize("seed", [1, 151, 1231])
    @pytest.mark.parametrize("finite_diff_step", [1e-3, 1e-2, 1e-1])
    def test_raw_metric_tensor(self, finite_diff_step, seed):
        """Test that the QNSPSA metric tensor estimation(before regularization) is correct by
        comparing the optimizer result to a straightforward implementation."""
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

        # expected raw metric tensor from a straightforward implementation
        rng = np.random.default_rng(seed)
        dir1 = rng.choice([-1, 1], size=params.shape)
        dir2 = rng.choice([-1, 1], size=params.shape)
        dir_vec1 = dir1.reshape(-1)
        dir_vec2 = dir2.reshape(-1)
        perturb1 = dir1 * finite_diff_step
        perturb2 = dir2 * finite_diff_step

        def get_operations(qnode, params):
            qnode.construct([params], {})
            return qnode.tape.operations

        def get_overlap_tape(qnode, params1, params2):
            op_forward = get_operations(qnode, params1)
            op_inv = get_operations(qnode, params2)

            with qml.tape.QuantumTape() as tape:
                for op in op_forward:
                    qml.apply(op)
                for op in reversed(op_inv):
                    op.adjoint()
                qml.probs(wires=qnode.tape.wires.labels)
            return tape

        def get_state_overlap(tape, dev):
            return qml.execute([tape], dev, None)[0][0][0]

        tapes = [
            get_overlap_tape(qnode, params, params + perturb1 + perturb2),
            get_overlap_tape(qnode, params, params + perturb1),
            get_overlap_tape(qnode, params, params - perturb1 + perturb2),
            get_overlap_tape(qnode, params, params - perturb1),
        ]

        tensor_finite_diff = (
            get_state_overlap(tapes[0], qnode.device)
            - get_state_overlap(tapes[1], qnode.device)
            - get_state_overlap(tapes[2], qnode.device)
            + get_state_overlap(tapes[3], qnode.device)
        )
        metric_tensor_expected = (
            -(np.tensordot(dir_vec1, dir_vec2, axes=0) + np.tensordot(dir_vec2, dir_vec1, axes=0))
            * tensor_finite_diff
            / (8 * finite_diff_step * finite_diff_step)
        )

        assert np.allclose(metric_tensor_res, metric_tensor_expected)

    def test_step_with_multi_input(self):
        """Test that the optimizer works with multiple inputs with a hard coded loss-vs-step result."""
        qnode = get_multi_input_qnode()

        params = [np.tensor(1.0) for _ in range(2)]

        opt = qml.QNSPSAOptimizer(
            stepsize=1e-1,
            regularization=1e-3,
            finite_diff_step=1e-2,
            resamplings=1,
            blocking=True,
            history_length=5,
            seed=1,
        )
        loss_res = []
        for i in range(10):
            params = opt.step(qnode, *params)
            loss_res.append(qnode(*params))

        loss_res = np.array(loss_res)

        # hard coded execution result
        loss_expected = np.array(
            [
                0.5531389,
                0.4983248,
                0.3243610,
                0.2741123,
                -0.0843600,
                -0.1138668,
                -0.5087606,
                -0.5160034,
                -0.5144091,
                -0.8287674,
            ]
        )

        assert np.allclose(
            loss_res,
            loss_expected,
        ), "Loss curve over steps does not match hard-coded result"

    def test_step_and_cost_with_single_input(self):
        """Test that the optimizer works with single input with a hard coded loss-vs-step result."""
        qnode, shape = get_single_input_qnode()

        params = np.ones(shape)

        opt = qml.QNSPSAOptimizer(
            stepsize=2e-1,
            regularization=1e-3,
            finite_diff_step=1e-2,
            resamplings=1,
            blocking=True,
            history_length=5,
            seed=1,
        )

        loss_res = []
        for i in range(10):
            params, loss = opt.step_and_cost(qnode, params)
            loss_res.append(loss)

        loss_res = np.array(loss_res)

        loss_expected = np.array(
            [
                0.61718389,
                0.51105689,
                0.51105689,
                0.51379872,
                0.51379872,
                0.34328399,
                0.34328399,
                0.37729847,
                0.33405945,
                0.33405945,
            ]
        )

        assert np.allclose(
            loss_res,
            loss_expected,
        ), "Loss curve over steps does not match hard-coded result"
