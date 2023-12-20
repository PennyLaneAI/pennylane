# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for create_initial_state in devices/apply_operation."""

import pytest
import pennylane as qml
import numpy as np
from pennylane import math
from pennylane.devices.qutrit_mixed.apply_operation import (
    apply_operation_einsum,
    apply_operation_tensordot,
    apply_operation,
)

density_matrix = np.array(
    [
        [
            [
                [0.09208281 + 0.0j, 0.03160535 + 0.0664417j, 0.03416066 + 0.0206764j],
                [-0.01227047 + 0.06373319j, 0.01815704 - 0.03131422j, -0.03700634 + 0.01956628j],
                [0.03128229 + 0.02937806j, 0.03555759 - 0.01100168j, -0.01982137 - 0.04605857j],
            ],
            [
                [0.03160535 - 0.0664417j, 0.10864802 + 0.0j, 0.01877872 + 0.01624366j],
                [0.10580479 + 0.04077997j, -0.06409966 + 0.01593581j, 0.00204818 + 0.03765312j],
                [-0.00255557 - 0.06777649j, -0.04497981 - 0.02030987j, -0.05656505 + 0.06693942j],
            ],
            [
                [0.03416066 - 0.0206764j, 0.01877872 - 0.01624366j, 0.08975985 + 0.0j],
                [0.03077889 - 0.02115111j, 0.04876844 + 0.04583181j, -0.04781717 + 0.04220666j],
                [-0.03402139 + 0.04582056j, 0.03198441 + 0.06076387j, 0.02845793 - 0.00996117j],
            ],
        ],
        [
            [
                [-0.01227047 - 0.06373319j, 0.10580479 - 0.04077997j, 0.03077889 + 0.02115111j],
                [0.14327529 + 0.0j, -0.07288875 + 0.07020128j, -0.000707 + 0.04396371j],
                [-0.05027514 - 0.08649698j, -0.07202654 + 0.01724066j, -0.03980968 + 0.11268854j],
            ],
            [
                [0.01815704 + 0.03131422j, -0.06409966 - 0.01593581j, 0.04876844 - 0.04583181j],
                [-0.07288875 - 0.07020128j, 0.12370217 + 0.0j, -0.00788202 + 0.02235794j],
                [-0.0128203 + 0.11522974j, 0.09896394 + 0.04999461j, 0.08419826 - 0.06733029j],
            ],
            [
                [-0.03700634 - 0.01956628j, 0.00204818 - 0.03765312j, -0.04781717 - 0.04220666j],
                [-0.000707 - 0.04396371j, -0.00788202 - 0.02235794j, 0.08931812 + 0.0j],
                [0.00766162 - 0.01285426j, -0.0084444 - 0.042322j, 0.00769262 + 0.03245046j],
            ],
        ],
        [
            [
                [0.03128229 - 0.02937806j, -0.00255557 + 0.06777649j, -0.03402139 - 0.04582056j],
                [-0.05027514 + 0.08649698j, -0.0128203 - 0.11522974j, 0.00766162 + 0.01285426j],
                [0.11637437 + 0.0j, 0.03960783 - 0.09361331j, -0.08419771 - 0.07692928j],
            ],
            [
                [0.03555759 + 0.01100168j, -0.04497981 + 0.02030987j, 0.03198441 - 0.06076387j],
                [-0.07202654 - 0.01724066j, 0.09896394 - 0.04999461j, -0.0084444 + 0.042322j],
                [0.03960783 + 0.09361331j, 0.10660842 + 0.0j, 0.02629697 - 0.08574598j],
            ],
            [
                [-0.01982137 + 0.04605857j, -0.05656505 - 0.06693942j, 0.02845793 + 0.00996117j],
                [-0.03980968 - 0.11268854j, 0.08419826 + 0.06733029j, 0.00769262 - 0.03245046j],
                [-0.08419771 + 0.07692928j, 0.02629697 + 0.08574598j, 0.13023096 + 0.0j],
            ],
        ],
    ]
)

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


methods = [apply_operation_einsum, apply_operation_tensordot, apply_operation]


def test_custom_operator_with_matrix():
    """Test that apply_operation works with any operation that defines a matrix."""
    pass


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("wire", (0, 1))
class TestTwoQubitStateSpecialCases:
    """Test the special cases on a two qubit state.  Also tests the special cases for einsum and tensor application methods
    for additional testing of these generic matrix application methods."""

    def test_TAdd(self, method, wire, ml_framework):
        """Test the application of a TAdd gate on a two qutrit state."""
        initial_state = math.asarray(density_matrix, like=ml_framework)

        control = wire
        target = int(not control)
        new_state = method(qml.TAdd((control, target)), initial_state)
        print(new_state)

        initial0 = math.take(initial_state, 0, axis=control)#TODO Fix for mixed!
        new0 = math.take(new_state, 0, axis=control)
        assert math.allclose(initial0, new0)

        initial1 = math.take(initial_state, 1, axis=control)
        new1 = math.take(new_state, 1, axis=control)
        assert math.allclose(initial1[2], new1[0])
        assert math.allclose(initial1[0], new1[1])
        assert math.allclose(initial1[1], new1[2])

        initial1 = math.take(initial_state, 2, axis=control)
        new1 = math.take(new_state, 2, axis=control)
        assert math.allclose(initial1[1], new1[0])
        assert math.allclose(initial1[2], new1[1])
        assert math.allclose(initial1[0], new1[2])

    def test_TShift(self, method, wire, ml_framework):
        """Test the application of a TShift gate on a two qutrit state."""
        initial_state = math.asarray(density_matrix, like=ml_framework)
        new_state = method(qml.TShift(wire), initial_state)

        initial0dim = math.take(initial_state, 0, axis=wire)#TODO Fix for mixed!
        new1dim = math.take(new_state, 1, axis=wire)

        assert math.allclose(initial0dim, new1dim)

        initial1dim = math.take(initial_state, 1, axis=wire)
        new2dim = math.take(new_state, 2, axis=wire)
        assert math.allclose(initial1dim, new2dim)

        initial2dim = math.take(initial_state, 2, axis=wire)
        new0dim = math.take(new_state, 0, axis=wire)
        assert math.allclose(initial2dim, new0dim)

    def test_TClock(self, method, wire, ml_framework):
        """Test the application of a TClock gate on a two qutrit state."""
        initial_state = math.asarray(density_matrix, like=ml_framework)
        new_state = method(qml.TShift(wire), initial_state)
        w = math.exp(2j*math.pi/3)

        new0 = math.take(new_state, 0, axis=wire)
        initial0 = math.take(initial_state, 0, axis=wire)
        assert math.allclose(new0, initial0)

        initial1 = math.take(initial_state, 1, axis=wire)
        new1 = math.take(new_state, 1, axis=wire)
        assert math.allclose(w * initial1, new1)

        initial2 = math.take(initial_state, 2, axis=wire)
        new2 = math.take(new_state, 2, axis=wire)
        assert math.allclose(w**2 * initial2, new2)

    # TODO: Add more tests as Special cases are added


# TODO add normal test


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestSnapshot:
    """Test that apply_operation works for Snapshot ops"""

    class Debugger:  # pylint: disable=too-few-public-methods
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

    def test_no_debugger(self, ml_framework):
        """Test nothing happens when there is no debugger"""
        initial_state = math.asarray(density_matrix, like=ml_framework)
        new_state = apply_operation(qml.Snapshot(), initial_state)

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

    def test_empty_tag(self, ml_framework):
        """Test a snapshot is recorded properly when there is no tag"""
        initial_state = math.asarray(density_matrix, like=ml_framework)

        debugger = self.Debugger()
        new_state = apply_operation(qml.Snapshot(), initial_state, debugger=debugger)

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [0]
        assert debugger.snapshots[0].shape == (9, 9)
        assert math.allclose(debugger.snapshots[0], math.reshape(initial_state, (9, 9)))

    def test_provided_tag(self, ml_framework):
        """Test a snapshot is recorded property when provided a tag"""
        initial_state = math.asarray(density_matrix, like=ml_framework)

        debugger = self.Debugger()
        tag = "dense"
        new_state = apply_operation(qml.Snapshot(tag), initial_state, debugger=debugger)

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [tag]
        assert debugger.snapshots[tag].shape == (9, 9)
        assert math.allclose(debugger.snapshots[tag], math.reshape(initial_state, (9, 9)))



@pytest.mark.parametrize("method", methods, "subspace")
class TestTRXCalcGrad:
    """Tests the application and differentiation of an TRX gate in the different interfaces."""

    state = density_matrix

    def compare_expected_result(self, phi, state, new_state, g):
        """Compare TODO"""
        expected0 = np.cos(phi / 2) * state[0, :, :] + -1j * np.sin(phi / 2) * state[1, :, :]
        expected1 = -1j * np.sin(phi / 2) * state[0, :, :] + np.cos(phi / 2) * state[1, :, :]

        assert math.allclose(new_state[0, :, :], expected0)
        assert math.allclose(new_state[1, :, :], expected1)

        g_expected0 = (
            -0.5 * np.sin(phi / 2) * state[0, :, :] - 0.5j * np.cos(phi / 2) * state[1, :, :]
        )
        g_expected1 = (
            -0.5j * np.cos(phi / 2) * state[0, :, :] - 0.5 * np.sin(phi / 2) * state[1, :, :]
        )

        assert math.allclose(g[0], g_expected0)
        assert math.allclose(g[1], g_expected1)

    @pytest.mark.autograd
    def test_rx_grad_autograd(self, method):
        """Test that the application of an rx gate is differentiable with autograd."""
        state = qml.numpy.array(self.state)

        def f(phi):
            op = qml.TRX(phi, wires=0)
            return method(op, state)

        phi = qml.numpy.array(0.325 + 0j, requires_grad=True)

        new_state = f(phi)
        g = qml.jacobian(lambda x: math.real(f(x)))(phi)
        self.compare_expected_result(phi, state, new_state, g)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_rx_grad_jax(self, method, use_jit):
        """Test that the application of an rx gate is differentiable with jax."""
        import jax

        state = jax.numpy.array(self.state)

        def f(phi):
            op = qml.RX(phi, wires=0)
            return method(op, state)

        if use_jit:
            f = jax.jit(f)

        phi = 0.325

        new_state = f(phi)
        g = jax.jacobian(f, holomorphic=True)(phi + 0j)
        self.compare_expected_result(phi, state, new_state, g)

    @pytest.mark.torch
    def test_rx_grad_torch(self, method):
        """Tests the application and differentiation of an rx gate with torch."""
        import torch

        state = torch.tensor(self.state)

        def f(phi):
            op = qml.RX(phi, wires=0)
            return method(op, state)

        phi = torch.tensor(0.325, requires_grad=True)

        new_state = f(phi)
        # forward-mode needed with complex results.
        # See bug: https://github.com/pytorch/pytorch/issues/94397
        g = torch.autograd.functional.jacobian(f, phi + 0j, strategy="forward-mode", vectorize=True)

        self.compare_expected_result(
            phi.detach().numpy(),
            state.detach().numpy(),
            new_state.detach().numpy(),
            g.detach().numpy(),
        )

    @pytest.mark.tf
    def test_rx_grad_tf(self, method):
        """Tests the application and differentiation of an rx gate with tensorflow"""
        import tensorflow as tf

        state = tf.Variable(self.state)
        phi = tf.Variable(0.8589 + 0j)

        with tf.GradientTape() as grad_tape:
            op = qml.RX(phi, wires=0)
            new_state = method(op, state)

        grads = grad_tape.jacobian(new_state, [phi])
        # tf takes gradient with respect to conj(z), so we need to conj the gradient
        phi_grad = tf.math.conj(grads[0])

        self.compare_expected_result(phi, state, new_state, phi_grad)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", methods)
class TestBroadcasting:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations are applied correctly."""

    broadcasted_ops = []
    unbroadcasted_ops = []

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op(self, op, method, ml_framework):
        """Tests that batched operations are applied correctly to an unbatched state."""
        pass

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_broadcasted_state(self, op, method, ml_framework):
        """Tests that unbatched operations are applied correctly to a batched state."""
        pass

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op_broadcasted_state(self, op, method, ml_framework):
        """Tests that batched operations are applied correctly to a batched state."""
        pass

    def test_batch_size_set_if_missing(self, method, ml_framework):
        """Tests that the batch_size is set on an operator if it was missing before.
        Mostly useful for TF-autograph since it may have batch size set to None."""
        pass


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", methods)
class TestBroadcasting:  # pylint: disable=too-few-public-methods
    """Tests that Arbritary Channel operation applied correctly."""

    def test_channel(self, method, ml_framework):
        """Tests that channels are applied correctly to an unbatched state."""
        pass
