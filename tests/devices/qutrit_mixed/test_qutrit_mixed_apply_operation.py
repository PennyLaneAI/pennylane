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

    def test_identity(self, method, wire, ml_framework):
        """Test the application of an Identity gate on a two qutrit state."""
        pass

    def test_TAdd(self, method, wire, ml_framework):
        """Test the application of an TAdd gate on a two qutrit state."""
        pass

    def test_diagonal_in_z(self, method, wire, ml_framework):
        """Test the application of an Identity gate on a two qutrit state."""
        pass
@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestSnapshot:
    """Test that apply_operation works for Snapshot ops"""
    pass

@pytest.mark.parametrize("method", methods)
class TestTRXCalcGrad:
    """Tests the application and differentiation of an TRX gate in the different interfaces."""
    state = None # TODO add init state

    def compare_expected_result(self, phi, state, new_state, g):
        """Compare TODO"""
        pass

    @pytest.mark.autograd
    def test_rx_grad_autograd(self, method):
        """Test that the application of an rx gate is differentiable with autograd."""
        pass

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_rx_grad_jax(self, method, use_jit):
        """Test that the application of an rx gate is differentiable with jax."""

        import jax
        pass

    @pytest.mark.torch
    def test_rx_grad_torch(self, method):
        """Tests the application and differentiation of an rx gate with torch."""

        import torch
        pass

    @pytest.mark.tf
    def test_rx_grad_tf(self, method):
        """Tests the application and differentiation of an rx gate with tensorflow"""
        import tensorflow as tf
        pass


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", methods)
class TestBroadcasting:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations are applied correctly."""

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