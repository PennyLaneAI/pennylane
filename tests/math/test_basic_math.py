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
"""Unit tests for the basic functions in qml.math
"""
import numpy as onp
import pennylane as qml
from pennylane import numpy as np
import pytest

from pennylane import math as fn

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestFrobeniusInnerProduct:
    """Test the frobenius_inner_product method."""

    @pytest.mark.parametrize(
        "A,B,normalize,expected",
        [
            (onp.eye(2), onp.eye(2), False, 2.0),
            (onp.eye(2), onp.zeros((2, 2)), False, 0.0),
            (
                onp.array([[1.0, 2.3], [-1.3, 2.4]]),
                onp.array([[0.7, -7.3], [-1.0, -2.9]]),
                False,
                -21.75,
            ),
            (onp.eye(2), onp.eye(2), True, 1.0),
            (
                onp.array([[1.0, 2.3], [-1.3, 2.4]]),
                onp.array([[0.7, -7.3], [-1.0, -2.9]]),
                True,
                -0.7381450594,
            ),
            (
                np.array([[1.0, 2.3], [-1.3, 2.4]]),
                np.array([[0.7, -7.3], [-1.0, -2.9]]),
                False,
                -21.75,
            ),
            (
                np.array([[1.0, 2.3], [-1.3, 2.4]]),
                np.array([[0.7, -7.3], [-1.0, -2.9]]),
                True,
                -0.7381450594,
            ),
            (
                jnp.array([[1.0, 2.3], [-1.3, 2.4]]),
                jnp.array([[0.7, -7.3], [-1.0, -2.9]]),
                False,
                -21.75,
            ),
            (
                jnp.array([[1.0, 2.3], [-1.3, 2.4]]),
                jnp.array([[0.7, -7.3], [-1.0, -2.9]]),
                True,
                -0.7381450594,
            ),
            (
                torch.tensor([[1.0, 2.3], [-1.3, 2.4]], dtype=torch.complex128),
                torch.tensor([[0.7, -7.3], [-1.0, -2.9]], dtype=torch.complex128),
                False,
                -21.75,
            ),
            (
                torch.tensor([[1.0, 2.3], [-1.3, 2.4]], dtype=torch.complex128),
                torch.tensor([[0.7, -7.3], [-1.0, -2.9]], dtype=torch.complex128),
                True,
                -0.7381450594,
            ),
            (
                tf.Variable([[1.0, 2.3], [-1.3, 2.4]], dtype=tf.complex128),
                tf.Variable([[0.7, -7.3], [-1.0, -2.9]], dtype=tf.complex128),
                False,
                -21.75,
            ),
            (
                tf.Variable([[1.0, 2.3], [-1.3, 2.4]], dtype=tf.complex128),
                tf.Variable([[0.7, -7.3], [-1.0, -2.9]], dtype=tf.complex128),
                True,
                -0.7381450594,
            ),
            (
                tf.constant([[1.0, 2.3], [-1.3, 2.4]], dtype=tf.complex128),
                tf.constant([[0.7, -7.3], [-1.0, -2.9]], dtype=tf.complex128),
                False,
                -21.75,
            ),
            (
                tf.constant([[1.0, 2.3], [-1.3, 2.4]], dtype=tf.complex128),
                tf.constant([[0.7, -7.3], [-1.0, -2.9]], dtype=tf.complex128),
                True,
                -0.7381450594,
            ),
        ],
    )
    def test_frobenius_inner_product(self, A, B, normalize, expected):
        """Test that the calculated inner product is as expected."""
        assert expected == pytest.approx(fn.frobenius_inner_product(A, B, normalize=normalize))

    def test_frobenius_inner_product_gradient(self):
        """Test that the calculated gradient is correct."""
        A = onp.array([[1.0, 2.3], [-1.3, 2.4]])
        B = torch.autograd.Variable(torch.randn(2, 2).type(torch.float), requires_grad=True)
        result = fn.frobenius_inner_product(A, B)
        result.backward()
        grad = B.grad

        assert np.allclose(grad, A)
