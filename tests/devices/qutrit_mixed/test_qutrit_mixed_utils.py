# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for util functions in devices/qutrit_mixed."""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.qutrit_mixed.utils import get_eigvals, expand_qutrit_vector

w = np.exp(2j * np.pi / 3)


class TestGetEigvals:
    @pytest.mark.parametrize(
        "observable, expected_eigvals",
        [
            (qml.GellMann(0, 8), np.array((1, 1, -2)) / np.sqrt(3)),
            (qml.s_prod(np.sqrt(3), qml.GellMann(0, 8)), np.array((1, 1, -2))),
            (
                qml.prod(qml.GellMann(0, 8), qml.GellMann(1, 1)),
                np.array((1, -1, 0, 1, -1, 0, -2, 2, 0)) / np.sqrt(3),
            ),
            (
                qml.prod(
                    qml.GellMann(0, 8), qml.GellMann(0, 1), qml.THadamard(1), qml.GellMann(1, 3)
                ),
                np.kron(
                    np.array([1, -1, 0]),
                    np.array(
                        [
                            0,
                            1 - w - np.sqrt(w**2 + 2 * w - 3),
                            1 - w + np.sqrt(w**2 + 2 * w - 3),
                        ]
                    ),
                )
                * -1j
                / 6,
            ),
            (
                qml.s_prod(np.sqrt(3), qml.prod(qml.GellMann(0, 1), qml.GellMann(1, 8))),
                np.array((1, 1, -2, -1, -1, 2, 0, 0, 0)),
            ),
        ],
    )
    def test_get_obs_eigvals(self, observable, expected_eigvals):
        """Tests get_eigvals gets correct eigenvalues for observables, especially prod and s_prod"""
        eigvals = get_eigvals(observable)
        assert np.allclose(eigvals, expected_eigvals)

    s = 2.34
    expected_g = np.array((1, -1, 0, 1, -1, 0, -2, 2, 0)) / np.sqrt(3)

    @staticmethod
    def f(s):
        obs = qml.s_prod(s, qml.prod(qml.GellMann(0, 8), qml.GellMann(1, 1)))
        return get_eigvals(obs)

    @pytest.mark.autograd
    def test_s_prod_eigval_dif_autograd(self):
        """Test that s_prod scalar gradients work in autograd"""
        s = qml.numpy.array(self.s)
        g = qml.jacobian(self.f)(s)
        assert qml.math.allclose(g, self.expected_g)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_s_prod_eigval_dif_jax(self, use_jit):
        """Test that s_prod scalar gradients work in jax"""
        import jax

        jax.config.update("jax_enable_x64", True)

        s = jax.numpy.array(self.s, dtype=jax.numpy.float64)
        f = jax.jit(self.f) if use_jit else self.f

        g = jax.jacobian(f)(s)
        assert qml.math.allclose(g, self.expected_g)

    @pytest.mark.torch
    def test_s_prod_eigval_dif_torch(self):
        """Test that s_prod scalar gradients work in torch"""
        import torch

        s = torch.tensor(self.s, requires_grad=True, dtype=torch.float64)
        g = torch.autograd.functional.jacobian(self.f, s)

        assert qml.math.allclose(g.detach().numpy(), self.expected_g)

    @pytest.mark.tf
    def test_s_prod_eigval_dif_tf(self):
        """Test that s_prod scalar gradients work in tensorflow"""
        import tensorflow as tf

        s = tf.Variable(self.s)

        with tf.GradientTape() as tape:
            out = self.f(s)
        g = tape.jacobian(out, s)

        assert qml.math.allclose(g, self.expected_g)


class TestExpandQutritVector:
    """Tests vector expansion to more wires, for qutrit case"""

    w = np.exp(2j * np.pi / 3)
    VECTOR1 = np.array([1, w, w**2])
    ONES = np.array([1, 1, 1])

    @pytest.mark.parametrize(
        "original_wires,expanded_wires,expected",
        [
            ([0], 3, np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([1], 3, np.kron(np.kron(ONES, VECTOR1), ONES)),
            ([2], 3, np.kron(np.kron(ONES, ONES), VECTOR1)),
            ([0], [0, 4, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([4], [0, 4, 7], np.kron(np.kron(ONES, VECTOR1), ONES)),
            ([7], [0, 4, 7], np.kron(np.kron(ONES, ONES), VECTOR1)),
            ([0], [0, 4, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([4], [4, 0, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([7], [7, 4, 0], np.kron(np.kron(VECTOR1, ONES), ONES)),
        ],
    )
    def test_expand_vector_single_wire(self, original_wires, expanded_wires, expected, tol):
        """Test that expand_vector works with a single-wire vector."""

        res = expand_qutrit_vector(TestExpandQutritVector.VECTOR1, original_wires, expanded_wires)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    VECTOR2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    @pytest.mark.parametrize(
        "original_wires,expanded_wires,expected",
        [
            ([0, 1], 3, np.kron(VECTOR2, ONES)),
            ([1, 2], 3, np.kron(ONES, VECTOR2)),
            ([0, 2], 3, np.array(([1, 2, 3] * 3) + ([4, 5, 6] * 3) + ([7, 8, 9] * 3))),
            ([0, 5], [0, 5, 9], np.kron(VECTOR2, ONES)),
            ([5, 9], [0, 5, 9], np.kron(ONES, VECTOR2)),
            ([0, 9], [0, 5, 9], np.array(([1, 2, 3] * 3) + ([4, 5, 6] * 3) + ([7, 8, 9] * 3))),
            ([9, 0], [0, 5, 9], np.array(([1, 4, 7] * 3) + ([2, 5, 8] * 3) + ([3, 6, 9] * 3))),
            ([0, 1], [0, 1], VECTOR2),
        ],
    )
    def test_expand_vector_two_wires(self, original_wires, expanded_wires, expected, tol):
        """Test that expand_vector works with a single-wire vector."""

        res = expand_qutrit_vector(TestExpandQutritVector.VECTOR2, original_wires, expanded_wires)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_vector_invalid_wires(self):
        """Test exception raised if unphysical subsystems provided."""
        with pytest.raises(
            ValueError,
            match="Invalid target subsystems provided in 'original_wires' argument",
        ):
            expand_qutrit_vector(TestExpandQutritVector.VECTOR2, [-1, 5], 4)

    def test_expand_vector_invalid_vector(self):
        """Test exception raised if incorrect sized vector provided."""
        with pytest.raises(ValueError, match="Vector parameter must be of length"):
            expand_qutrit_vector(TestExpandQutritVector.VECTOR1, [0, 1], 4)
