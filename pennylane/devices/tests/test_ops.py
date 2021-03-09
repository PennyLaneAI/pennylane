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
Tests operations defined across different interfaces
"""

import pennylane as qml
from pennylane.devices import jax_ops, tf_ops, autograd_ops
import numpy as np
import jax.numpy as np
import pytest


class TestJaxOps:
    """
    Tests that the matrix defining jax operations matches the intended one
    """

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation(self, phi):

        jx_mat = jax_ops.SingleExcitation(phi)
        m = qml.SingleExcitation(phi, wires=[0, 1])

        assert np.allclose(jx_mat, m.matrix)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_plus(self, phi):
        jx_mat = jax_ops.SingleExcitationPlus(phi)
        m = qml.SingleExcitationPlus(phi, wires=[0, 1])

        assert np.allclose(jx_mat, m.matrix)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_minus(self, phi):
        jx_mat = jax_ops.SingleExcitationMinus(phi)
        m = qml.SingleExcitationMinus(phi, wires=[0, 1])

        assert np.allclose(jx_mat, m.matrix)


class TestAutogradOps:
    """
    Tests that the matrix defining autograd operations matches the intended one
    """

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation(self, phi):

        a_mat = autograd_ops.SingleExcitation(phi)
        m = qml.SingleExcitation(phi, wires=[0, 1])

        assert np.allclose(a_mat, m.matrix)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_plus(self, phi):
        a_mat = autograd_ops.SingleExcitationPlus(phi)
        m = qml.SingleExcitationPlus(phi, wires=[0, 1])

        assert np.allclose(a_mat, m.matrix)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_minus(self, phi):
        a_mat = autograd_ops.SingleExcitationMinus(phi)
        m = qml.SingleExcitationMinus(phi, wires=[0, 1])

        assert np.allclose(a_mat, m.matrix)


class TestTFOps:
    """
    Tests that the matrix defining tensorflow operations matches the intended one
    """

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation(self, phi):

        tf_mat = tf_ops.SingleExcitation(phi)
        m = qml.SingleExcitation(phi, wires=[0, 1])

        assert np.allclose(tf_mat, m.matrix)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_plus(self, phi):
        tf_mat = tf_ops.SingleExcitationPlus(phi)
        m = qml.SingleExcitationPlus(phi, wires=[0, 1])

        assert np.allclose(tf_mat, m.matrix)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_minus(self, phi):
        tf_mat = tf_ops.SingleExcitationMinus(phi)
        m = qml.SingleExcitationMinus(phi, wires=[0, 1])

        assert np.allclose(tf_mat, m.matrix)
