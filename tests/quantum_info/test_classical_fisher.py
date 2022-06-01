# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Tests for the classical fisher information matrix in the pennylane.qinfo
"""
from xml.dom.minidom import Element
import pytest

import pennylane as qml
import numpy as np
import pennylane.numpy as pnp

from pennylane.quantum_info import _compute_cfim, CFIM


class TestComputeCFIMfn:
    """Testing that given p and dp, _compute_cfim() computes the correct outputs"""

    @pytest.mark.parametrize("n_params", np.arange(1, 10))
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_construction_of_compute_cfim(self, n_params, n_wires):
        """Ensuring the construction in _compute_cfim is correct"""
        dp = np.arange(2**n_wires * n_params, dtype=float).reshape(2**n_wires, n_params)
        p = np.ones(2**n_wires)

        res = _compute_cfim(p, dp, None)

        assert np.allclose(res, res.T)
        assert all(
            [
                res[i, j] == np.sum(dp[:, i] * dp[:, j] / p)
                for i in range(n_params)
                for j in range(n_params)
            ]
        )

    @pytest.mark.parametrize("n_params", np.arange(1, 10))
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_compute_cfim_trivial_distribution(self, n_params, n_wires):
        """Test that the classical fisher information matrix (CFIM) is
        constructed correctly for the trivial distirbution p=(1,0,0,...)
        and some dummy gradient dp"""
        # implicitly also does the unit test for the correct output shape

        dp = np.ones(2**n_wires * n_params, dtype=float).reshape(2**n_wires, n_params)
        p = np.zeros(2**n_wires)
        p[0] = 1
        res = _compute_cfim(p, dp, None)
        assert np.allclose(res, np.ones((n_params, n_params)))


class TestIntegration:
    """Integration test of classical fisher information matrix CFIM"""
    
    @pytest.mark.parametrize("n_wires", np.arange(1,5))
    @pytest.mark.parametrize("n_params", np.arange(1,5))
    def test_different_sizes(n_wires, n_params):
        """Testing that for any number of wires and parameters, the correct size and values are computed"""
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="autograd")
        def circ(params):
            for i in range(n_wires):
                qml.Hadamard(wires=i)

            for x in params:
                for j in range(n_wires):
                    qml.RX(x, wires=j)
                    qml.RY(x, wires=j)
                    qml.RZ(x, wires=j)

            return qml.probs(wires=range(n_wires))
        params = pnp.zeros(n_params, requires_grad=True)
        res = qml.quantum_info.CFIM(circ)(params)
        assert np.allclose(res, n_wires * np.ones((n_params, n_params)))


class TestInterfaces:
    """Integration tests for the classical fisher information matrix CFIM"""

    @pytest.mark.autograd
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_cfim_allnonzero_autograd(self, n_wires):
        """Integration test of CFIM() with autograd for examples where all probabilities are all nonzero"""

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="autograd")
        def circ(params):
            for i in range(n_wires):
                qml.RX(params[0], wires=i)
            for i in range(n_wires):
                qml.RY(params[1], wires=i)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * pnp.ones(2, requires_grad=True)
        cfim = CFIM(circ)(params)
        assert np.allclose(cfim, (n_wires / 3.0) * np.ones((2, 2)))

    @pytest.mark.autograd
    @pytest.mark.parametrize("n_wires", np.arange(2, 5))
    def test_cfim_contains_zeros_autograd(self, n_wires):
        """Integration test of CFIM() with autograd for examples that have 0s in the probabilities and non-zero gradient"""
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="autograd")
        def circ(params):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[0], wires=1)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * pnp.ones(2, requires_grad=True)
        cfim = CFIM(circ)(params)
        assert np.allclose(cfim, np.ones((2, 2)))

    @pytest.mark.jax
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_cfim_allnonzero_jax(self, n_wires):
        """Integration test of CFIM() with jax for examples where all probabilities are all nonzero"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="jax")
        def circ(params):
            for i in range(n_wires):
                qml.RX(params[0], wires=i)
            for i in range(n_wires):
                qml.RY(params[1], wires=i)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * jnp.ones(2)
        cfim = CFIM(circ)(params)
        assert np.allclose(cfim, (n_wires / 3.0) * np.ones((2, 2)))

    @pytest.mark.jax
    @pytest.mark.parametrize("n_wires", np.arange(2, 5))
    def test_cfim_contains_zeros_jax(self, n_wires):
        """Integration test of CFIM() with jax for examples that have 0s in the probabilities and non-zero gradient"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="jax")
        def circ(params):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[0], wires=1)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * jnp.ones(2)
        cfim = CFIM(circ)(params)
        assert np.allclose(cfim, np.ones((2, 2)))

    @pytest.mark.torch
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_cfim_allnonzero_torch(self, n_wires):
        """Integration test of CFIM() with torch for examples where all probabilities are all nonzero"""
        import torch

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="torch")
        def circ(params):
            for i in range(n_wires):
                qml.RX(params[0], wires=i)
            for i in range(n_wires):
                qml.RY(params[1], wires=i)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * torch.tensor([1.0, 1.0], requires_grad=True)
        cfim = CFIM(circ)(params)
        assert np.allclose(cfim.detach().numpy(), (n_wires / 3.0) * np.ones((2, 2)))

    @pytest.mark.torch
    @pytest.mark.parametrize("n_wires", np.arange(2, 5))
    def test_cfim_contains_zeros_torch(self, n_wires):
        """Integration test of CFIM() with torch for examples that have 0s in the probabilities and non-zero gradient"""
        import torch

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="torch")
        def circ(params):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[0], wires=1)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * torch.tensor([1.0, 1.0], requires_grad=True)
        cfim = CFIM(circ)(params)
        assert np.allclose(cfim.detach().numpy(), np.ones((2, 2)))

    @pytest.mark.tf
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_cfim_allnonzero_tf(self, n_wires):
        """Integration test of CFIM() with tf for examples where all probabilities are all nonzero"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="tf")
        def circ(params):
            for i in range(n_wires):
                qml.RX(params[0], wires=i)
            for i in range(n_wires):
                qml.RY(params[1], wires=i)
            return qml.probs(wires=range(n_wires))

        params = tf.Variable([np.pi / 4, np.pi / 4])
        cfim = CFIM(circ)(params)
        assert np.allclose(cfim, (n_wires / 3.0) * np.ones((2, 2)))

    @pytest.mark.tf
    @pytest.mark.parametrize("n_wires", np.arange(2, 5))
    def test_cfim_contains_zeros_tf(self, n_wires):
        """Integration test of CFIM() with tf for examples that have 0s in the probabilities and non-zero gradient"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="tf")
        def circ(params):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[0], wires=1)
            return qml.probs(wires=range(n_wires))

        params = tf.Variable([np.pi / 4, np.pi / 4])
        cfim = CFIM(circ)(params)
        assert np.allclose(cfim, np.ones((2, 2)))
