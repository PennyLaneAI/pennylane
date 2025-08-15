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
Tests for the classical and quantum fisher information matrix in the module pennylane.gradients
"""
import numpy as np

# pylint: disable=no-self-use, import-outside-toplevel, no-member, import-error, too-few-public-methods, bad-continuation
import pytest

import pennylane as qml
import pennylane.numpy as pnp
from pennylane.gradients.fisher import _compute_cfim, _make_probs, classical_fisher, quantum_fisher


class TestMakeProbs:
    """Class to test the private function _make_probs."""

    def test_make_probs_makes_probs(self):
        """Testing the correctness of _make_probs."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode(x):
            qml.RX(x, 0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        x = pnp.array(0.5)
        new_qnode = _make_probs(qnode)
        res = new_qnode(x)

        assert isinstance(res, np.ndarray)
        assert res.shape == (4,)
        assert np.isclose(sum(res), 1)

    @pytest.mark.parametrize("shots", [None, 100])
    def test_make_probs(self, shots):
        """Testing the private _make_probs transform"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(0)
            qml.PauliZ(1)
            qml.PauliY(2)
        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        new_tape, fn = _make_probs(tape)
        assert len(new_tape) == 1
        assert np.isclose(fn([1]), 1)
        assert new_tape[0].shots == tape.shots


class TestComputeClassicalFisher:
    """Testing that given p and dp, _compute_cfim() computes the correct outputs"""

    @pytest.mark.parametrize("n_params", np.arange(1, 10))
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_construction_of_compute_cfim(self, n_params, n_wires):
        """Test that the construction in _compute_cfim is correct"""
        dp = np.arange(2**n_wires * n_params, dtype=float).reshape(2**n_wires, n_params)
        p = np.ones(2**n_wires)

        res = _compute_cfim(p, dp)

        assert np.allclose(res, res.T)
        assert all(
            res[i, j] == np.sum(dp[:, i] * dp[:, j] / p)
            for i in range(n_params)
            for j in range(n_params)
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
        res = _compute_cfim(p, dp)
        assert np.allclose(res, np.ones((n_params, n_params)))


class TestIntegration:
    """Integration test of classical and quantum fisher information matrices"""

    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    @pytest.mark.parametrize("n_params", np.arange(1, 5))
    def test_different_sizes(self, n_wires, n_params):
        """Testing that for any number of wires and parameters, the correct size and values are computed"""
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
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
        res = classical_fisher(circ)(params)
        assert np.allclose(res, n_wires * np.ones((n_params, n_params)))

    def test_hardware_compatibility_classical_fisher(self):
        """Testing that classical_fisher can be computed with finite shots"""
        n_wires = 3
        n_params = 3

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.set_shots(10000)
        @qml.qnode(dev)
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
        res = classical_fisher(circ)(params)
        assert np.allclose(res, n_wires * np.ones((n_params, n_params)), atol=1)

    @pytest.mark.parametrize(
        "dev",
        (
            qml.device("default.qubit"),
            qml.device("default.mixed", wires=3),
            qml.device("lightning.qubit", wires=3),
        ),
    )
    def test_quantum_fisher_info(self, dev, seed):
        """Integration test of quantum fisher information matrix CFIM. This is just calling
        ``qml.metric_tensor`` or ``qml.adjoint_metric_tensor`` and multiplying by a factor of 4"""

        n_wires = 2

        rng = pnp.random.default_rng(seed)
        dev_hard = qml.device("default.qubit", wires=n_wires + 1, seed=rng)

        def qfunc(params):
            qml.RX(params[1], wires=0)
            qml.RX(params[0] / 3, wires=0)
            qml.CNOT(wires=(0, 1))
            return qml.probs(wires=[0, 1])

        params = rng.random(2, requires_grad=True)

        circ_hard = qml.set_shots(qml.QNode(qfunc, dev_hard), shots=1000)
        QFIM_hard = quantum_fisher(circ_hard)(params)
        QFIM1_hard = 4.0 * qml.metric_tensor(circ_hard)(params)

        circ = qml.QNode(qfunc, dev)
        QFIM = quantum_fisher(circ)(params)
        QFIM1 = 4.0 * qml.adjoint_metric_tensor(circ)(params)
        assert np.allclose(QFIM, QFIM1)
        assert np.allclose(QFIM_hard, QFIM1_hard, atol=1e-1)


class TestInterfacesClassicalFisher:
    """Integration tests for the classical fisher information matrix CFIM"""

    @pytest.mark.autograd
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_cfim_allnonzero_autograd(self, n_wires):
        """Integration test of classical_fisher() with autograd for examples where all probabilities are all nonzero"""

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            for i in range(n_wires):
                qml.RX(params[0], wires=i)
            for i in range(n_wires):
                qml.RY(params[1], wires=i)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * pnp.ones(2, requires_grad=True)
        cfim = classical_fisher(circ)(params)
        assert np.allclose(cfim, (n_wires / 3.0) * np.ones((2, 2)))

    @pytest.mark.autograd
    @pytest.mark.parametrize("n_wires", np.arange(2, 5))
    def test_cfim_contains_zeros_autograd(self, n_wires):
        """Integration test of classical_fisher() with autograd for examples that have 0s in the probabilities and non-zero gradient"""
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[0], wires=1)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * pnp.ones(2, requires_grad=True)
        cfim = classical_fisher(circ)(params)
        assert np.allclose(cfim, np.ones((2, 2)))

    @pytest.mark.jax
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_cfim_allnonzero_jax(self, n_wires):
        """Integration test of classical_fisher() with jax for examples where all probabilities are all nonzero"""
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            for i in range(n_wires):
                qml.RX(params[0], wires=i)
            for i in range(n_wires):
                qml.RY(params[1], wires=i)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * jnp.ones(2)
        cfim = classical_fisher(circ)(params)
        assert np.allclose(cfim, (n_wires / 3.0) * np.ones((2, 2)))

    @pytest.mark.jax
    @pytest.mark.parametrize("n_wires", np.arange(2, 5))
    def test_cfim_contains_zeros_jax(self, n_wires):
        """Integration test of classical_fisher() with jax for examples that have 0s in the probabilities and non-zero gradient"""
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[0], wires=1)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * jnp.ones(2)
        cfim = classical_fisher(circ)(params)
        assert np.allclose(cfim, np.ones((2, 2)))

    @pytest.mark.jax
    def test_cfim_multiple_args_jax(self):
        """Testing multiple args to be differentiated using jax"""
        import jax.numpy as jnp

        n_wires = 3

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(x, y, z):
            for xi in x:
                qml.RX(xi, wires=0)
                qml.RX(xi, wires=1)
            for yi in y:
                qml.RY(yi, wires=0)
                qml.RY(yi, wires=1)
            for zi in z:
                qml.RZ(zi, wires=0)
                qml.RZ(zi, wires=1)
            return qml.probs(wires=range(n_wires))

        x = jnp.pi / 8 * jnp.ones(2)
        y = jnp.pi / 8 * jnp.ones(10)
        z = jnp.ones(1)
        cfim = classical_fisher(circ, argnums=(0, 1, 2))(x, y, z)
        assert qml.math.allclose(cfim[0], 2.0 / 3.0 * np.ones((2, 2)))
        assert qml.math.allclose(cfim[1], 2.0 / 3.0 * np.ones((10, 10)))
        assert qml.math.allclose(cfim[2], np.zeros((1, 1)))

    @pytest.mark.torch
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_cfim_allnonzero_torch(self, n_wires):
        """Integration test of classical_fisher() with torch for examples where all probabilities are all nonzero"""
        import torch

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            for i in range(n_wires):
                qml.RX(params[0], wires=i)
            for i in range(n_wires):
                qml.RY(params[1], wires=i)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * torch.tensor([1.0, 1.0], requires_grad=True)
        cfim = classical_fisher(circ)(params)
        assert np.allclose(cfim.detach().numpy(), (n_wires / 3.0) * np.ones((2, 2)))

    @pytest.mark.torch
    @pytest.mark.parametrize("n_wires", np.arange(2, 5))
    def test_cfim_contains_zeros_torch(self, n_wires):
        """Integration test of classical_fisher() with torch for examples that have 0s in the probabilities and non-zero gradient"""
        import torch

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[0], wires=1)
            return qml.probs(wires=range(n_wires))

        params = np.pi / 4 * torch.tensor([1.0, 1.0], requires_grad=True)
        cfim = classical_fisher(circ)(params)
        assert np.allclose(cfim.detach().numpy(), np.ones((2, 2)))

    @pytest.mark.torch
    def test_cfim_multiple_args_torch(self):
        """Testing multiple args to be differentiated using torch"""
        import torch

        n_wires = 3

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(x, y, z):
            for xi in x:
                qml.RX(xi, wires=0)
                qml.RX(xi, wires=1)
            for yi in y:
                qml.RY(yi, wires=0)
                qml.RY(yi, wires=1)
            for zi in z:
                qml.RZ(zi, wires=0)
                qml.RZ(zi, wires=1)
            return qml.probs(wires=range(n_wires))

        x = np.pi / 8 * torch.ones(2, requires_grad=True)
        y = np.pi / 8 * torch.ones(10, requires_grad=True)
        z = torch.ones(1, requires_grad=True)
        cfim = classical_fisher(circ)(x, y, z)
        assert np.allclose(cfim[0].detach().numpy(), 2.0 / 3.0 * np.ones((2, 2)))
        assert np.allclose(cfim[1].detach().numpy(), 2.0 / 3.0 * np.ones((10, 10)))
        assert np.allclose(cfim[2].detach().numpy(), np.zeros((1, 1)))

    @pytest.mark.tf
    @pytest.mark.parametrize("n_wires", np.arange(1, 5))
    def test_cfim_allnonzero_tf(self, n_wires):
        """Integration test of classical_fisher() with tf for examples where all probabilities are all nonzero"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            for i in range(n_wires):
                qml.RX(params[0], wires=i)
            for i in range(n_wires):
                qml.RY(params[1], wires=i)
            return qml.probs(wires=range(n_wires))

        params = tf.Variable([np.pi / 4, np.pi / 4])
        cfim = classical_fisher(circ)(params)
        assert np.allclose(cfim, (n_wires / 3.0) * np.ones((2, 2)))

    @pytest.mark.tf
    @pytest.mark.parametrize("n_wires", np.arange(2, 5))
    def test_cfim_contains_zeros_tf(self, n_wires):
        """Integration test of classical_fisher() with tf for examples that have 0s in the probabilities and non-zero gradient"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.RX(params[0], wires=1)
            return qml.probs(wires=range(n_wires))

        params = tf.Variable([np.pi / 4, np.pi / 4])
        cfim = classical_fisher(circ)(params)
        assert np.allclose(cfim, np.ones((2, 2)))

    @pytest.mark.tf
    def test_cfim_multiple_args_tf(self):
        """Testing multiple args to be differentiated using tf"""
        import tensorflow as tf

        n_wires = 3

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(x, y, z):
            for xi in tf.unstack(x):
                qml.RX(xi, wires=0)
                qml.RX(xi, wires=1)
            for yi in tf.unstack(y):
                qml.RY(yi, wires=0)
                qml.RY(yi, wires=1)
            for zi in tf.unstack(z):
                qml.RZ(zi, wires=0)
                qml.RZ(zi, wires=1)
            return qml.probs(wires=range(n_wires))

        x = tf.Variable(np.pi / 8 * np.ones(2), trainable=True)
        y = tf.Variable(np.pi / 8 * np.ones(10), trainable=True)
        z = tf.Variable([1.0], trainable=True)
        cfim = classical_fisher(circ)(x, y, z)
        assert np.allclose(cfim[0], 2.0 / 3.0 * np.ones((2, 2)))
        assert np.allclose(cfim[1], 2.0 / 3.0 * np.ones((10, 10)))
        assert np.allclose(cfim[2], np.zeros((1, 1)))


class TestDiffCFIM:
    """Testing differentiability of classical fisher info matrix (CFIM)"""

    @pytest.mark.autograd
    def test_diffability_autograd(self):
        """Testing diffability with an analytic example for autograd. The CFIM of this single qubit is constant, so the gradient should be zero."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(params):
            qml.RY(params, wires=0)
            return qml.probs(wires=range(1))

        params = pnp.array(np.pi / 4, requires_grad=True)

        assert np.allclose(classical_fisher(circ)(params), 1)

        result = np.zeros((1, 1, 1), dtype="float64")
        result_calc = qml.jacobian(classical_fisher(circ))(params)

        assert np.allclose(result, result_calc, atol=1e-6)

    @pytest.mark.jax
    def test_diffability_jax(self):
        """Testing diffability with an analytic example for jax. The CFIM of this single qubit is constant, so the gradient should be zero."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(params):
            qml.RY(params, wires=0)
            return qml.probs(wires=range(1))

        # no matter what the input the CFIM here is always constant = 1
        # so the derivative should be 0

        params = jnp.array(np.pi / 4)

        assert qml.math.allclose(classical_fisher(circ)(params), 1.0)

        result = np.zeros((1, 1, 1), dtype="float64")
        result_calc = jax.jacobian(classical_fisher(circ))(params)

        assert np.allclose(result, result_calc, atol=1e-6)

    @pytest.mark.tf
    def test_diffability_tf(self):
        """Testing diffability with an analytic example for tf. The CFIM of this single
        qubit is constant, so the gradient should be zero."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(params):
            qml.RY(params, wires=0)
            return qml.probs(wires=range(1))

        params = tf.Variable(np.pi / 4)

        assert np.allclose(classical_fisher(circ)(params), 1)

        result = np.zeros((1, 1, 1), dtype="float64")
        with tf.GradientTape() as tape:
            loss = classical_fisher(circ)(params)

        result_calc = tape.jacobian(loss, params)

        assert np.allclose(result, result_calc, atol=1e-6)

    @pytest.mark.torch
    def test_diffability_torch(self):
        """Testing diffability with an analytic example for torch. The CFIM of this single qubit is constant, so the gradient should be zero."""
        import torch

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(params):
            qml.RY(params, wires=0)
            return qml.probs(wires=range(1))

        params = torch.tensor(np.pi / 4, requires_grad=True)

        assert np.allclose(classical_fisher(circ)(params).detach().numpy(), 1)

        result = np.zeros((1, 1, 1), dtype="float64")
        result_calc = torch.autograd.functional.jacobian(classical_fisher(circ), params)

        assert np.allclose(result, result_calc, atol=1e-6)

    @pytest.mark.all_interfaces
    def test_consistency(self):
        """Testing that the derivative of the cfim is giving consistently the same results for all interfaces.
        Currently failing as (jax and autograd) and (torch and tf) are giving two different results.
        """
        import jax
        import jax.numpy as jnp
        import torch

        dev = qml.device("default.qubit", wires=3)

        def qfunc(weights):
            qml.RX(weights[0], wires=0)
            qml.RX(weights[1], wires=1)
            qml.RX(weights[2], wires=2)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[4], wires=0)
            qml.RY(weights[5], wires=1)
            qml.RY(weights[6], wires=2)
            qml.CNOT(wires=[1, 2])
            qml.CRX(weights[7], wires=[0, 1])
            qml.CRY(weights[8], wires=[1, 2])
            qml.CRZ(weights[9], wires=[2, 0])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        # Compute gradients of CFIM for different interfaces
        circuit = qml.QNode(qfunc, dev)
        weights = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], requires_grad=True
        )
        grad_torch = torch.autograd.functional.jacobian(classical_fisher(circuit), weights)

        circuit = qml.QNode(qfunc, dev)
        weights = pnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], requires_grad=True)
        grad_autograd = qml.jacobian(classical_fisher(circuit))(weights)

        circuit = qml.QNode(qfunc, dev)
        weights = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        grad_jax = jax.jacobian(classical_fisher(circuit))(weights)

        # Evaluate and compare
        grads = [grad_autograd, grad_torch, grad_jax]

        is_same = np.zeros((len(grads), len(grads)))

        for i, g1 in enumerate(grads):
            for j, g2 in enumerate(grads):
                is_same[i, j] = np.allclose(g1, g2, atol=1e-6)

        assert np.allclose(is_same, np.ones((len(grads), len(grads))))
