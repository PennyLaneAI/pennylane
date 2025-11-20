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
"""
Tests for the AmplitudeAmplification template.
"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.amplitude_amplification import _get_fixed_point_angles


@qml.prod
def generator(wires):
    for wire in wires:
        qml.Hadamard(wire)


@qml.prod
def oracle(items, wires):
    for item in items:
        qml.FlipSign(item, wires=wires)


class TestInitialization:
    """Test that AmplitudeAmplification initializes correctly."""

    def test_error_none_wire(self):
        """Test that an error is raised if work_wire is None and fixed_point is True."""

        U = generator(wires=range(3))
        O = oracle([0, 2], wires=range(3))

        with pytest.raises(
            qml.wires.WireError, match="work_wire must be specified if fixed_point == True."
        ):
            qml.AmplitudeAmplification(U, O, iters=3, fixed_point=True)

    @pytest.mark.parametrize(
        "wires, fixed_point, work_wire",
        (
            ([0, 1, 2], True, 2),
            (["a", "b"], True, "a"),
        ),
    )
    def test_error_wrong_work_wire(self, wires, fixed_point, work_wire):
        """Test that an error is raised if work_wire is part of the O wires."""

        U = generator(wires=wires)
        O = oracle([0], wires=wires)

        with pytest.raises(ValueError, match="work_wire must be different from the wires of O."):
            qml.AmplitudeAmplification(U, O, iters=3, fixed_point=fixed_point, work_wire=work_wire)

    @pytest.mark.jax
    def test_standard_validity(self):
        """Test standard validity using assert_valid."""
        U = generator(wires=range(3))
        O = oracle([0, 2], wires=range(3))
        op = qml.AmplitudeAmplification(U, O, iters=3, fixed_point=False)
        qml.ops.functions.assert_valid(op)


@pytest.mark.parametrize(
    "n_wires, items, iters",
    (
        (3, [0, 2], 1),
        (3, [1, 2], 2),
        (5, [4, 5, 7, 12], 3),
        (5, [0, 1, 2, 3, 4], 4),
    ),
)
def test_compare_grover(n_wires, items, iters):
    """Test that Grover's algorithm gives the same result with GroverOperator and AmplitudeAmplification."""
    U = generator(wires=range(n_wires))
    O = oracle(items, wires=range(n_wires))

    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit_amplitude_amplification():
        generator(wires=range(n_wires))
        qml.AmplitudeAmplification(U, O, iters)
        return qml.probs(wires=range(n_wires))

    @qml.qnode(dev)
    def circuit_grover():
        generator(wires=range(n_wires))

        for _ in range(iters):
            oracle(items, wires=range(n_wires))
            qml.GroverOperator(wires=range(n_wires))

        return qml.probs(wires=range(n_wires))

    assert np.allclose(circuit_amplitude_amplification(), circuit_grover(), atol=1e-5)


def test_default_lightning_devices():
    """Test that AmplitudeAmplification executes with the default.qubit and lightning.qubit simulators."""

    def circuit():
        """Test circuit"""
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)

        qml.AmplitudeAmplification(
            generator(range(3)), oracle([0], range(3)), fixed_point=True, iters=3, work_wire=3
        )
        return qml.probs(wires=range(3))

    dev1 = qml.device("default.qubit")
    qnode1 = qml.QNode(circuit, dev1, interface=None)

    res1 = qnode1()

    dev2 = qml.device("lightning.qubit", wires=4)
    qnode2 = qml.QNode(circuit, dev2)

    res2 = qnode2()

    assert np.allclose(res1, res2, atol=1e-5)


class TestDifferentiability:
    """Test that AmplitudeAmplification is differentiable"""

    @staticmethod
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.AmplitudeAmplification(
            qml.RY(params[0], wires=0),
            qml.RZ(params[1], wires=0),
            iters=3,
            fixed_point=True,
            work_wire=2,
        )

        return qml.expval(qml.PauliZ(0))

    # calculated numerically with finite diff method (h = 1e-5)
    exp_grad = np.array([-0.88109663, -0.66429297])

    params = np.array([0.9, 0.1])

    @pytest.mark.autograd
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_autograd(self, shots, seed):
        """Test that the QNode executes with Autograd."""

        dev = qml.device("default.qubit", wires=3, seed=seed)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="autograd", diff_method=diff_method), shots=shots
        )

        params = qml.numpy.array(self.params, requires_grad=True)
        res = qml.grad(qnode)(params)
        assert qml.math.shape(res) == (2,)
        assert np.allclose(res, self.exp_grad, atol=0.01)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_jax(self, shots, use_jit, seed):
        """Test that the QNode executes and is differentiable with JAX. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", seed=seed)

        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="jax", diff_method=diff_method), shots=shots
        )
        if use_jit:
            qnode = jax.jit(qnode)

        params = jax.numpy.array(self.params)

        jac_fn = jax.jacobian(qnode)
        if use_jit:
            jac_fn = jax.jit(jac_fn)

        jac = jac_fn(params)
        assert jac.shape == (2,)
        assert np.allclose(jac, self.exp_grad, atol=0.02)

    @pytest.mark.torch
    @pytest.mark.parametrize("shots", [None, 50000])
    def test_qnode_torch(self, shots, seed):
        """Test that the QNode executes and is differentiable with Torch. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import torch

        dev = qml.device("default.qubit", seed=seed)

        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="torch", diff_method=diff_method), shots=shots
        )

        params = torch.tensor(self.params, requires_grad=True)
        jac = torch.autograd.functional.jacobian(qnode, params)
        assert qml.math.shape(jac) == (2,)
        assert qml.math.allclose(jac, self.exp_grad, atol=0.01)

    @pytest.mark.tf
    @pytest.mark.parametrize("shots", [None, 50000])
    @pytest.mark.xfail(reason="tf gradient doesn't seem to be working, returns ()")
    def test_qnode_tf(self, shots, seed):
        """Test that the QNode executes and is differentiable with TensorFlow. The shots
        argument controls whether autodiff or parameter-shift gradients are used."""
        import tensorflow as tf

        dev = qml.device("default.qubit", seed=seed)
        diff_method = "backprop" if shots is None else "parameter-shift"
        qnode = qml.set_shots(
            qml.QNode(self.circuit, dev, interface="tf", diff_method=diff_method), shots=shots
        )

        params = tf.Variable(self.params)
        with tf.GradientTape() as tape:
            res = qnode(params)

        jac = tape.gradient(res, params)
        assert qml.math.shape(jac) == (8,)
        assert qml.math.allclose(res, self.exp_grad, atol=0.001)


def test_correct_queueing():
    """Test that operations in a circuit containing AmplitudeAmplification are correctly queued"""
    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit1():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)

        qml.AmplitudeAmplification(generator(range(3)), oracle([0], range(3)))
        return qml.state()

    @qml.qnode(dev)
    def circuit2():
        generator(wires=[0, 1, 2])

        qml.AmplitudeAmplification(generator(range(3)), oracle([0], range(3)))
        return qml.state()

    U = generator(wires=[0, 1, 2])
    O = oracle([0], wires=[0, 1, 2])

    @qml.qnode(dev)
    def circuit3():
        generator(wires=[0, 1, 2])

        qml.AmplitudeAmplification(U=U, O=O)
        return qml.state()

    assert np.allclose(circuit1(), circuit2())
    assert np.allclose(circuit1(), circuit3())


def test_amplification():
    """Test that AmplitudeAmplification amplifies a marked element."""

    U = generator(wires=range(3))
    O = oracle([2], wires=range(3))

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit():
        generator(wires=range(3))
        qml.AmplitudeAmplification(U, O, iters=5, fixed_point=True, work_wire=3)

        return qml.probs(wires=range(3))

    res = np.round(circuit(), 3)

    expected = np.array([0.013, 0.013, 0.91, 0.013, 0.013, 0.013, 0.013, 0.013])

    assert np.allclose(res, expected)


@pytest.mark.parametrize(("p_min"), [0.7, 0.8, 0.9])
def test_p_min(p_min):
    """Test that the p_min parameter works correctly."""

    dev = qml.device("default.qubit")

    U = generator(wires=range(4))
    O = oracle([0], wires=range(4))

    @qml.qnode(dev)
    def circuit():
        generator(wires=range(4))

        qml.AmplitudeAmplification(U, O, fixed_point=True, work_wire=4, p_min=p_min, iters=11)

        return qml.probs(wires=range(4))

    assert circuit()[0] >= p_min


@pytest.mark.parametrize(
    "iters, p_min",
    (
        (4, 0.8),
        (5, 0.9),
        (6, 0.95),
    ),
)
def test_fixed_point_angles_function(iters, p_min):
    """Test that the _get_fixed_point_angles function works correctly."""

    alphas, betas = _get_fixed_point_angles(iters, p_min)

    assert np.all(alphas[:-1] > alphas[1:])
    assert np.all(betas[:-1] > betas[1:])

    assert np.allclose(betas, np.array([-alpha for alpha in reversed(alphas)]))

    assert all(isinstance(x, float) for x in alphas)
    assert all(isinstance(x, float) for x in betas)


@pytest.mark.parametrize(
    "n_wires, items, iters, fixed",
    (
        (3, [0, 2], 1, True),
        (3, [1, 2], 2, False),
        (5, [4, 5, 7, 12], 3, True),
        (5, [0, 1, 2, 3, 4], 4, False),
    ),
)
def test_decomposition_new(n_wires, items, iters, fixed):
    """Tests the decomposition rule implemented with the new system."""
    U = generator(wires=range(n_wires))
    O = oracle(items, wires=range(n_wires))

    op = qml.AmplitudeAmplification(U, O, iters, work_wire=n_wires, fixed_point=fixed)

    for rule in qml.list_decomps(qml.AmplitudeAmplification):
        _test_decomposition_rule(op, rule)
