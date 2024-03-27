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
Unit tests for the ``broadcast_expand`` transform.
"""
# pylint: disable=too-few-public-methods
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


dev = qml.device("default.qubit", wires=2)
"""Defines the device used for all tests"""


def make_ops(x, y, z):
    """Queue three parametrized and two unparametrized operations and return them."""
    ops = [
        qml.RX(x, wires=0),
        qml.PauliY(0),
        qml.RX(y, wires=1),
        qml.RZ(z, wires=1),
        qml.Hadamard(1),
    ]
    return ops


parameters = [
    (0.2, np.array([0.1, 0.8, 2.1]), -1.5),
    (0.2, np.array([0.1]), np.array([-0.3])),
    (
        0.2,
        pnp.array([0.1, 0.3], requires_grad=True),
        pnp.array([-0.3, 2.1], requires_grad=False),
    ),
]
sizes = [3, 1, 2]

coeffs0 = [0.3, -5.1]
H0 = qml.Hamiltonian(qml.math.array(coeffs0), [qml.PauliZ(0), qml.PauliY(1)])


# Here we exploit the product structure of our circuit
def exp_fn_Z0(x, y, z):
    out = -qml.math.cos(x) * qml.math.ones_like(y) * qml.math.ones_like(z)
    return out[0] if len(out) == 1 else out


def exp_fn_Y1(x, y, z):
    out = qml.math.sin(y) * qml.math.cos(z) * qml.math.ones_like(x)
    return out[0] if len(out) == 1 else out


exp_fn_Z0Y1 = lambda x, y, z: exp_fn_Z0(x, y, z) * exp_fn_Y1(x, y, z)
exp_fn_Z0_and_Y1 = lambda x, y, z: qml.math.stack([exp_fn_Z0(x, y, z), exp_fn_Y1(x, y, z)])
exp_fn_H0 = lambda x, y, z: exp_fn_Z0(x, y, z) * coeffs0[0] + exp_fn_Y1(x, y, z) * coeffs0[1]

observables_and_exp_fns = [
    ([qml.PauliZ(0)], exp_fn_Z0),
    ([qml.PauliZ(0) @ qml.PauliY(1)], exp_fn_Z0Y1),
    ([qml.PauliZ(0), qml.PauliY(1)], exp_fn_Z0_and_Y1),
    ([H0], exp_fn_H0),
]


class TestBroadcastExpand:
    """Tests for the broadcast_expand transform"""

    @pytest.mark.parametrize("params, size", list(zip(parameters, sizes)))
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    def test_expansion(self, params, size, obs, exp_fn):
        """Test that the expansion works as expected."""
        ops = make_ops(*params)
        expvals = [qml.expval(ob) for ob in obs]
        tape = qml.tape.QuantumScript(ops, expvals)
        assert tape.batch_size == size

        tapes, fn = qml.transforms.broadcast_expand(tape)
        assert len(tapes) == size
        assert all(_tape.batch_size is None for _tape in tapes)

        result = fn(qml.execute(tapes, dev, None))
        expected = exp_fn(*params)

        assert qml.math.allclose(result, expected)

    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    def test_expansion_qnode(self, params, obs, exp_fn):
        """Test that the transform integrates correctly with the transform program"""

        @qml.transforms.broadcast_expand
        @qml.qnode(dev)
        def circuit(x, y, z, obs):
            qml.StatePrep(np.array([1, 0, 0, 0]), wires=[0, 1])
            _ = make_ops(x, y, z)
            return [qml.expval(ob) for ob in obs]

        result = circuit(*params, obs)
        expected = exp_fn(*params)

        assert qml.math.allclose(result, expected)

    def test_state_prep(self):
        """Test that expansion works for state preparations"""
        ops = [qml.CNOT([0, 1])]
        meas = [qml.expval(qml.PauliZ(1))]
        prep = [qml.StatePrep(np.eye(4), wires=[0, 1])]
        tape = qml.tape.QuantumScript(prep + ops, meas)

        tapes, fn = qml.transforms.broadcast_expand(tape)
        assert len(tapes) == 4
        assert all(t.batch_size is None for t in tapes)

        result = fn(qml.execute(tapes, dev, None))
        expected = np.array([1, -1, -1, 1])

        assert qml.math.allclose(result, expected)

    def test_not_copied(self):
        """Test that unbroadcasted operators are not copied"""
        x = np.array([0.5, 0.7, 0.9])
        y = np.array(1.5)

        ops = [qml.RX(x, wires=0), qml.RY(y, wires=0)]
        meas = [qml.expval(qml.PauliZ(0))]
        tape = qml.tape.QuantumScript(ops, meas)

        tapes = qml.transforms.broadcast_expand(tape)[0]
        assert len(tapes) == 3
        assert all(t.batch_size is None for t in tapes)

        for t in tapes:
            # different instance of RX
            assert t.operations[0] is not tape.operations[0]

            # same instance of RY
            assert t.operations[1] is tape.operations[1]

    @pytest.mark.autograd
    @pytest.mark.filterwarnings("ignore:Output seems independent of input")
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    def test_autograd(self, params, obs, exp_fn, diff_method):
        """Test that the expansion works with autograd and is differentiable."""
        params = tuple(pnp.array(p, requires_grad=True) for p in params)
        diff_method = "parameter-shift"

        @qml.transforms.broadcast_expand
        @qml.qnode(dev, interface="autograd", diff_method=diff_method)
        def cost(*params):
            make_ops(*params)
            return qml.math.stack([qml.expval(ob) for ob in obs])

        expected = exp_fn(*params)

        assert qml.math.allclose(cost(*params), expected)

        jac = qml.jacobian(cost)(*params)
        exp_jac = qml.jacobian(exp_fn)(*params)

        assert all(qml.math.allclose(_jac, e_jac) for _jac, e_jac in zip(jac, exp_jac))

    @pytest.mark.jax
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    @pytest.mark.parametrize("use_jit", [True, False])
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    def test_jax(self, params, obs, exp_fn, use_jit, diff_method):
        """Test that the expansion works with jax and is differentiable."""
        # pylint: disable=too-many-arguments
        import jax

        jax.config.update("jax_enable_x64", True)

        params = tuple(jax.numpy.array(p) for p in params)
        diff_method = "parameter-shift"

        @qml.transforms.broadcast_expand
        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def cost(*params):
            make_ops(*params)
            return tuple(qml.expval(ob) for ob in obs)

        if use_jit:
            cost = jax.jit(cost)

        expected = exp_fn(*params)

        assert qml.math.allclose(cost(*params), expected)

        jac = jax.jacobian(cost, argnums=[0, 1, 2])(*params)

        exp_jac = jax.jacobian(exp_fn, argnums=[0, 1, 2])(*params)

        if len(obs) > 1:
            assert all(qml.math.allclose(_jac, e_jac) for _jac, e_jac in zip(jac[0], exp_jac[0]))
            assert all(qml.math.allclose(_jac, e_jac) for _jac, e_jac in zip(jac[1], exp_jac[1]))
        else:
            assert all(qml.math.allclose(_jac, e_jac) for _jac, e_jac in zip(jac, exp_jac))

    @pytest.mark.slow
    @pytest.mark.tf
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    def test_tf(self, params, obs, exp_fn):
        """Test that the expansion works with TensorFlow and is differentiable."""
        import tensorflow as tf

        params = tuple(tf.Variable(p, dtype=tf.float64) for p in params)
        diff_method = "parameter-shift"

        @qml.transforms.broadcast_expand
        @qml.qnode(dev, interface="tensorflow", diff_method=diff_method)
        def cost(*params):
            make_ops(*params)
            return tuple(qml.expval(ob) for ob in obs)

        with tf.GradientTape(persistent=True) as t:
            out = tf.stack(cost(*params))
            exp = exp_fn(*params)

        jac = t.jacobian(out, params)
        exp_jac = t.jacobian(exp, params)

        for _jac, e_jac in zip(jac, exp_jac):
            if e_jac is None:
                assert qml.math.allclose(_jac, 0.0)
            else:
                assert qml.math.allclose(_jac, e_jac)

    @pytest.mark.torch
    @pytest.mark.filterwarnings("ignore:Output seems independent of input")
    @pytest.mark.parametrize("params", parameters)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    def test_torch(self, params, obs, exp_fn, diff_method):
        """Test that the expansion works with torch and is differentiable."""
        import torch

        torch_params = tuple(
            torch.tensor(p, requires_grad=True, dtype=torch.float64) for p in params
        )
        params = tuple(pnp.array(p, requires_grad=True) for p in params)
        diff_method = "parameter-shift"

        @qml.transforms.broadcast_expand
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def cost(*params):
            make_ops(*params)
            return tuple(qml.expval(ob) for ob in obs)

        res = cost(*torch_params)
        jac = torch.autograd.functional.jacobian(cost, torch_params)
        exp_jac = torch.autograd.functional.jacobian(exp_fn, torch_params)

        if len(obs) > 1:
            assert all(qml.math.allclose(r, e) for r, e in zip(res, exp_fn(*params)))
            # Need to perform a transpose because the broadcast_expand transform pulls out the
            # broadcasting axis, which the `exp_fn` does not do.
            jac = tuple(
                qml.math.stack([jac[i][j] for i in range(len(obs))]) for j in range(len(params))
            )
        else:
            assert qml.math.allclose(res, exp_fn(*params))

        assert all(qml.math.allclose(_jac, e_jac) for _jac, e_jac in zip(jac, exp_jac))
