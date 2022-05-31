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

import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class RX_broadcasted(qml.RX):
    """A version of qml.RX that detects batching."""

    ndim_params = (0,)
    compute_decomposition = staticmethod(lambda theta, wires=None: qml.RX(theta, wires=wires))


class RZ_broadcasted(qml.RZ):
    """A version of qml.RZ that detects batching."""

    ndim_params = (0,)
    compute_decomposition = staticmethod(lambda theta, wires=None: qml.RZ(theta, wires=wires))


dev = qml.device("default.qubit", wires=2)
"""Defines the device used for all tests"""


def make_tape(x, y, z, obs):
    """Construct a tape with three parametrized, two unparametrized
    operations and expvals of provided observables."""
    with qml.tape.QuantumTape() as tape:
        RX_broadcasted(x, wires=0)
        qml.PauliY(0)
        RX_broadcasted(y, wires=1)
        RZ_broadcasted(z, wires=1)
        qml.Hadamard(1)
        for ob in obs:
            qml.expval(ob)

    return tape


parameters_and_size = [
    [(0.2, np.array([0.1, 0.8, 2.1]), -1.5), 3],
    [(0.2, np.array([0.1]), np.array([-0.3])), 1],
    [
        (
            0.2,
            pnp.array([0.1, 0.3], requires_grad=True),
            pnp.array([-0.3, 2.1], requires_grad=False),
        ),
        2,
    ],
]

coeffs0 = [0.3, -5.1]
H0 = qml.Hamiltonian(qml.math.array(coeffs0), [qml.PauliZ(0), qml.PauliY(1)])

# Here we exploit the product structure of our circuit
exp_fn_Z0 = lambda x, y, z: -qml.math.cos(x) * qml.math.ones_like(y) * qml.math.ones_like(z)
exp_fn_Y1 = lambda x, y, z: qml.math.sin(y) * qml.math.cos(z) * qml.math.ones_like(x)
exp_fn_Z0Y1 = lambda x, y, z: exp_fn_Z0(x, y, z) * exp_fn_Y1(x, y, z)
exp_fn_Z0_and_Y1 = lambda x, y, z: qml.math.transpose(
    qml.math.array(
        [exp_fn_Z0(x, y, z), exp_fn_Y1(x, y, z)],
        like=exp_fn_Z0(x, y, z) + exp_fn_Y1(x, y, z),
    )
)
exp_fn_H0 = lambda x, y, z: exp_fn_Z0(x, y, z) * coeffs0[0] + exp_fn_Y1(x, y, z) * coeffs0[1]

observables_and_exp_fns = [
    ([qml.PauliZ(0)], exp_fn_Z0),
    ([qml.PauliZ(0) @ qml.PauliY(1)], exp_fn_Z0Y1),
    ([qml.PauliZ(0), qml.PauliY(1)], exp_fn_Z0_and_Y1),
    ([H0], exp_fn_H0),
]


class TestBroadcastExpand:
    """Tests for the broadcast_expand transform"""

    @pytest.mark.parametrize("params, size", parameters_and_size)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    def test_expansion(self, params, size, obs, exp_fn):
        """Test that the expansion works as expected."""
        tape = make_tape(*params, obs)
        assert tape.batch_size == size

        tapes, fn = qml.transforms.broadcast_expand(tape)
        assert len(tapes) == size
        assert all(_tape.batch_size is None for _tape in tapes)

        result = fn(qml.execute(tapes, dev, None))
        assert isinstance(result, np.ndarray)
        assert qml.math.allclose(result, exp_fn(*params))

    def test_without_broadcasting(self):
        tape = make_tape(0.2, 0.1, 0.5, [qml.PauliZ(0)])
        with pytest.raises(ValueError, match="The provided tape is not broadcasted."):
            qml.transforms.broadcast_expand(tape)

    @pytest.mark.autograd
    @pytest.mark.filterwarnings("ignore:Output seems independent of input")
    @pytest.mark.parametrize("params, size", parameters_and_size)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    def test_autograd(self, params, size, obs, exp_fn):
        """Test that the expansion works with autograd and is differentiable."""
        params = tuple(pnp.array(p, requires_grad=True) for p in params)

        def cost(*params):
            tape = make_tape(*params, obs)
            tapes, fn = qml.transforms.broadcast_expand(tape)
            return fn(qml.execute(tapes, dev, qml.gradients.param_shift))

        assert qml.math.allclose(cost(*params), exp_fn(*params))

        jac = qml.jacobian(cost)(*params)
        exp_jac = qml.jacobian(exp_fn)(*params)

        assert all(qml.math.allclose(_jac, e_jac) for _jac, e_jac in zip(jac, exp_jac))

    @pytest.mark.jax
    @pytest.mark.parametrize("params, size", parameters_and_size)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    def test_jax(self, params, size, obs, exp_fn):
        """Test that the expansion works with jax and is differentiable."""
        import jax

        params = tuple(jax.numpy.array(p) for p in params)

        def cost(*params):
            tape = make_tape(*params, obs)
            tapes, fn = qml.transforms.broadcast_expand(tape)
            return fn(qml.execute(tapes, dev, qml.gradients.param_shift, interface="jax"))

        assert qml.math.allclose(cost(*params), exp_fn(*params))

        jac = jax.jacobian(cost, argnums=[0, 1, 2])(*params)
        exp_jac = jax.jacobian(exp_fn, argnums=[0, 1, 2])(*params)

        assert all(qml.math.allclose(_jac, e_jac) for _jac, e_jac in zip(jac, exp_jac))

    @pytest.mark.tf
    @pytest.mark.parametrize("params, size", parameters_and_size)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    def test_tf(self, params, size, obs, exp_fn):
        """Test that the expansion works with TensorFlow and is differentiable."""
        import tensorflow as tf

        params = tuple(tf.Variable(p, dtype=tf.float64) for p in params)

        def cost(*params):
            tape = make_tape(*params, obs)
            tapes, fn = qml.transforms.broadcast_expand(tape)
            return fn(qml.execute(tapes, dev, qml.gradients.param_shift, interface="tf"))

        with tf.GradientTape(persistent=True) as t:
            out = cost(*params)
            exp = exp_fn(*params)

        jac = t.jacobian(out, params)
        exp_jac = t.jacobian(exp, params)

        assert qml.math.allclose(out, exp)
        for _jac, e_jac in zip(jac, exp_jac):
            if e_jac is None:
                assert qml.math.allclose(_jac, 0.0)
            else:
                assert qml.math.allclose(_jac, e_jac)

    @pytest.mark.torch
    @pytest.mark.filterwarnings("ignore:Output seems independent of input")
    @pytest.mark.parametrize("params, size", parameters_and_size)
    @pytest.mark.parametrize("obs, exp_fn", observables_and_exp_fns)
    def test_torch(self, params, size, obs, exp_fn):
        """Test that the expansion works with torch and is differentiable."""
        import torch

        torch_params = tuple(torch.tensor(p, requires_grad=True) for p in params)
        params = tuple(pnp.array(p, requires_grad=True) for p in params)

        def cost(*params):
            tape = make_tape(*params, obs)
            tapes, fn = qml.transforms.broadcast_expand(tape)
            return fn(qml.execute(tapes, dev, qml.gradients.param_shift, interface="torch"))

        assert qml.math.allclose(cost(*torch_params), exp_fn(*params))

        jac = torch.autograd.functional.jacobian(cost, torch_params)
        exp_jac = qml.jacobian(exp_fn)(*params)

        assert all(qml.math.allclose(_jac, e_jac) for _jac, e_jac in zip(jac, exp_jac))
