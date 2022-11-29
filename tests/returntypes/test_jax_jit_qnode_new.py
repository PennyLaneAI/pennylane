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
"""Integration tests for using the jax interface and its jittable variant with
a QNode"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import qnode
from pennylane.tape import QuantumTape

jit_qubit_device_and_diff_method = [
    ["default.qubit", "backprop", "forward"],
    # Jit
    ["default.qubit", "finite-diff", "backward"],
    ["default.qubit", "parameter-shift", "backward"],
    ["default.qubit", "adjoint", "forward"],
    ["default.qubit", "adjoint", "backward"],
]

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
jnp = jax.numpy

from jax.config import config

config.update("jax_enable_x64", True)

jacobian_fn = [jax.jacobian, jax.jacrev, jax.jacfwd]


@pytest.mark.parametrize("dev_name,diff_method,mode", jit_qubit_device_and_diff_method)
@pytest.mark.parametrize("jacobian", jacobian_fn)
class TestJIT:
    """Test JAX JIT integration with the QNode and automatic resolution of the
    correct JAX interface variant."""

    def test_gradient(self, dev_name, diff_method, mode, jacobian, tol):
        """Test derivative calculation of a scalar valued QNode"""
        dev = qml.device(dev_name, wires=1)

        if diff_method == "adjoint":
            pytest.xfail(reason="The adjoint method is not using host-callback currently")

        @jax.jit
        @qnode(dev, diff_method=diff_method, interface="jax-jit", mode=mode)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = jnp.array([1.0, 2.0])
        res = circuit(x)
        g = jacobian(circuit)(x)

        a, b = x

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

    @pytest.mark.filterwarnings(
        "ignore:Requested adjoint differentiation to be computed with finite shots."
    )
    @pytest.mark.parametrize("shots", [10, 1000])
    def test_hermitian(self, dev_name, diff_method, mode, shots, jacobian):
        """Test that the jax device works with qml.Hermitian and jitting even
        when shots>0.

        Note: before a fix, the cases of shots=10 and shots=1000 were failing due
        to different reasons, hence the parametrization in the test.
        """
        dev = qml.device(dev_name, wires=2, shots=shots)

        if diff_method == "backprop":
            pytest.skip("Backpropagation is unsupported if shots > 0.")

        if diff_method == "adjoint" and mode == "forward":
            pytest.skip("Computing the gradient for Hermitian is not supported with adjoint.")

        projector = np.array(qml.matrix(qml.PauliZ(0) @ qml.PauliZ(1)))

        @jax.jit
        @qml.qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circ(projector):
            return qml.expval(qml.Hermitian(projector, wires=range(2)))

        assert jnp.allclose(circ(projector), 1)

    @pytest.mark.filterwarnings(
        "ignore:Requested adjoint differentiation to be computed with finite shots."
    )
    @pytest.mark.parametrize("shots", [10, 1000])
    def test_probs_obs_none(self, dev_name, diff_method, mode, shots, jacobian):
        """Test that the jax device works with qml.probs, a MeasurementProcess
        that has obs=None even when shots>0."""
        dev = qml.device(dev_name, wires=2, shots=shots)

        if diff_method == "backprop":
            pytest.skip("Backpropagation is unsupported if shots > 0.")

        @qml.qnode(dev, interface="jax", diff_method="parameter-shift")
        def circuit():
            return qml.probs(wires=0)

        assert jnp.allclose(circuit(), jnp.array([1.0, 0.0]))

    @pytest.mark.xfail(
        reason="Non-trainable parameters are not being correctly unwrapped by the interface"
    )
    def test_gradient_subset(self, dev_name, diff_method, mode, jacobian, tol):
        """Test derivative calculation of a scalar valued QNode with respect
        to a subset of arguments"""
        a = jnp.array(0.1)
        b = jnp.array(0.2)

        dev = qml.device(dev_name, wires=1)

        @jax.jit
        @qnode(dev, diff_method=diff_method, interface="jax", mode=mode)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.RZ(c, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = jacobian(circuit, argnums=[0, 1])(a, b, 0.0)

        expected_res = np.cos(a) * np.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

    def test_gradient_scalar_cost_vector_valued_qnode(
        self, dev_name, diff_method, mode, jacobian, tol
    ):
        """Test derivative calculation of a scalar valued cost function that
        uses the output of a vector-valued QNode"""
        dev = qml.device(dev_name, wires=2)

        if diff_method == "adjoint":
            pytest.xfail(reason="The adjoint method is not using host-callback currently")

        @jax.jit
        @qnode(dev, diff_method=diff_method, interface="jax", mode=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        def cost(x, y, idx):
            res = circuit(x, y)
            return res[idx]

        x = jnp.array(1.0)
        y = jnp.array(2.0)
        expected_g = (
            np.array([-np.sin(x) * np.cos(y) / 2, np.cos(y) * np.sin(x) / 2]),
            np.array([-np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2]),
        )

        idx = 0
        g0 = jacobian(cost, argnums=0)(x, y, idx)
        g1 = jacobian(cost, argnums=1)(x, y, idx)
        assert np.allclose(g0, expected_g[0][idx], atol=tol, rtol=0)
        assert np.allclose(g1, expected_g[1][idx], atol=tol, rtol=0)

        idx = 1
        g0 = jacobian(cost, argnums=0)(x, y, idx)
        g1 = jacobian(cost, argnums=1)(x, y, idx)

        assert np.allclose(g0, expected_g[0][idx], atol=tol, rtol=0)
        assert np.allclose(g1, expected_g[1][idx], atol=tol, rtol=0)


qubit_device_and_diff_method_and_mode = [
    ["default.qubit", "backprop", "forward"],
    ["default.qubit", "finite-diff", "backward"],
    ["default.qubit", "parameter-shift", "backward"],
    ["default.qubit", "adjoint", "forward"],
    ["default.qubit", "adjoint", "backward"],
]


@pytest.mark.parametrize("dev_name,diff_method,mode", qubit_device_and_diff_method_and_mode)
@pytest.mark.parametrize("shots", [None, 10000])
@pytest.mark.parametrize("jacobian", jacobian_fn)
class TestReturn:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types."""

    def test_grad_single_measurement_param(self, dev_name, diff_method, mode, jacobian, shots):
        """For one measurement and one param, the gradient is a float."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=1, shots=shots)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)

        grad = jacobian(circuit)(a)

        assert isinstance(grad, jax.numpy.ndarray)
        assert grad.shape == ()

    def test_grad_single_measurement_multiple_param(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=1, shots=shots)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        grad = jacobian(circuit, argnums=[0, 1])(a, b)

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        assert grad[0].shape == ()
        assert grad[1].shape == ()

    def test_grad_single_measurement_multiple_param_array(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=1, shots=shots)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        grad = jacobian(circuit)(a)

        assert isinstance(grad, jax.numpy.ndarray)
        assert grad.shape == (2,)

    def test_jacobian_single_measurement_param_probs(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jacobian(circuit)(a)

        assert isinstance(jac, jax.numpy.ndarray)
        assert jac.shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)
        b = jax.numpy.array(0.2)

        jac = jacobian(circuit, argnums=[0, 1])(a, b)

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = jax.numpy.array([0.1, 0.2])
        jac = jacobian(circuit)(a)

        assert isinstance(jac, jax.numpy.ndarray)
        assert jac.shape == (4, 2)

    def test_jacobian_expval_expval_multiple_params(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """The jacobian of multiple measurements with multiple params return a tuple of arrays."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        dev = qml.device(dev_name, wires=2, shots=shots)

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        jac = jacobian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == ()
        assert isinstance(jac[0][1], jax.numpy.ndarray)
        assert jac[0][1].shape == ()

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == ()
        assert isinstance(jac[1][1], jax.numpy.ndarray)
        assert jac[1][1].shape == ()

    def test_jacobian_expval_expval_multiple_params_array(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        dev = qml.device(dev_name, wires=2, shots=shots)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (2,)

    def test_jacobian_var_var_multiple_params(self, dev_name, diff_method, mode, jacobian, shots):
        """The jacobian of multiple measurements with multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        par_0 = jax.numpy.array(0.1)
        par_1 = jax.numpy.array(0.2)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        jac = jacobian(circuit, argnums=[0, 1])(par_0, par_1)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == ()
        assert isinstance(jac[0][1], jax.numpy.ndarray)
        assert jac[0][1].shape == ()

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == ()
        assert isinstance(jac[1][1], jax.numpy.ndarray)
        assert jac[1][1].shape == ()

    def test_jacobian_var_var_multiple_params_array(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of var.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.var(qml.PauliZ(0))

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (2,)

    def test_jacobian_multiple_measurement_single_param(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """The jacobian of multiple measurements with a single params return an array."""
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")
        dev = qml.device(dev_name, wires=2, shots=shots)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array(0.1)

        jac = jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == ()

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4,)

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = np.array(0.1, requires_grad=True)
        b = np.array(0.2, requires_grad=True)

        jac = jacobian(circuit, argnums=[0, 1])(a, b)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tuple)
        assert len(jac[0]) == 2
        assert isinstance(jac[0][0], jax.numpy.ndarray)
        assert jac[0][0].shape == ()
        assert isinstance(jac[0][1], jax.numpy.ndarray)
        assert jac[0][1].shape == ()

        assert isinstance(jac[1], tuple)
        assert len(jac[1]) == 2
        assert isinstance(jac[1][0], jax.numpy.ndarray)
        assert jac[1][0].shape == (4,)
        assert isinstance(jac[1][1], jax.numpy.ndarray)
        assert jac[1][1].shape == (4,)

    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev_name, diff_method, mode, jacobian, shots
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")
        if shots is not None and diff_method in ("backprop", "adjoint"):
            pytest.skip("Test does not support finite shots and adjoint/backprop")

        dev = qml.device(dev_name, wires=2, shots=shots)

        @jax.jit
        @qnode(dev, interface="jax", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = jax.numpy.array([0.1, 0.2])

        jac = jacobian(circuit)(a)

        assert isinstance(jac, tuple)
        assert len(jac) == 2  # measurements

        assert isinstance(jac[0], jax.numpy.ndarray)
        assert jac[0].shape == (2,)

        assert isinstance(jac[1], jax.numpy.ndarray)
        assert jac[1].shape == (4, 2)
