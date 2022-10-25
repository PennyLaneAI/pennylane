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

jax = pytest.importorskip("jax", minversion="0.2")
jnp = jax.numpy
import numpy as np
from jax.config import config

import pennylane as qml
from pennylane import DeviceError
from pennylane.devices.default_qubit_jax import DefaultQubitJax


@pytest.mark.jax
class TestQNodeIntegration:
    """Integration tests for default.qubit.jax. This test ensures it integrates
    properly with the PennyLane UI, in particular the new QNode."""

    def test_defines_correct_capabilities(self):
        """Test that the device defines the right capabilities"""

        dev = qml.device("default.qubit.jax", wires=1)
        cap = dev.capabilities()
        capabilities = {
            "model": "qubit",
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_probs": True,
            "returns_state": True,
            "supports_inverse_operations": True,
            "supports_analytic_computation": True,
            "supports_broadcasting": True,
            "passthru_interface": "jax",
            "passthru_devices": {
                "torch": "default.qubit.torch",
                "tf": "default.qubit.tf",
                "autograd": "default.qubit.autograd",
                "jax": "default.qubit.jax",
            },
        }
        assert cap == capabilities

    def test_defines_correct_capabilities_directly_from_class(self):
        """Test that the device defines the right capabilities"""

        dev = DefaultQubitJax(wires=1)
        cap = dev.capabilities()
        assert cap["passthru_interface"] == "jax"

    def test_load_device(self):
        """Test that the plugin device loads correctly"""
        dev = qml.device("default.qubit.jax", wires=2)
        assert dev.num_wires == 2
        assert dev.shots == None
        assert dev.short_name == "default.qubit.jax"
        assert dev.capabilities()["passthru_interface"] == "jax"

    @pytest.mark.parametrize(
        "jax_enable_x64, c_dtype, r_dtype",
        ([True, np.complex128, np.float64], [False, np.complex64, np.float32]),
    )
    def test_float_precision(self, jax_enable_x64, c_dtype, r_dtype):
        """Test that the plugin device uses the same float precision as the jax config."""
        config.update("jax_enable_x64", jax_enable_x64)
        dev = qml.device("default.qubit.jax", wires=2)
        assert dev.state.dtype == c_dtype
        assert dev.state.real.dtype == r_dtype

    def test_qubit_circuit(self, tol):
        """Test that the device provides the correct
        result for a simple circuit."""
        p = jnp.array(0.543)

        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -jnp.sin(p)
        assert jnp.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_qubit_circuit_with_jit(self, tol):
        """Test that the device provides the correct
        result for a simple circuit under a jax.jit."""
        p = jnp.array(0.543)

        dev = qml.device("default.qubit.jax", wires=1)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -jnp.sin(p)
        # Do not test isinstance here since the @jax.jit changes the function
        # type. Just test that it works and spits our the right value.
        assert jnp.isclose(circuit(p), expected, atol=tol, rtol=0)

        # Test with broadcasted parameters
        p = jnp.array([0.543, 0.21, 1.5])
        expected = -jnp.sin(p)
        assert jnp.allclose(circuit(p), expected, atol=tol, rtol=0)

    def test_qubit_circuit_broadcasted(self, tol):
        """Test that the device provides the correct
        result for a simple broadcasted circuit."""
        p = jnp.array([0.543, 0.21, 1.5])

        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -jnp.sin(p)

        assert jnp.allclose(circuit(p), expected, atol=tol, rtol=0)

    def test_correct_state(self, tol):
        """Test that the device state is correct after applying a
        quantum function on the device"""

        dev = qml.device("default.qubit.jax", wires=2)

        state = dev.state
        expected = jnp.array([1, 0, 0, 0])
        assert jnp.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit():
            qml.Hadamard(wires=0)
            qml.RZ(jnp.pi / 4, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev.state

        amplitude = jnp.exp(-1j * jnp.pi / 8) / jnp.sqrt(2)

        expected = jnp.array([amplitude, 0, jnp.conj(amplitude), 0])
        assert jnp.allclose(state, expected, atol=tol, rtol=0)

    def test_correct_state_broadcasted(self, tol):
        """Test that the device state is correct after applying a
        broadcasted quantum function on the device"""

        dev = qml.device("default.qubit.jax", wires=2)

        state = dev.state
        expected = jnp.array([1, 0, 0, 0])
        assert jnp.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit():
            qml.Hadamard(wires=0)
            qml.RZ(jnp.array([np.pi / 4, np.pi / 2]), wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev.state

        phase = jnp.exp(-1j * jnp.pi / 8)

        expected = np.array(
            [
                [phase / jnp.sqrt(2), 0, jnp.conj(phase) / jnp.sqrt(2), 0],
                [phase**2 / jnp.sqrt(2), 0, jnp.conj(phase) ** 2 / jnp.sqrt(2), 0],
            ]
        )
        assert jnp.allclose(state, expected, atol=tol, rtol=0)

    def test_correct_state_returned(self, tol):
        """Test that the device state is correct after applying a
        quantum function on the device"""
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit():
            qml.Hadamard(wires=0)
            qml.RZ(jnp.pi / 4, wires=0)
            return qml.state()

        state = circuit()

        amplitude = jnp.exp(-1j * jnp.pi / 8) / jnp.sqrt(2)

        expected = jnp.array([amplitude, 0, jnp.conj(amplitude), 0])
        assert jnp.allclose(state, expected, atol=tol, rtol=0)

    def test_correct_state_returned_broadcasted(self, tol):
        """Test that the device state is correct after applying a
        broadcasted quantum function on the device"""
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit():
            qml.Hadamard(wires=0)
            qml.RZ(jnp.array([np.pi / 4, np.pi / 2]), wires=0)
            return qml.state()

        state = circuit()

        phase = jnp.exp(-1j * jnp.pi / 8)

        expected = np.array(
            [
                [phase / jnp.sqrt(2), 0, jnp.conj(phase) / jnp.sqrt(2), 0],
                [phase**2 / jnp.sqrt(2), 0, jnp.conj(phase) ** 2 / jnp.sqrt(2), 0],
            ]
        )
        assert jnp.allclose(state, expected, atol=tol, rtol=0)

    def test_probs_jax(self, tol):
        """Test that returning probs works with jax"""
        dev = qml.device("default.qubit.jax", wires=1, shots=100)
        expected = jnp.array([0.0, 1.0])

        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            qml.PauliX(wires=0)
            return qml.probs(wires=0)

        result = circuit()
        assert jnp.allclose(result, expected, atol=tol)

    def test_probs_jax_broadcasted(self, tol):
        """Test that returning probs works with jax"""
        dev = qml.device("default.qubit.jax", wires=1, shots=100)
        expected = jnp.array([[0.0, 1.0]] * 3)

        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            qml.RX(jnp.zeros(3), 0)
            qml.PauliX(wires=0)
            return qml.probs(wires=0)

        result = circuit()
        assert jnp.allclose(result, expected, atol=tol)

    def test_probs_jax_jit(self, tol):
        """Test that returning probs works with jax and jit"""
        dev = qml.device("default.qubit.jax", wires=1, shots=100)
        expected = jnp.array([0.0, 1.0])

        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit(z):
            qml.RX(z, wires=0)
            qml.PauliX(wires=0)
            return qml.probs(wires=0)

        result = circuit(0.0)
        assert jnp.allclose(result, expected, atol=tol)

        # Test with broadcasting
        result = circuit(jnp.zeros(3))
        expected = jnp.array([[0.0, 1.0]] * 3)
        assert jnp.allclose(result, expected, atol=tol)

    def test_custom_shots_probs_jax_jit(self, tol):
        """Test that returning probs works with jax and jit when using custom shot vector"""
        dev = qml.device("default.qubit.jax", wires=1, shots=(2, 2))
        expected = jnp.array([[0.0, 1.0], [0.0, 1.0]])

        @jax.jit
        @qml.qnode(dev, diff_method=None, interface="jax")
        def circuit():
            qml.PauliX(wires=0)
            return qml.probs(wires=0)

        result = circuit()
        assert jnp.allclose(result, expected, atol=tol)

    @pytest.mark.skip("Shot lists are not supported with broadcasting yet")
    def test_custom_shots_probs_jax_jit_broadcasted(self, tol):
        """Test that returning probs works with jax and jit when
        using a custom shot vector and broadcasting"""
        dev = qml.device("default.qubit.jax", wires=1, shots=(2, 2))
        expected = jnp.array([[[0.0, 1.0], [0.0, 1.0]]] * 5)

        @jax.jit
        @qml.qnode(dev, diff_method=None, interface="jax")
        def circuit():
            qml.RX(jnp.zeros(5), 0)
            qml.PauliX(wires=0)
            return qml.probs(wires=0)

        result = circuit()
        assert jnp.allclose(result, expected, atol=tol)

    def test_sampling_with_jit(self):
        """Test that sampling works with a jax.jit"""

        @jax.jit
        def circuit(x, key):
            dev = qml.device("default.qubit.jax", wires=1, shots=1000, prng_key=key)

            @qml.qnode(dev, interface="jax", diff_method=None)
            def inner_circuit():
                qml.RX(x, wires=0)
                qml.Hadamard(0)
                return qml.sample(qml.PauliZ(wires=0))

            return inner_circuit()

        a = circuit(0.0, jax.random.PRNGKey(0))
        b = circuit(0.0, jax.random.PRNGKey(0))
        c = circuit(0.0, jax.random.PRNGKey(1))
        np.testing.assert_array_equal(a, b)
        assert not np.all(a == c)

        # Test with broadcasting
        d = circuit(jnp.zeros(5), jax.random.PRNGKey(9))
        assert qml.math.shape(d) == (5, 1000)

    @pytest.mark.parametrize(
        "state_vector",
        [np.array([0.5 + 0.5j, 0.5 + 0.5j, 0, 0]), jnp.array([0.5 + 0.5j, 0.5 + 0.5j, 0, 0])],
    )
    def test_qubit_state_vector_arg_jax_jit(self, state_vector, tol):
        """Test that Qubit state vector as argument works with a jax.jit"""
        dev = qml.device("default.qubit.jax", wires=list(range(2)))

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            wires = list(range(2))
            qml.QubitStateVector(x, wires=wires)
            return [qml.expval(qml.PauliX(wires=i)) for i in wires]

        res = circuit(state_vector)
        assert jnp.allclose(jnp.array(res), jnp.array([0, 1]), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "state_vector",
        [np.array([0.5 + 0.5j, 0.5 + 0.5j, 0, 0]), jnp.array([0.5 + 0.5j, 0.5 + 0.5j, 0, 0])],
    )
    def test_qubit_state_vector_arg_jax(self, state_vector, tol):
        """Test that Qubit state vector as argument works with jax"""
        dev = qml.device("default.qubit.jax", wires=list(range(2)))

        @qml.qnode(dev, interface="jax")
        def circuit(x):
            wires = list(range(2))
            qml.QubitStateVector(x, wires=wires)
            return [qml.expval(qml.PauliX(wires=i)) for i in wires]

        res = circuit(state_vector)
        assert jnp.allclose(jnp.array(res), jnp.array([0, 1]), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "state_vector",
        [np.array([0.5 + 0.5j, 0.5 + 0.5j, 0, 0]), jnp.array([0.5 + 0.5j, 0.5 + 0.5j, 0, 0])],
    )
    def test_qubit_state_vector_jax_jit(self, state_vector, tol):
        """Test that Qubit state vector works with a jax.jit"""
        dev = qml.device("default.qubit.jax", wires=list(range(2)))

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.QubitStateVector(state_vector, wires=dev.wires)
            for w in dev.wires:
                qml.RZ(x, wires=w, id="x")
            return qml.expval(qml.PauliZ(wires=0))

        res = circuit(0.1)
        assert jnp.allclose(jnp.array(res), 1, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "state_vector",
        [np.array([0.5 + 0.5j, 0.5 + 0.5j, 0, 0]), jnp.array([0.5 + 0.5j, 0.5 + 0.5j, 0, 0])],
    )
    def test_qubit_state_vector_jax(self, state_vector, tol):
        """Test that Qubit state vector works with a jax"""
        dev = qml.device("default.qubit.jax", wires=list(range(2)))

        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.QubitStateVector(state_vector, wires=dev.wires)
            for w in dev.wires:
                qml.RZ(x, wires=w, id="x")
            return qml.expval(qml.PauliZ(wires=0))

        res = circuit(0.1)
        assert jnp.allclose(jnp.array(res), 1, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "state_vector",
        [np.array([0.1 + 0.1j, 0.2 + 0.2j, 0, 0]), jnp.array([0.1 + 0.1j, 0.2 + 0.2j, 0, 0])],
    )
    def test_qubit_state_vector_jax_not_normed(self, state_vector, tol):
        """Test that an error is raised when Qubit state vector is not normed works with a jax"""
        dev = qml.device("default.qubit.jax", wires=list(range(2)))

        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.QubitStateVector(state_vector, wires=dev.wires)
            for w in dev.wires:
                qml.RZ(x, wires=w, id="x")
            return qml.expval(qml.PauliZ(wires=0))

        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            circuit(0.1)

    def test_sampling_op_by_op(self):
        """Test that op-by-op sampling works as a new user would expect"""
        dev = qml.device("default.qubit.jax", wires=1, shots=1000)

        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            qml.Hadamard(0)
            return qml.sample(qml.PauliZ(wires=0))

        a = circuit()
        b = circuit()
        assert not np.all(a == b)

    def test_sampling_analytic_mode(self):
        """Test that when sampling with shots=None an error is raised."""
        dev = qml.device("default.qubit.jax", wires=1, shots=None)

        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            return qml.sample(qml.PauliZ(wires=0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The number of shots has to be explicitly set on the device "
            "when using sample-based measurements.",
        ):
            res = circuit()

    def test_sampling_analytic_mode_with_counts(self):
        """Test that when sampling with counts and shots=None an error is raised."""
        dev = qml.device("default.qubit.jax", wires=1, shots=None)

        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            return qml.counts(qml.PauliZ(wires=0))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The number of shots has to be explicitly set on the device "
            "when using sample-based measurements.",
        ):
            res = circuit()

    def test_gates_dont_crash(self):
        """Test for gates that weren't covered by other tests."""
        dev = qml.device("default.qubit.jax", wires=2, shots=1000)

        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            qml.CRZ(0.0, wires=[0, 1])
            qml.CRX(0.0, wires=[0, 1])
            qml.PhaseShift(0.0, wires=0)
            qml.ControlledPhaseShift(0.0, wires=[1, 0])
            qml.CRot(1.0, 0.0, 0.0, wires=[0, 1])
            qml.CRY(0.0, wires=[0, 1])
            return qml.sample(qml.PauliZ(wires=0))

        circuit()  # Just don't crash.

    def test_diagonal_doesnt_crash(self):
        """Test that diagonal gates can be used."""
        dev = qml.device("default.qubit.jax", wires=1, shots=1000)

        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            qml.DiagonalQubitUnitary(np.array([1.0, 1.0]), wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        circuit()  # Just don't crash.

    def test_broadcasted_diagonal_doesnt_crash(self):
        """Test that diagonal gates can be used."""
        dev = qml.device("default.qubit.jax", wires=1, shots=1000)

        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            qml.DiagonalQubitUnitary(np.array([[-1, -1], [1j, -1], [1.0, 1.0]]), wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        circuit()  # Just don't crash.


@pytest.mark.jax
class TestPassthruIntegration:
    """Tests for integration with the PassthruQNode"""

    @pytest.mark.parametrize("jacobian_transform", [jax.jacfwd, jax.jacrev])
    def test_jacobian_variable_multiply(self, tol, jacobian_transform):
        """Test that jacobian of a QNode with an attached default.qubit.jax device
        gives the correct result in the case of parameters multiplied by scalars"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        weights = jnp.array([x, y, z])

        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(weights)

        expected = jnp.cos(3 * x) * jnp.cos(y) * jnp.cos(z / 2) - jnp.sin(3 * x) * jnp.sin(z / 2)
        assert jnp.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = jacobian_transform(circuit, 0)
        res = grad_fn(jnp.array(weights))

        expected = jnp.array(
            [
                -3
                * (jnp.sin(3 * x) * jnp.cos(y) * jnp.cos(z / 2) + jnp.cos(3 * x) * jnp.sin(z / 2)),
                -jnp.cos(3 * x) * jnp.sin(y) * jnp.cos(z / 2),
                -0.5
                * (jnp.sin(3 * x) * jnp.cos(z / 2) + jnp.cos(3 * x) * jnp.cos(y) * jnp.sin(z / 2)),
            ]
        )

        assert jnp.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_variable_multiply_broadcasted(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.jax device
        gives the correct result in the case of broadcasted parameters multiplied by scalars"""
        x = jnp.array([0.43316321, 92.1, -0.5129])
        y = jnp.array([0.2162158, 0.241, -0.51])
        z = jnp.array([0.75110998, 0.12512, 9.12])
        weights = jnp.array([x, y, z])

        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.gradient_fn == "backprop"
        res = circuit(weights)

        expected = jnp.cos(3 * x) * jnp.cos(y) * jnp.cos(z / 2) - jnp.sin(3 * x) * jnp.sin(z / 2)
        assert jnp.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = jax.jacobian(circuit, 0)
        res = grad_fn(jnp.array(weights))

        expected = jnp.array(
            [
                -3
                * (jnp.sin(3 * x) * jnp.cos(y) * jnp.cos(z / 2) + jnp.cos(3 * x) * jnp.sin(z / 2)),
                -jnp.cos(3 * x) * jnp.sin(y) * jnp.cos(z / 2),
                -0.5
                * (jnp.sin(3 * x) * jnp.cos(z / 2) + jnp.cos(3 * x) * jnp.cos(y) * jnp.sin(z / 2)),
            ]
        )

        assert all(jnp.allclose(res[i, :, i], expected[:, i], atol=tol, rtol=0) for i in range(3))

    @pytest.mark.parametrize("jacobian_transform", [jax.jacfwd, jax.jacrev])
    def test_jacobian_repeated(self, tol, jacobian_transform):
        """Test that jacobian of a QNode with an attached default.qubit.jax device
        gives the correct result in the case of repeated parameters"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        p = jnp.array([x, y, z])
        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(p)

        expected = jnp.cos(y) ** 2 - jnp.sin(x) * jnp.sin(y) ** 2
        assert jnp.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = jacobian_transform(circuit, 0)
        res = grad_fn(p)

        expected = jnp.array(
            [-jnp.cos(x) * jnp.sin(y) ** 2, -2 * (jnp.sin(x) + 1) * jnp.sin(y) * jnp.cos(y), 0]
        )
        assert jnp.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_repeated_broadcasted(self, tol):
        """Test that jacobian of a QNode with an attached default.qubit.jax device
        gives the correct result in the case of repeated broadcasted parameters"""
        p = jnp.array([[0.433, 92.1, -0.512], [0.218, 0.241, -0.51], [0.71, 0.152, 9.12]])
        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        res = circuit(p)

        x, y, z = p
        expected = jnp.cos(y) ** 2 - jnp.sin(x) * jnp.sin(y) ** 2
        assert jnp.allclose(res, expected, atol=tol, rtol=0)

        grad_fn = jax.jacobian(circuit)
        res = grad_fn(p)

        expected = jnp.array(
            [
                -jnp.cos(x) * jnp.sin(y) ** 2,
                -2 * (jnp.sin(x) + 1) * jnp.sin(y) * jnp.cos(y),
                jnp.zeros_like(x),
            ]
        )
        assert all(jnp.allclose(res[i, :, i], expected[:, i], atol=tol, rtol=0) for i in range(3))

    @pytest.mark.parametrize("wires", [[0], ["abc"]])
    def test_state_differentiability(self, wires, tol):
        """Test that the device state can be differentiated"""
        dev = qml.device("default.qubit.jax", wires=wires)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a):
            qml.RY(a, wires=wires[0])
            return qml.state()

        a = jnp.array(0.54)

        def cost(a):
            """A function of the device quantum state, as a function
            of input QNode parameters."""
            res = jnp.abs(circuit(a)) ** 2
            return res[1] - res[0]

        grad = jax.grad(cost)(a)
        expected = jnp.sin(a)
        assert jnp.allclose(grad, expected, atol=tol, rtol=0)

    def test_state_differentiability_broadcasted(self, tol):
        """Test that the broadcasted device state can be differentiated"""
        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a):
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = jnp.array([0.54, 0.32, 1.2])

        def cost(a):
            """A function of the device quantum state, as a function
            of input QNode parameters."""
            circuit(a)
            res = jnp.abs(dev.state) ** 2
            return res[:, 1] - res[:, 0]

        jac = jax.jacobian(cost)(a)
        expected = jnp.diag(jnp.sin(a))
        assert jnp.allclose(jac, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of a arbitrary controlled Euler-angle-parameterized
        gate is correct."""
        dev = qml.device("default.qubit.jax", wires=2)
        a, b, c = np.array([theta, theta**3, np.sqrt(2) * theta])

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a, b, c):
            qml.QubitStateVector(np.array([1.0, -1.0]) / np.sqrt(2), wires=0)
            qml.CRot(a, b, c, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        res = circuit(a, b, c)
        expected = -np.cos(b / 2) * np.cos(0.5 * (a + c))
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = jax.grad(circuit, argnums=(0, 1, 2))(a, b, c)
        expected = np.array(
            [
                [
                    0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
                    0.5 * np.sin(b / 2) * np.cos(0.5 * (a + c)),
                    0.5 * np.cos(b / 2) * np.sin(0.5 * (a + c)),
                ]
            ]
        )
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_prob_differentiability(self, tol):
        """Test that the device probability can be differentiated"""
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = jnp.array(0.54)
        b = jnp.array(0.12)

        def cost(a, b):
            prob_wire_1 = circuit(a, b).squeeze()
            return prob_wire_1[1] - prob_wire_1[0]

        res = cost(a, b)
        expected = -jnp.cos(a) * jnp.cos(b)
        assert jnp.allclose(res, expected, atol=tol, rtol=0)

        grad = jax.jit(jax.grad(cost, argnums=(0, 1)))(a, b)
        expected = [jnp.sin(a) * jnp.cos(b), jnp.cos(a) * jnp.sin(b)]
        assert jnp.allclose(jnp.array(grad), jnp.array(expected), atol=tol, rtol=0)

    def test_prob_differentiability_broadcasted(self, tol):
        """Test that the broadcasted device probability can be differentiated"""
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = jnp.array([0.54, 0.32, 1.2])
        b = jnp.array(0.12)

        def cost(a, b):
            prob_wire_1 = circuit(a, b)
            return prob_wire_1[:, 1] - prob_wire_1[:, 0]

        res = cost(a, b)
        expected = -jnp.cos(a) * jnp.cos(b)
        assert jnp.allclose(res, expected, atol=tol, rtol=0)

        jac = jax.jacobian(cost, argnums=[0, 1])(a, b)
        expected = jnp.array([jnp.sin(a) * jnp.cos(b), jnp.cos(a) * jnp.sin(b)])
        expected = (jnp.diag(expected[0]), expected[1])  # Only first parameter is broadcasted
        assert all(jnp.allclose(j, e, atol=tol, rtol=0) for j, e in zip(jac, expected))

    def test_backprop_gradient(self, tol):
        """Tests that the gradient of the qnode is correct"""
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = jnp.array(-0.234)
        b = jnp.array(0.654)

        res = circuit(a, b)
        expected_cost = 0.5 * (jnp.cos(a) * jnp.cos(b) + jnp.cos(a) - jnp.cos(b) + 1)
        assert jnp.allclose(res, expected_cost, atol=tol, rtol=0)
        res = jax.grad(circuit, argnums=(0, 1))(a, b)
        expected_grad = jnp.array(
            [-0.5 * jnp.sin(a) * (jnp.cos(b) + 1), 0.5 * jnp.sin(b) * (1 - jnp.cos(a))]
        )

        assert jnp.allclose(jnp.array(res), jnp.array(expected_grad), atol=tol, rtol=0)

    def test_backprop_gradient_broadcasted(self, tol):
        """Tests that the gradient of the broadcasted qnode is correct"""
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = jnp.array(0.12)
        b = jnp.array([0.54, 0.32, 1.2])

        res = circuit(a, b)
        expected_cost = 0.5 * (jnp.cos(a) * jnp.cos(b) + jnp.cos(a) - jnp.cos(b) + 1)
        assert jnp.allclose(res, expected_cost, atol=tol, rtol=0)

        res = jax.jacobian(circuit, argnums=[0, 1])(a, b)
        expected = jnp.array(
            [-0.5 * jnp.sin(a) * (jnp.cos(b) + 1), 0.5 * jnp.sin(b) * (1 - jnp.cos(a))]
        )
        expected = (expected[0], jnp.diag(expected[1]))
        assert all(jnp.allclose(r, e, atol=tol, rtol=0) for r, e in zip(res, expected))

    @pytest.mark.parametrize("x, shift", [(0.0, 0.0), (0.5, -0.5)])
    def test_hessian_at_zero(self, x, shift):
        """Tests that the Hessian at vanishing state vector amplitudes
        is correct."""
        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(x):
            qml.RY(shift, wires=0)
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert qml.math.isclose(jax.grad(circuit)(x), 0.0)
        assert qml.math.isclose(jax.jacobian(jax.jacobian(circuit))(x), -1.0)
        assert qml.math.isclose(jax.grad(jax.grad(circuit))(x), -1.0)

    @pytest.mark.parametrize("operation", [qml.U3, qml.U3.compute_decomposition])
    @pytest.mark.parametrize("diff_method", ["backprop"])
    def test_jax_interface_gradient(self, operation, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the Jax interface, using a variety of differentiation methods."""
        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, diff_method=diff_method, interface="jax")
        def circuit(x, weights, w=None):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.QubitStateVector(1j * jnp.array([1, -1]) / jnp.sqrt(2), wires=w)
            operation(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        def cost(params):
            """Perform some classical processing"""
            return (circuit(params[0], params[1:], w=0) ** 2).reshape(())

        theta = 0.543
        phi = -0.234
        lam = 0.654

        params = jnp.array([theta, phi, lam])

        res = cost(params)
        expected_cost = (
            jnp.sin(lam) * jnp.sin(phi) - jnp.cos(theta) * jnp.cos(lam) * jnp.cos(phi)
        ) ** 2
        assert jnp.allclose(res, expected_cost, atol=tol, rtol=0)

        res = jax.grad(cost)(params)
        expected_grad = (
            jnp.array(
                [
                    jnp.sin(theta) * jnp.cos(lam) * jnp.cos(phi),
                    jnp.cos(theta) * jnp.cos(lam) * jnp.sin(phi) + jnp.sin(lam) * jnp.cos(phi),
                    jnp.cos(theta) * jnp.sin(lam) * jnp.cos(phi) + jnp.cos(lam) * jnp.sin(phi),
                ]
            )
            * 2
            * (jnp.sin(lam) * jnp.sin(phi) - jnp.cos(theta) * jnp.cos(lam) * jnp.cos(phi))
        )
        assert jnp.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("interface", ["autograd", "tf", "torch"])
    def test_error_backprop_wrong_interface(self, interface, tol):
        """Tests that an error is raised if diff_method='backprop' but not using
        the Jax interface"""
        dev = qml.device("default.qubit.jax", wires=1)

        def circuit(x, w=None):
            qml.RZ(x, wires=w)
            return qml.expval(qml.PauliX(w))

        error_type = qml.QuantumFunctionError
        with pytest.raises(
            error_type,
            match="default.qubit.jax only supports diff_method='backprop' when using the jax interface",
        ):
            qml.qnode(dev, diff_method="backprop", interface=interface)(circuit)

    def test_no_jax_interface_applied(self):
        """Tests that the JAX interface is not applied and no error is raised if qml.probs is used with the Jax
        interface when diff_method='backprop'

        When the JAX interface is applied, we can only get the expectation value and the variance of a QNode.
        """
        dev = qml.device("default.qubit.jax", wires=1, shots=None)

        def circuit():
            return qml.probs(wires=0)

        qnode = qml.qnode(dev, diff_method="backprop", interface="jax")(circuit)
        assert jnp.allclose(qnode(), jnp.array([1, 0]))


@pytest.mark.jax
class TestHighLevelIntegration:
    """Tests for integration with higher level components of PennyLane."""

    def test_do_not_split_analytic_jax(self, mocker):
        """Tests that the Hamiltonian is not split for shots=None using the jax device."""
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")

        dev = qml.device("default.qubit.jax", wires=2)
        H = qml.Hamiltonian(jnp.array([0.1, 0.2]), [qml.PauliX(0), qml.PauliZ(1)])

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit():
            return qml.expval(H)

        spy = mocker.spy(dev, "expval")

        circuit()
        # evaluated one expval altogether
        assert spy.call_count == 1

    def test_direct_eval_hamiltonian_broadcasted_error_jax(self, mocker):
        """Tests that an error is raised when attempting to evaluate a Hamiltonian with
        broadcasting and shots=None directly via its sparse representation with Jax."""
        dev = qml.device("default.qubit.jax", wires=2)
        H = qml.Hamiltonian(jnp.array([0.1, 0.2]), [qml.PauliX(0), qml.PauliZ(1)])

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit():
            qml.RX(jnp.zeros(5), 0)
            return qml.expval(H)

        spy = mocker.spy(dev, "expval")

        with pytest.raises(NotImplementedError, match="Hamiltonians for interface!=None"):
            circuit()

    def test_template_integration(self):
        """Test that a PassthruQNode using default.qubit.jax works with templates."""
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        weights = jnp.array(
            np.random.random(qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2))
        )

        grad = jax.grad(circuit)(weights)
        assert grad.shape == weights.shape

    def test_qnode_collection_integration(self):
        """Test that a PassthruQNode using default.qubit.jax works with QNodeCollections."""
        dev = qml.device("default.qubit.jax", wires=2)

        def ansatz(weights, **kwargs):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])

        obs_list = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)]
        qnodes = qml.map(ansatz, obs_list, dev, interface="jax")

        weights = jnp.array([0.1, 0.2])

        def cost(weights):
            return jnp.sum(jnp.array(qnodes(weights)))

        grad = jax.grad(cost)(weights)
        assert grad.shape == weights.shape

    def test_qnode_collection_integration_broadcasted(self):
        """Test that a broadcasted PassthruQNode default.qubit.jax
        works with QNodeCollections."""
        dev = qml.device("default.qubit.jax", wires=2)

        def ansatz(weights, **kwargs):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])

        obs_list = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)]
        qnodes = qml.map(ansatz, obs_list, dev, interface="jax")

        assert qnodes.interface == "jax"

        weights = jnp.array([[0.1, 0.65, 1.2], [0.2, 1.9, -0.6]])

        def cost(weights):
            return jnp.sum(qnodes(weights), axis=-1)

        res = cost(weights)
        assert res.shape == (3,)

        jac = jax.jacobian(cost)(weights)
        assert jac.shape == (3, 2, 3)


@pytest.mark.jax
class TestOps:
    """Unit tests for operations supported by the default.qubit.jax device"""

    @pytest.mark.parametrize("jacobian_transform", [jax.jacfwd, jax.jacrev])
    def test_multirz_jacobian(self, jacobian_transform):
        """Test that the patched numpy functions are used for the MultiRZ
        operation and the jacobian can be computed."""
        wires = 4
        dev = qml.device("default.qubit.jax", wires=wires)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(param):
            qml.MultiRZ(param, wires=[0, 1])
            return qml.probs(wires=list(range(wires)))

        param = 0.3
        res = jacobian_transform(circuit)(param)
        assert jnp.allclose(res, jnp.zeros(wires**2))

    def test_inverse_operation_jacobian_backprop(self, tol):
        """Test that inverse operations work in backprop
        mode"""
        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(param):
            qml.RY(param, wires=0).inv()
            return qml.expval(qml.PauliX(0))

        x = 0.3
        res = circuit(x)
        assert np.allclose(res, -np.sin(x), atol=tol, rtol=0)

        grad = jax.grad(lambda a: circuit(a).reshape(()))(x)
        assert np.allclose(grad, -np.cos(x), atol=tol, rtol=0)

    def test_full_subsystem(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultQubitJax(wires=["a", "b", "c"])
        state = jnp.array([1, 0, 0, 0, 1, 0, 1, 1]) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        assert jnp.all(dev._state.flatten() == state)
        spy.assert_not_called()

    def test_partial_subsystem(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""

        dev = DefaultQubitJax(wires=["a", "b", "c"])
        state = jnp.array([1, 0, 1, 0]) / jnp.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)
        res = jnp.sum(dev._state, axis=(1,)).flatten()

        assert jnp.all(res == state)
        spy.assert_called()


@pytest.mark.jax
class TestOpsBroadcasted:
    """Unit tests for broadcasted operations supported by the default.qubit.jax device"""

    @pytest.mark.parametrize("jacobian_transform", [jax.jacfwd, jax.jacrev])
    def test_multirz_jacobian_broadcasted(self, jacobian_transform):
        """Test that the patched numpy functions are used for the MultiRZ
        operation and the jacobian can be computed."""
        wires = 4
        dev = qml.device("default.qubit.jax", wires=wires)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(param):
            qml.MultiRZ(param, wires=[0, 1])
            return qml.probs(wires=list(range(wires)))

        param = jnp.array([0.3, 0.9, -4.3])
        res = jacobian_transform(circuit)(param)
        assert jnp.allclose(res, jnp.zeros((3, wires**2, 3)))

    def test_inverse_operation_jacobian_backprop_broadcasted(self, tol):
        """Test that inverse operations work in backprop
        mode"""
        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(param):
            qml.RY(param, wires=0).inv()
            return qml.expval(qml.PauliX(0))

        x = jnp.array([0.3, 0.9, -4.3])
        res = circuit(x)
        assert jnp.allclose(res, -jnp.sin(x), atol=tol, rtol=0)

        grad = jax.jacobian(circuit)(x)
        assert jnp.allclose(grad, -jnp.diag(jnp.cos(x)), atol=tol, rtol=0)

    def test_full_subsystem_broadcasted(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultQubitJax(wires=["a", "b", "c"])
        state = jnp.array([[1, 0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 0]]) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        assert jnp.all(dev._state.reshape((2, 8)) == state)
        spy.assert_not_called()

    def test_partial_subsystem_broadcasted(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""

        dev = DefaultQubitJax(wires=["a", "b", "c"])
        state = jnp.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]) / jnp.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        spy = mocker.spy(dev, "_scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)
        res = jnp.sum(dev._state, axis=(2,)).reshape((3, 4))

        assert jnp.allclose(res, state)
        spy.assert_called()


@pytest.mark.jax
class TestEstimateProb:
    """Test the estimate_probability method"""

    @pytest.mark.parametrize(
        "wires, expected", [([0], [0.5, 0.5]), (None, [0.5, 0, 0, 0.5]), ([0, 1], [0.5, 0, 0, 0.5])]
    )
    def test_estimate_probability(self, wires, expected, monkeypatch):
        """Tests the estimate_probability method"""
        dev = qml.device("default.qubit.jax", wires=2)
        samples = jnp.array([[0, 0], [1, 1], [1, 1], [0, 0]])

        with monkeypatch.context() as m:
            m.setattr(dev, "_samples", samples)
            res = dev.estimate_probability(wires=wires)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "wires, expected",
        [
            ([0], [[0.0, 0.5], [1.0, 0.5]]),
            (None, [[0.0, 0.5], [0, 0], [0, 0.5], [1.0, 0]]),
            ([0, 1], [[0.0, 0.5], [0, 0], [0, 0.5], [1.0, 0]]),
        ],
    )
    def test_estimate_probability_with_binsize(self, wires, expected, monkeypatch):
        """Tests the estimate_probability method with a bin size"""
        dev = qml.device("default.qubit.jax", wires=2)
        samples = jnp.array([[1, 1], [1, 1], [1, 0], [0, 0]])
        bin_size = 2

        with monkeypatch.context() as m:
            m.setattr(dev, "_samples", samples)
            res = dev.estimate_probability(wires=wires, bin_size=bin_size)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "wires, expected",
        [
            ([0], [[0.0, 1.0], [0.5, 0.5], [0.25, 0.75]]),
            (None, [[0, 0, 0.25, 0.75], [0.5, 0, 0, 0.5], [0.25, 0, 0.25, 0.5]]),
            ([0, 1], [[0, 0, 0.25, 0.75], [0.5, 0, 0, 0.5], [0.25, 0, 0.25, 0.5]]),
        ],
    )
    def test_estimate_probability_with_broadcasting(self, wires, expected, monkeypatch):
        """Tests the estimate_probability method with parameter broadcasting"""
        dev = qml.device("default.qubit.jax", wires=2)
        samples = jnp.array(
            [
                [[1, 0], [1, 1], [1, 1], [1, 1]],
                [[0, 0], [1, 1], [1, 1], [0, 0]],
                [[1, 0], [1, 1], [1, 1], [0, 0]],
            ]
        )

        with monkeypatch.context() as m:
            m.setattr(dev, "_samples", samples)
            res = dev.estimate_probability(wires=wires)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "wires, expected",
        [
            (
                [0],
                [
                    [[0, 0, 0.5], [1, 1, 0.5]],
                    [[0.5, 0.5, 0], [0.5, 0.5, 1]],
                    [[0, 0.5, 1], [1, 0.5, 0]],
                ],
            ),
            (
                None,
                [
                    [[0, 0, 0], [0, 0, 0.5], [0.5, 0, 0], [0.5, 1, 0.5]],
                    [[0.5, 0.5, 0], [0, 0, 0], [0, 0, 0], [0.5, 0.5, 1]],
                    [[0, 0.5, 0.5], [0, 0, 0.5], [0.5, 0, 0], [0.5, 0.5, 0]],
                ],
            ),
            (
                [0, 1],
                [
                    [[0, 0, 0], [0, 0, 0.5], [0.5, 0, 0], [0.5, 1, 0.5]],
                    [[0.5, 0.5, 0], [0, 0, 0], [0, 0, 0], [0.5, 0.5, 1]],
                    [[0, 0.5, 0.5], [0, 0, 0.5], [0.5, 0, 0], [0.5, 0.5, 0]],
                ],
            ),
        ],
    )
    def test_estimate_probability_with_binsize_with_broadcasting(
        self, wires, expected, monkeypatch
    ):
        """Tests the estimate_probability method with a bin size and parameter broadcasting"""
        dev = qml.device("default.qubit.jax", wires=2)
        bin_size = 2
        samples = jnp.array(
            [
                [[1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1]],
                [[0, 0], [1, 1], [1, 1], [0, 0], [1, 1], [1, 1]],
                [[1, 0], [1, 1], [1, 1], [0, 0], [0, 1], [0, 0]],
            ]
        )

        with monkeypatch.context() as m:
            m.setattr(dev, "_samples", samples)
            res = dev.estimate_probability(wires=wires, bin_size=bin_size)

        assert np.allclose(res, expected)
