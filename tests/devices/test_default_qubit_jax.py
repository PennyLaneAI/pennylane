import pytest

jax = pytest.importorskip("jax", minversion="0.2")
jnp = jax.numpy
from jax.config import config
import numpy as np
import pennylane as qml
from pennylane.devices.default_qubit_jax import DefaultQubitJax
from pennylane import DeviceError


def test_analytic_deprecation():
    """Tests if the kwarg `analytic` is used and displays error message."""
    msg = "The analytic argument has been replaced by shots=None. "
    msg += "Please use shots=None instead of analytic=True."

    with pytest.raises(
        DeviceError,
        match=msg,
    ):
        qml.device("default.qubit.jax", wires=1, shots=1, analytic=True)


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
            "supports_reversible_diff": False,
            "supports_inverse_operations": True,
            "supports_analytic_computation": True,
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
        assert cap["supports_reversible_diff"] == False
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
        # type.
        # Just test that it works and spits our the right value.
        assert jnp.isclose(circuit(p), expected, atol=tol, rtol=0)

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

    def test_sampling_with_jit(self):
        """Test that sampling works with a jax.jit"""

        @jax.jit
        def circuit(key):
            dev = qml.device("default.qubit.jax", wires=1, shots=1000, prng_key=key)

            @qml.qnode(dev, interface="jax", diff_method=None)
            def inner_circuit():
                qml.Hadamard(0)
                return qml.sample(qml.PauliZ(wires=0))

            return inner_circuit()

        a = circuit(jax.random.PRNGKey(0))
        b = circuit(jax.random.PRNGKey(0))
        c = circuit(jax.random.PRNGKey(1))
        np.testing.assert_array_equal(a, b)
        assert not np.all(a == c)

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

    def test_state_differentiability(self, tol):
        """Test that the device state can be differentiated"""
        dev = qml.device("default.qubit.jax", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a):
            qml.RY(a, wires=0)
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


    @pytest.mark.parametrize("theta", np.linspace(-2 * np.pi, np.pi, 7))
    def test_CRot_gradient(self, theta, tol):
        """Tests that the automatic gradient of a arbitrary controlled Euler-angle-parameterized
        gate is correct."""
        dev = qml.device("default.qubit.jax", wires=2)
        a, b, c = np.array([theta, theta ** 3, np.sqrt(2) * theta])

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
        assert jnp.allclose(grad, expected, atol=tol, rtol=0)

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
        assert jnp.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("operation", [qml.U3, qml.U3.decomposition])
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

class TestHighLevelIntegration:
    """Tests for integration with higher level components of PennyLane."""

    def test_template_integration(self):
        """Test that a PassthruQNode using default.qubit.jax works with templates."""
        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        weights = jnp.array(qml.init.strong_ent_layers_normal(n_wires=2, n_layers=2))

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
        assert jnp.allclose(res, jnp.zeros(wires ** 2))

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
