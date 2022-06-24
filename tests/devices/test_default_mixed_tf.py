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
Tests for the ``default.mixed`` device for the TensorFlow interface
"""
import re
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.devices.default_mixed import DefaultMixed
from pennylane import DeviceError

pytestmark = pytest.mark.tf

tf = pytest.importorskip("tensorflow", minversion="2.1")

# The decorator and interface pairs to test:
#   1. No QNode decorator and "tf" interface
#   2. QNode decorated with tf.function and "tf" interface
#   3. No QNode decorator and "tf-autograph" interface
decorators_interfaces = [(lambda x: x, "tf"), (tf.function, "tf"), (lambda x: x, "tf-autograph")]


class TestQNodeIntegration:
    """Integration tests for default.mixed.tf. This test ensures it integrates
    properly with the PennyLane UI, in particular the QNode."""

    def test_load_device(self):
        """Test that the plugin device loads correctly"""
        dev = qml.device("default.mixed", wires=2)
        assert dev.num_wires == 2
        assert dev.shots == None
        assert dev.short_name == "default.mixed"
        assert dev.capabilities()["passthru_devices"]["tf"] == "default.mixed"

    def test_qubit_circuit(self, tol):
        """Test that the device provides the correct
        result for a simple circuit."""
        p = tf.Variable(0.543)

        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert circuit.gradient_fn == "backprop"
        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_correct_state(self, tol):
        """Test that the device state is correct after evaluating a
        quantum function on the device"""

        dev = qml.device("default.mixed", wires=2)

        state = dev.state
        expected = np.zeros((4, 4))
        expected[0, 0] = 1
        assert np.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(a):
            qml.Hadamard(wires=0)
            qml.RZ(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit(tf.constant(np.pi / 4))
        state = dev.state

        amplitude = np.exp(-1j * np.pi / 4) / 2
        expected = np.array(
            [[0.5, 0, amplitude, 0], [0, 0, 0, 0], [np.conj(amplitude), 0, 0.5, 0], [0, 0, 0, 0]]
        )

        assert np.allclose(state, expected, atol=tol, rtol=0)


class TestDtypePreserved:
    """Test that the user-defined dtype of the device is preserved for QNode
    evaluation"""

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        "measurement",
        [
            qml.expval(qml.PauliY(0)),
            qml.var(qml.PauliY(0)),
            qml.probs(wires=[1]),
            qml.probs(wires=[2, 0]),
        ],
    )
    def test_real_dtype(self, r_dtype, measurement, tol):
        """Test that the user-defined dtype of the device is preserved
        for QNodes with real-valued outputs"""
        p = tf.constant(0.543)

        dev = qml.device("default.mixed", wires=3, r_dtype=r_dtype)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == r_dtype

    @pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize(
        "measurement",
        [qml.state(), qml.density_matrix(wires=[1]), qml.density_matrix(wires=[2, 0])],
    )
    def test_complex_dtype(self, c_dtype, measurement, tol):
        """Test that the user-defined dtype of the device is preserved
        for QNodes with complex-valued outputs"""
        p = tf.constant(0.543)

        dev = qml.device("default.mixed", wires=3, c_dtype=c_dtype)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == c_dtype


class TestOps:
    """Unit tests for operations supported by the default.mixed.tf device"""

    def test_multirz_jacobian(self):
        """Test that the patched numpy functions are used for the MultiRZ
        operation and the jacobian can be computed."""
        wires = 4
        dev = qml.device("default.mixed", wires=wires)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(param):
            qml.MultiRZ(param, wires=[0, 1])
            return qml.probs(wires=list(range(wires)))

        param = tf.Variable(0.3, trainable=True)

        with tf.GradientTape() as tape:
            out = circuit(param)

        res = tape.gradient(out, param)

        assert np.allclose(res, np.zeros(wires**2))

    def test_full_subsystem(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultMixed(wires=["a", "b", "c"])
        state = tf.constant([1, 0, 0, 0, 1, 0, 1, 1], dtype=tf.complex128) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(qml.math, "scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        state = np.outer(state, np.conj(state))

        assert np.all(tf.reshape(dev._state, (-1,)) == tf.reshape(state, (-1,)))
        spy.assert_not_called()

    def test_partial_subsystem(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""

        dev = DefaultMixed(wires=["a", "b", "c"])
        state = tf.constant([1, 0, 1, 0], dtype=tf.complex128) / np.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        spy = mocker.spy(qml.math, "scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        state = np.kron(np.outer(state, np.conj(state)), np.array([[1, 0], [0, 0]]))

        assert np.all(tf.reshape(dev._state, (8, 8)) == state)
        spy.assert_called()


class TestPassthruIntegration:
    """Tests for integration with the PassthruQNode"""

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    def test_jacobian_variable_multiply(self, decorator, interface, tol):
        """Test that jacobian of a QNode with an attached default.mixed.tf device
        gives the correct result in the case of parameters multiplied by scalars"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        weights = tf.Variable([x, y, z], trainable=True)

        dev = qml.device("default.mixed", wires=1)

        @decorator
        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.gradient_fn == "backprop"
        res = circuit(weights)

        expected = np.cos(3 * x) * np.cos(y) * np.cos(z / 2) - np.sin(3 * x) * np.sin(z / 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            out = circuit(weights)

        res = tape.jacobian(out, weights)

        expected = np.array(
            [
                -3 * (np.sin(3 * x) * np.cos(y) * np.cos(z / 2) + np.cos(3 * x) * np.sin(z / 2)),
                -np.cos(3 * x) * np.sin(y) * np.cos(z / 2),
                -0.5 * (np.sin(3 * x) * np.cos(z / 2) + np.cos(3 * x) * np.cos(y) * np.sin(z / 2)),
            ]
        )

        assert qml.math.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    def test_jacobian_repeated(self, decorator, interface, tol):
        """Test that jacobian of a QNode with an attached default.mixed.tf device
        gives the correct result in the case of repeated parameters"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        p = tf.Variable([x, y, z], trainable=True)
        dev = qml.device("default.mixed", wires=1)

        @decorator
        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(p)

        expected = np.cos(y) ** 2 - np.sin(x) * np.sin(y) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, p)

        expected = np.array(
            [-np.cos(x) * np.sin(y) ** 2, -2 * (np.sin(x) + 1) * np.sin(y) * np.cos(y), 0]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    def test_backprop_jacobian_agrees_parameter_shift(self, decorator, interface, tol):
        """Test that jacobian of a QNode with an attached default.mixed.tf device
        gives the correct result with respect to the parameter-shift method"""
        p = np.array([0.43316321, 0.2162158, 0.75110998, 0.94714242])
        p_tf = tf.Variable(p, trainable=True)

        def circuit(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        dev1 = qml.device("default.mixed", wires=3)
        dev2 = qml.device("default.mixed", wires=3)

        circuit1 = decorator(qml.QNode(circuit, dev1, diff_method="backprop", interface=interface))
        circuit2 = qml.QNode(circuit, dev2, diff_method="parameter-shift")

        assert circuit1.gradient_fn == "backprop"
        assert circuit2.gradient_fn is qml.gradients.param_shift

        with tf.GradientTape() as tape:
            res = circuit1(p_tf)

        assert np.allclose(res, circuit2(p), atol=tol, rtol=0)

        res = tape.jacobian(res, p_tf)
        assert np.allclose(res, qml.jacobian(circuit2)(p), atol=tol, rtol=0)

    @pytest.mark.parametrize("wires", [[0], ["abc"]])
    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    def test_state_differentiability(self, decorator, interface, wires, tol):
        """Test that the device state can be differentiated"""
        dev = qml.device("default.mixed", wires=wires)

        @decorator
        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit(a):
            qml.RY(a, wires=wires[0])
            return qml.state()

        a = tf.Variable(0.54, trainable=True)

        with tf.GradientTape() as tape:
            state = circuit(a)
            res = tf.abs(state) ** 2
            res = res[1][1] - res[0][0]

        grad = tape.gradient(res, a)
        expected = np.sin(a)

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    def test_state_vector_differentiability(self, decorator, interface, tol):
        """Test that the device state vector can be differentiated directly"""
        dev = qml.device("default.mixed", wires=1)

        @decorator
        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit(a):
            qml.RY(a, wires=0)
            return qml.state()

        a = tf.Variable(0.54, dtype=tf.complex128, trainable=True)

        with tf.GradientTape() as tape:
            res = circuit(a)

        grad = tape.jacobian(res, a)
        expected = 0.5 * np.array([[-np.sin(a), np.cos(a)], [np.cos(a), np.sin(a)]])

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    @pytest.mark.parametrize("wires", [range(2), [-12.32, "abc"]])
    def test_density_matrix_differentiability(self, decorator, interface, wires, tol):
        """Test that the density matrix can be differentiated"""
        dev = qml.device("default.mixed", wires=wires)

        @decorator
        @qml.qnode(dev, diff_method="backprop", interface=interface)
        def circuit(a):
            qml.RY(a, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            return qml.density_matrix(wires=wires[1])

        a = tf.Variable(0.54, trainable=True)

        with tf.GradientTape() as tape:
            state = circuit(a)
            res = tf.abs(state) ** 2
            res = res[1][1] - res[0][0]

        grad = tape.gradient(res, a)
        expected = np.sin(a)

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    def test_prob_differentiability(self, decorator, interface, tol):
        """Test that the device probability can be differentiated"""
        dev = qml.device("default.mixed", wires=2)

        @decorator
        @qml.qnode(dev, diff_method="backprop", interface=interface)
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = tf.Variable(0.54, trainable=True)
        b = tf.Variable(0.12, trainable=True)

        with tf.GradientTape() as tape:
            prob_wire_1 = circuit(a, b)
            res = prob_wire_1[1] - prob_wire_1[0]

        expected = -np.cos(a) * np.cos(b)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = tape.gradient(res, [a, b])
        expected = [np.sin(a) * np.cos(b), np.cos(a) * np.sin(b)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    def test_prob_vector_differentiability(self, decorator, interface, tol):
        """Test that the device probability vector can be differentiated directly"""
        dev = qml.device("default.mixed", wires=2)

        @decorator
        @qml.qnode(dev, diff_method="backprop", interface=interface)
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = tf.Variable(0.54, trainable=True)
        b = tf.Variable(0.12, trainable=True)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        expected = [
            np.cos(a / 2) ** 2 * np.cos(b / 2) ** 2 + np.sin(a / 2) ** 2 * np.sin(b / 2) ** 2,
            np.cos(a / 2) ** 2 * np.sin(b / 2) ** 2 + np.sin(a / 2) ** 2 * np.cos(b / 2) ** 2,
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = tape.jacobian(res, [a, b])
        expected = 0.5 * np.array(
            [
                [-np.sin(a) * np.cos(b), np.sin(a) * np.cos(b)],
                [-np.cos(a) * np.sin(b), np.cos(a) * np.sin(b)],
            ]
        )

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_sample_backprop_error(self):
        """Test that sampling in backpropagation mode raises an error"""
        dev = qml.device("default.mixed", wires=1, shots=100)

        msg = "Backpropagation is only supported when shots=None"

        with pytest.raises(qml.QuantumFunctionError, match=msg):

            @qml.qnode(dev, diff_method="backprop", interface="tf")
            def circuit(a):
                qml.RY(a, wires=0)
                return qml.sample(qml.PauliZ(0))

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    def test_expval_gradient(self, decorator, interface, tol):
        """Tests that the gradient of expval is correct"""
        dev = qml.device("default.mixed", wires=2)

        @decorator
        @qml.qnode(dev, diff_method="backprop", interface=interface)
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = tf.Variable(-0.234, trainable=True)
        b = tf.Variable(0.654, trainable=True)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        expected_cost = 0.5 * (np.cos(a) * np.cos(b) + np.cos(a) - np.cos(b) + 1)
        assert np.allclose(res, expected_cost, atol=tol, rtol=0)

        res = tape.gradient(res, [a, b])
        expected_grad = np.array(
            [-0.5 * np.sin(a) * (np.cos(b) + 1), 0.5 * np.sin(b) * (1 - np.cos(a))]
        )
        assert np.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    @pytest.mark.parametrize("x, shift", [(0.0, 0.0), (0.5, -0.5)])
    def test_hessian_at_zero(self, decorator, interface, x, shift):
        """Tests that the Hessian at vanishing state vector amplitudes
        is correct."""
        dev = qml.device("default.mixed", wires=1)

        shift = tf.constant(shift)
        x = tf.Variable(x)

        @decorator
        @qml.qnode(dev, interface=interface, diff_method="backprop")
        def circuit(x):
            qml.RY(shift, wires=0)
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape(persistent=True) as t1:
                value = circuit(x)
            grad = t1.gradient(value, x)
            jac = t1.jacobian(value, x)
        hess_grad = t2.gradient(grad, x)
        hess_jac = t2.jacobian(jac, x)

        assert qml.math.isclose(grad, 0.0)
        assert qml.math.isclose(hess_grad, -1.0)
        assert qml.math.isclose(hess_jac, -1.0)

    @pytest.mark.parametrize("operation", [qml.U3, qml.U3.compute_decomposition])
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
    def test_tf_interface_gradient(self, operation, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the TF interface, using a variety of differentiation methods."""
        dev = qml.device("default.mixed", wires=1)
        state = tf.Variable(1j * np.array([1, -1]) / np.sqrt(2), trainable=False)

        @qml.qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(x, weights, w):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.QubitStateVector(state, wires=w)
            operation(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        def cost(params):
            """Perform some classical processing"""
            return circuit(params[0], params[1:], w=0) ** 2

        theta = 0.543
        phi = -0.234
        lam = 0.654

        params = tf.Variable([theta, phi, lam], trainable=True)

        res = cost(params)
        expected_cost = (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi)) ** 2
        assert np.allclose(res, expected_cost, atol=tol, rtol=0)

        # Check that the correct differentiation method is being used.
        if diff_method == "backprop":
            assert circuit.gradient_fn == "backprop"
        elif diff_method == "parameter-shift":
            assert circuit.gradient_fn is qml.gradients.param_shift
        else:
            assert circuit.gradient_fn is qml.gradients.finite_diff

        with tf.GradientTape() as tape:
            out = cost(params)

        res = tape.gradient(out, params)

        expected_grad = (
            np.array(
                [
                    np.sin(theta) * np.cos(lam) * np.cos(phi),
                    np.cos(theta) * np.cos(lam) * np.sin(phi) + np.sin(lam) * np.cos(phi),
                    np.cos(theta) * np.sin(lam) * np.cos(phi) + np.cos(lam) * np.sin(phi),
                ]
            )
            * 2
            * (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi))
        )
        assert np.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    @pytest.mark.parametrize(
        "dev_name,diff_method,mode",
        [
            ["default.mixed", "finite-diff", "backward"],
            ["default.mixed", "parameter-shift", "backward"],
            ["default.mixed", "backprop", "forward"],
        ],
    )
    def test_ragged_differentiation(self, decorator, interface, dev_name, diff_method, mode, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        dev = qml.device(dev_name, wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qml.qnode(dev, diff_method=diff_method, mode=mode, interface=interface)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])]

        with tf.GradientTape() as tape:
            res = circuit(x, y)

        expected = np.array(
            [
                tf.cos(x),
                (1 + tf.cos(x) * tf.cos(y)) / 2,
                (1 - tf.cos(x) * tf.cos(y)) / 2,
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [x, y])
        expected = np.array(
            [
                [-tf.sin(x), -tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                [0, -tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator, interface", decorators_interfaces)
    def test_batching(self, decorator, interface, tol):
        """Tests that the gradient of the qnode is correct with batching"""
        dev = qml.device("default.mixed", wires=2)

        @decorator
        @qml.batch_params
        @qml.qnode(dev, diff_method="backprop", interface=interface)
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = tf.Variable([-0.234, 0.678], trainable=True)
        b = tf.Variable([0.654, 1.236], trainable=True)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        expected_cost = 0.5 * (np.cos(a) * np.cos(b) + np.cos(a) - np.cos(b) + 1)
        assert np.allclose(res, expected_cost, atol=tol, rtol=0)

        res_a, res_b = tape.jacobian(res, [a, b])
        expected_a, expected_b = [
            -0.5 * np.sin(a) * (np.cos(b) + 1),
            0.5 * np.sin(b) * (1 - np.cos(a)),
        ]

        assert np.allclose(tf.linalg.diag_part(res_a), expected_a, atol=tol, rtol=0)
        assert np.allclose(tf.linalg.diag_part(res_b), expected_b, atol=tol, rtol=0)


class TestHighLevelIntegration:
    """Tests for integration with higher level components of PennyLane."""

    def test_template_integration(self):
        """Test that a PassthruQNode default.mixed.tf works with templates."""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        weights = tf.Variable(np.random.random(shape), trainable=True)

        with tf.GradientTape() as tape:
            res = circuit(weights)

        grad = tape.gradient(res, weights)
        assert isinstance(grad, tf.Tensor)
        assert grad.shape == weights.shape

    def test_qnode_collection_integration(self):
        """Test that a PassthruQNode default.mixed.tf works with QNodeCollections."""
        dev = qml.device("default.mixed", wires=2)

        obs_list = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)]
        qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="tf")

        assert qnodes.interface == "tf"

        weights = tf.Variable(
            np.random.random(qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2))
        )

        @tf.function
        def cost(weights):
            return tf.reduce_sum(qnodes(weights))

        with tf.GradientTape() as tape:
            res = qnodes(weights)

        grad = tape.gradient(res, weights)

        assert isinstance(grad, tf.Tensor)
        assert grad.shape == weights.shape

    def test_tf_function_channel_ops(self):
        """Test that tf.function works for a QNode with channel ops"""
        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="tf")
        def circuit(p):
            qml.AmplitudeDamping(p, wires=0)
            qml.GeneralizedAmplitudeDamping(p, p, wires=0)
            qml.PhaseDamping(p, wires=0)
            qml.DepolarizingChannel(p, wires=0)
            qml.BitFlip(p, wires=0)
            qml.ResetError(p, p, wires=0)
            qml.PauliError("X", p, wires=0)
            qml.PhaseFlip(p, wires=0)
            qml.ThermalRelaxationError(p, p, p, 0.0001, wires=0)
            return qml.expval(qml.PauliZ(0))

        vcircuit = tf.function(circuit)

        x = tf.Variable(0.005)
        res = vcircuit(x)

        # compare results to results of non-decorated circuit
        assert np.allclose(circuit(x), res)
