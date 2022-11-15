# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for using the TensorFlow interface with a QNode"""
import numpy as np
import pytest

pytestmark = pytest.mark.tf

tf = pytest.importorskip("tensorflow")

import pennylane as qml
from pennylane import qnode
from pennylane.tape import QuantumTape

qubit_device_and_diff_method = [
    ["default.qubit", "finite-diff", "backward"],
    ["default.qubit", "parameter-shift", "backward"],
    ["default.qubit", "backprop", "forward"],
    ["default.qubit", "adjoint", "forward"],
    ["default.qubit", "adjoint", "backward"],
]

interface_and_qubit_device_and_diff_method = [
    ["auto"] + inner_list for inner_list in qubit_device_and_diff_method
] + [["tf"] + inner_list for inner_list in qubit_device_and_diff_method]


@pytest.mark.parametrize(
    "interface, dev_name,diff_method,mode", interface_and_qubit_device_and_diff_method
)
class TestQNode:
    """Test that using the QNode with TensorFlow integrates with the PennyLane stack"""

    def test_execution_with_interface(self, interface, dev_name, diff_method, mode):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface=interface, diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(0.1)
        circuit(a)

        # if executing outside a gradient tape, the number of trainable parameters
        # cannot be determined by TensorFlow
        assert circuit.qtape.trainable_params == []

        with tf.GradientTape() as tape:
            res = circuit(a)

        assert circuit.interface == interface

        # with the interface, the tape returns tensorflow tensors
        assert isinstance(res, tf.Tensor)
        assert res.shape == tuple()

        # the tape is able to deduce trainable parameters
        assert circuit.qtape.trainable_params == [0]

        # gradients should work
        grad = tape.gradient(res, a)
        assert isinstance(grad, tf.Tensor)
        assert grad.shape == tuple()

    def test_interface_swap(self, interface, dev_name, diff_method, mode, tol):
        """Test that the TF interface can be applied to a QNode
        with a pre-existing interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="autograd", diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        from pennylane import numpy as anp

        a = anp.array(0.1, requires_grad=True)

        res1 = circuit(a)
        grad_fn = qml.grad(circuit)
        grad1 = grad_fn(a)

        # switch to TF interface
        circuit.interface = interface

        a = tf.Variable(0.1, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res2 = circuit(a)

        grad2 = tape.gradient(res2, a)
        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(grad1, grad2, atol=tol, rtol=0)

    def test_drawing(self, interface, dev_name, diff_method, mode):
        """Test circuit drawing when using the TF interface"""

        x = tf.Variable(0.1, dtype=tf.float64)
        y = tf.Variable([0.2, 0.3], dtype=tf.float64)
        z = tf.Variable(0.4, dtype=tf.float64)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, interface=interface, diff_method=diff_method, mode=mode)
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        result = qml.draw(circuit)(p1=x, p3=z)
        expected = "0: ──RX(0.10)──RX(0.40)─╭●─┤  State\n" "1: ──RY(0.06)───────────╰X─┤  State"
        assert result == expected

    def test_jacobian(self, interface, dev_name, diff_method, mode, mocker, tol):
        """Test jacobian calculation"""
        if diff_method == "parameter-shift":
            spy = mocker.spy(qml.gradients.param_shift, "transform_fn")
        elif diff_method == "finite-diff":
            spy = mocker.spy(qml.gradients.finite_diff, "transform_fn")

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, mode=mode, interface=interface)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        assert circuit.qtape.trainable_params == [0, 1]

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [a, b])
        expected = [[-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        if diff_method in ("parameter-shift", "finite-diff"):
            spy.assert_called()

    @pytest.mark.xfail
    def test_jacobian_dtype(self, interface, dev_name, diff_method, mode, tol):
        """Test calculating the jacobian with a different datatype"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        a = tf.Variable(0.1, dtype=tf.float32)
        b = tf.Variable(0.2, dtype=tf.float32)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, diff_method=diff_method, mode=mode)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        circuit.to_tf(dtype=tf.float32)
        assert circuit.dtype is tf.float32

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        assert circuit.qtape.interface == interface
        assert circuit.qtape.trainable_params == [0, 1]

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)
        assert res.dtype is tf.float32

        res = tape.jacobian(res, [a, b])
        assert [r.dtype is tf.float32 for r in res]

    def test_jacobian_options(self, interface, dev_name, diff_method, mode, mocker, tol):
        """Test setting finite-difference jacobian options"""
        if diff_method != "finite-diff":
            pytest.skip("Test only works with finite diff")

        spy = mocker.spy(qml.gradients.finite_diff, "transform_fn")

        a = tf.Variable([0.1, 0.2])

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface=interface, h=1e-8, approx_order=2, diff_method=diff_method, mode=mode)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(a)

        tape.jacobian(res, a)

        for args in spy.call_args_list:
            assert args[1]["approx_order"] == 2
            assert args[1]["h"] == 1e-8

    def test_changing_trainability(self, interface, dev_name, diff_method, mode, mocker, tol):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface=interface, diff_method="parameter-shift")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        # the tape has reported both gate arguments as trainable
        assert circuit.qtape.trainable_params == [0, 1]

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy = mocker.spy(qml.gradients.param_shift, "transform_fn")

        jac = tape.jacobian(res, [a, b])
        expected = [
            [-tf.sin(a), tf.sin(a) * tf.sin(b)],
            [0, -tf.cos(a) * tf.cos(b)],
        ]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        # The parameter-shift rule has been called for each argument
        assert len(spy.spy_return[0]) == 4

        # make the second QNode argument a constant
        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.constant(0.8, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        # the tape has reported only the first argument as trainable
        assert circuit.qtape.trainable_params == [0]

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy.call_args_list = []
        jac = tape.jacobian(res, a)
        expected = [-tf.sin(a), tf.sin(a) * tf.sin(b)]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        # the gradient transform has only been called once
        assert len(spy.call_args_list) == 1

    def test_classical_processing(self, interface, dev_name, diff_method, mode, tol):
        """Test classical processing within the quantum tape"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.constant(0.2, dtype=tf.float64)
        c = tf.Variable(0.3, dtype=tf.float64)

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, mode=mode, interface=interface)
        def circuit(x, y, z):
            qml.RY(x * z, wires=0)
            qml.RZ(y, wires=0)
            qml.RX(z + z**2 + tf.sin(a), wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(a, b, c)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == [0, 2]
            assert circuit.qtape.get_parameters() == [a * c, c + c**2 + tf.sin(a)]

        res = tape.jacobian(res, [a, b, c])

        assert isinstance(res[0], tf.Tensor)
        assert res[1] is None
        assert isinstance(res[2], tf.Tensor)

    def test_no_trainable_parameters(self, interface, dev_name, diff_method, mode, tol):
        """Test evaluation if there are no trainable parameters"""
        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, mode=mode, interface=interface)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = 0.1
        b = tf.constant(0.2, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == []

        assert res.shape == (2,)
        assert isinstance(res, tf.Tensor)

    @pytest.mark.parametrize("U", [tf.constant([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])])
    def test_matrix_parameter(self, interface, dev_name, diff_method, mode, U, tol):
        """Test that the TF interface works correctly
        with a matrix parameter"""
        a = tf.Variable(0.1, dtype=tf.float64)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, mode=mode, interface=interface)
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(U, a)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == [1]

        assert np.allclose(res, -tf.cos(a), atol=tol, rtol=0)

        res = tape.jacobian(res, a)
        assert np.allclose(res, tf.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, interface, dev_name, diff_method, mode, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""

        class U3(qml.U3):
            def expand(self):
                theta, phi, lam = self.data
                wires = self.wires

                with QuantumTape() as tape:
                    qml.Rot(lam, theta, -lam, wires=wires)
                    qml.PhaseShift(phi + lam, wires=wires)

                return tape

        dev = qml.device(dev_name, wires=1)
        a = np.array(0.1)
        p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, mode=mode, interface=interface)
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        with tf.GradientTape() as tape:
            res = circuit(a, p)

        assert circuit.qtape.trainable_params == [1, 2, 3]

        expected = tf.cos(a) * tf.cos(p[1]) * tf.sin(p[0]) + tf.sin(a) * (
            tf.cos(p[2]) * tf.sin(p[1]) + tf.cos(p[0]) * tf.cos(p[1]) * tf.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, p)
        expected = np.array(
            [
                tf.cos(p[1]) * (tf.cos(a) * tf.cos(p[0]) - tf.sin(a) * tf.sin(p[0]) * tf.sin(p[2])),
                tf.cos(p[1]) * tf.cos(p[2]) * tf.sin(a)
                - tf.sin(p[1])
                * (tf.cos(a) * tf.sin(p[0]) + tf.cos(p[0]) * tf.sin(a) * tf.sin(p[2])),
                tf.sin(a)
                * (tf.cos(p[0]) * tf.cos(p[1]) * tf.cos(p[2]) - tf.sin(p[1]) * tf.sin(p[2])),
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("interface", ["auto", "tf"])
class TestShotsIntegration:
    """Test that the QNode correctly changes shot value, and
    differentiates it."""

    def test_changing_shots(self, interface, mocker, tol):
        """Test that changing shots works on execution"""
        dev = qml.device("default.qubit", wires=2, shots=None)
        a, b = [0.543, -0.654]
        weights = tf.Variable([a, b], dtype=tf.float64)

        @qnode(dev, interface=interface, diff_method=qml.gradients.param_shift)
        def circuit(weights):
            qml.RY(weights[0], wires=0)
            qml.RX(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        spy = mocker.spy(dev, "sample")

        # execute with device default shots (None)
        res = circuit(weights)
        assert np.allclose(res, -np.cos(a) * np.sin(b), atol=tol, rtol=0)
        spy.assert_not_called()

        # execute with shots=100
        res = circuit(weights, shots=100)
        spy.assert_called()
        assert spy.spy_return.shape == (100,)

        # device state has been unaffected
        assert dev.shots is None
        spy = mocker.spy(dev, "sample")
        res = circuit(weights)
        assert np.allclose(res, -np.cos(a) * np.sin(b), atol=tol, rtol=0)
        spy.assert_not_called()

    def test_gradient_integration(self, interface, tol):
        """Test that temporarily setting the shots works
        for gradient computations"""
        dev = qml.device("default.qubit", wires=2, shots=None)
        a, b = [0.543, -0.654]
        weights = tf.Variable([a, b], dtype=tf.float64)

        @qnode(dev, interface=interface, diff_method=qml.gradients.param_shift)
        def circuit(weights):
            qml.RY(weights[0], wires=0)
            qml.RX(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        with tf.GradientTape() as tape:
            res = circuit(weights, shots=[10000, 10000, 10000])
            res = tf.transpose(tf.stack(res))

        assert dev.shots is None
        assert len(res) == 3

        jacobian = tape.jacobian(res, weights)
        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(np.mean(jacobian, axis=0), expected, atol=0.1, rtol=0)

    def test_multiple_gradient_integration(self, interface, tol):
        """Test that temporarily setting the shots works
        for gradient computations, even if the QNode has been re-evaluated
        with a different number of shots in the meantime."""
        dev = qml.device("default.qubit", wires=2, shots=None)
        a, b = [0.543, -0.654]
        weights = tf.Variable([a, b], dtype=tf.float64)

        @qnode(dev, interface=interface, diff_method=qml.gradients.param_shift)
        def circuit(weights):
            qml.RY(weights[0], wires=0)
            qml.RX(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        with tf.GradientTape() as tape:
            res1 = circuit(weights)

        assert qml.math.shape(res1) == tuple()

        res2 = circuit(weights, shots=[(1, 1000)])
        assert qml.math.shape(res2) == (1000,)

        grad = tape.gradient(res1, weights)
        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_update_diff_method(self, interface, mocker, tol):
        """Test that temporarily setting the shots updates the diff method"""
        dev = qml.device("default.qubit", wires=2, shots=100)
        weights = tf.Variable([0.543, -0.654], dtype=tf.float64)

        spy = mocker.spy(qml, "execute")

        @qnode(dev, interface=interface)
        def circuit(weights):
            qml.RY(weights[0], wires=0)
            qml.RX(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        circuit(weights)
        # since we are using finite shots, parameter-shift will
        # be chosen
        assert circuit.gradient_fn is qml.gradients.param_shift
        assert spy.call_args[1]["gradient_fn"] is qml.gradients.param_shift

        # if we set the shots to None, backprop can now be used
        circuit(weights, shots=None)
        assert spy.call_args[1]["gradient_fn"] == "backprop"

        # original QNode settings are unaffected
        assert circuit.gradient_fn is qml.gradients.param_shift
        circuit(weights)
        assert spy.call_args[1]["gradient_fn"] is qml.gradients.param_shift


@pytest.mark.parametrize("interface", ["auto", "tf"])
class TestAdjoint:
    """Specific integration tests for the adjoint method"""

    def test_reuse_state(self, interface, mocker):
        """Tests that the TF interface reuses the device state for adjoint differentiation"""
        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, diff_method="adjoint", interface=interface)
        def circ(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=(0, 1))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        spy = mocker.spy(dev, "adjoint_jacobian")

        weights = tf.Variable([0.1, 0.2], dtype=tf.float64)
        x, y = 1.0 * weights

        with tf.GradientTape() as tape:
            res = tf.reduce_sum(circ(weights))

        grad = tape.gradient(res, weights)
        expected_grad = [-tf.sin(x), tf.cos(y)]

        assert np.allclose(grad, expected_grad)
        assert circ.device.num_executions == 1
        spy.assert_called_with(mocker.ANY, use_device_state=mocker.ANY)

    def test_resuse_state_multiple_evals(self, interface, mocker, tol):
        """Tests that the TF interface reuses the device state for adjoint differentiation,
        even where there are intermediate evaluations."""
        dev = qml.device("default.qubit", wires=2)

        x_val = 0.543
        y_val = -0.654
        x = tf.Variable(x_val, dtype=tf.float64)
        y = tf.Variable(y_val, dtype=tf.float64)

        @qnode(dev, diff_method="adjoint", interface=interface)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(dev, "adjoint_jacobian")

        with tf.GradientTape() as tape:
            res1 = circuit(x, y)

        assert np.allclose(res1, np.cos(x_val), atol=tol, rtol=0)

        # intermediate evaluation with different values
        res2 = circuit(tf.math.tan(x), tf.math.cosh(y))

        # the adjoint method will continue to compute the correct derivative
        grad = tape.gradient(res1, x)
        assert np.allclose(grad, -np.sin(x_val), atol=tol, rtol=0)
        assert dev.num_executions == 2
        spy.assert_called_with(mocker.ANY, use_device_state=mocker.ANY)


@pytest.mark.parametrize(
    "interface, dev_name,diff_method,mode", interface_and_qubit_device_and_diff_method
)
class TestQubitIntegration:
    """Tests that ensure various qubit circuits integrate correctly"""

    def test_probability_differentiation(self, interface, dev_name, diff_method, mode, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple probs outputs"""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        dev = qml.device(dev_name, wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, mode=mode, interface=interface)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0]), qml.probs(wires=[1])

        with tf.GradientTape() as tape:
            res = circuit(x, y)

        expected = np.array(
            [
                [tf.cos(x / 2) ** 2, tf.sin(x / 2) ** 2],
                [(1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [x, y])
        expected = np.array(
            [
                [
                    [-tf.sin(x) / 2, tf.sin(x) / 2],
                    [-tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                ],
                [
                    [0, 0],
                    [-tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
                ],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_ragged_differentiation(self, interface, dev_name, diff_method, mode, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        dev = qml.device(dev_name, wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, mode=mode, interface=interface)
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

    def test_second_derivative(self, interface, dev_name, diff_method, mode, tol):
        """Test second derivative calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, mode=mode, max_diff=2, interface=interface)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = tf.Variable([1.0, 2.0], dtype=tf.float64)

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                res = circuit(x)
            g = tape2.gradient(res, x)
            res2 = tf.reduce_sum(g)

        g2 = tape1.gradient(res2, x)
        a, b = x * 1.0

        expected_res = tf.cos(a) * tf.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-tf.sin(a) * tf.cos(b), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        expected_g2 = [
            -tf.cos(a) * tf.cos(b) + tf.sin(a) * tf.sin(b),
            tf.sin(a) * tf.sin(b) - tf.cos(a) * tf.cos(b),
        ]
        assert np.allclose(g2, expected_g2, atol=tol, rtol=0)

    def test_hessian(self, interface, dev_name, diff_method, mode, tol):
        """Test hessian calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, mode=mode, max_diff=2, interface=interface)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = tf.Variable([1.0, 2.0], dtype=tf.float64)

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                res = circuit(x)
            g = tape2.gradient(res, x)

        hess = tape1.jacobian(g, x)
        a, b = x * 1.0

        expected_res = tf.cos(a) * tf.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-tf.sin(a) * tf.cos(b), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        expected_hess = [
            [-tf.cos(a) * tf.cos(b), tf.sin(a) * tf.sin(b)],
            [tf.sin(a) * tf.sin(b), -tf.cos(a) * tf.cos(b)],
        ]
        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_vector_valued(self, interface, dev_name, diff_method, mode, tol):
        """Test hessian calculation of a vector valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, mode=mode, max_diff=2, interface=interface)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.probs(wires=0)

        x = tf.Variable([1.0, 2.0], dtype=tf.float64)

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(x)
            g = tape2.jacobian(res, x, experimental_use_pfor=False)

        hess = tape1.jacobian(g, x)

        a, b = x * 1.0

        expected_res = [
            0.5 + 0.5 * tf.cos(a) * tf.cos(b),
            0.5 - 0.5 * tf.cos(a) * tf.cos(b),
        ]
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [
            [-0.5 * tf.sin(a) * tf.cos(b), -0.5 * tf.cos(a) * tf.sin(b)],
            [0.5 * tf.sin(a) * tf.cos(b), 0.5 * tf.cos(a) * tf.sin(b)],
        ]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        expected_hess = [
            [
                [-0.5 * tf.cos(a) * tf.cos(b), 0.5 * tf.sin(a) * tf.sin(b)],
                [0.5 * tf.sin(a) * tf.sin(b), -0.5 * tf.cos(a) * tf.cos(b)],
            ],
            [
                [0.5 * tf.cos(a) * tf.cos(b), -0.5 * tf.sin(a) * tf.sin(b)],
                [-0.5 * tf.sin(a) * tf.sin(b), 0.5 * tf.cos(a) * tf.cos(b)],
            ],
        ]
        np.testing.assert_allclose(hess, expected_hess, atol=tol, rtol=0, verbose=True)

    def test_hessian_vector_valued_postprocessing(
        self, interface, dev_name, diff_method, mode, tol
    ):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, mode=mode, max_diff=2, interface=interface)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))]

        x = tf.Variable([0.76, -0.87], dtype=tf.float64)

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = tf.tensordot(x, circuit(x), axes=[0, 0])

            g = tape2.jacobian(res, x, experimental_use_pfor=False)

        hess = tape1.jacobian(g, x)
        a, b = x * 1.0

        expected_res = a * tf.cos(a) * tf.cos(b) + b * tf.cos(a) * tf.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [
            tf.cos(b) * (tf.cos(a) - (a + b) * tf.sin(a)),
            tf.cos(a) * (tf.cos(b) - (a + b) * tf.sin(b)),
        ]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        expected_hess = [
            [
                -(tf.cos(b) * ((a + b) * tf.cos(a) + 2 * tf.sin(a))),
                -(tf.cos(b) * tf.sin(a)) + (-tf.cos(a) + (a + b) * tf.sin(a)) * tf.sin(b),
            ],
            [
                -(tf.cos(b) * tf.sin(a)) + (-tf.cos(a) + (a + b) * tf.sin(a)) * tf.sin(b),
                -(tf.cos(a) * ((a + b) * tf.cos(b) + 2 * tf.sin(b))),
            ],
        ]
        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_hessian_ragged(self, interface, dev_name, diff_method, mode, tol):
        """Test hessian calculation of a ragged QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, mode=mode, max_diff=2, interface=interface)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=1)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=1)

        x = tf.Variable([1.0, 2.0], dtype=tf.float64)
        res = circuit(x)

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(x)
            g = tape2.jacobian(res, x, experimental_use_pfor=False)

        hess = tape1.jacobian(g, x)
        a, b = x * 1.0

        expected_res = [
            tf.cos(a) * tf.cos(b),
            0.5 + 0.5 * tf.cos(a) * tf.cos(b),
            0.5 - 0.5 * tf.cos(a) * tf.cos(b),
        ]
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [
            [-tf.sin(a) * tf.cos(b), -tf.cos(a) * tf.sin(b)],
            [-0.5 * tf.sin(a) * tf.cos(b), -0.5 * tf.cos(a) * tf.sin(b)],
            [0.5 * tf.sin(a) * tf.cos(b), 0.5 * tf.cos(a) * tf.sin(b)],
        ]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        expected_hess = [
            [
                [-tf.cos(a) * tf.cos(b), tf.sin(a) * tf.sin(b)],
                [tf.sin(a) * tf.sin(b), -tf.cos(a) * tf.cos(b)],
            ],
            [
                [-0.5 * tf.cos(a) * tf.cos(b), 0.5 * tf.sin(a) * tf.sin(b)],
                [0.5 * tf.sin(a) * tf.sin(b), -0.5 * tf.cos(a) * tf.cos(b)],
            ],
            [
                [0.5 * tf.cos(a) * tf.cos(b), -0.5 * tf.sin(a) * tf.sin(b)],
                [-0.5 * tf.sin(a) * tf.sin(b), 0.5 * tf.cos(a) * tf.cos(b)],
            ],
        ]
        np.testing.assert_allclose(hess, expected_hess, atol=tol, rtol=0, verbose=True)

    def test_state(self, interface, dev_name, diff_method, mode, tol):
        """Test that the state can be returned and differentiated"""
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support states")

        dev = qml.device(dev_name, wires=2)

        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface=interface, mode=mode)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.state()

        def cost_fn(x, y):
            res = circuit(x, y)
            assert res.dtype is tf.complex128
            probs = tf.math.abs(res) ** 2
            return probs[0] + probs[2]

        with tf.GradientTape() as tape:
            res = cost_fn(x, y)

        if diff_method not in {"backprop"}:
            pytest.skip("Test only supports backprop")

        grad = tape.gradient(res, [x, y])
        expected = [-np.sin(x) * np.cos(y) / 2, -np.cos(x) * np.sin(y) / 2]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_projector(self, interface, dev_name, diff_method, mode, tol):
        """Test that the variance of a projector is correctly returned"""
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support projectors")

        dev = qml.device(dev_name, wires=2)
        P = tf.constant([1])

        x, y = 0.765, -0.654
        weights = tf.Variable([x, y], dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface=interface, mode=mode)
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.Projector(P, wires=0) @ qml.PauliX(1))

        with tf.GradientTape() as tape:
            res = circuit(weights)

        expected = 0.25 * np.sin(x / 2) ** 2 * (3 + np.cos(2 * y) + 2 * np.cos(x) * np.sin(y) ** 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = tape.gradient(res, weights)
        expected = [
            0.5 * np.sin(x) * (np.cos(x / 2) ** 2 + np.cos(2 * y) * np.sin(x / 2) ** 2),
            -2 * np.cos(y) * np.sin(x / 2) ** 4 * np.sin(y),
        ]
        assert np.allclose(grad, expected, atol=tol, rtol=0)


@pytest.mark.parametrize(
    "diff_method,kwargs",
    [["finite-diff", {}], ("parameter-shift", {}), ("parameter-shift", {"force_order2": True})],
)
class TestCV:
    """Tests for CV integration"""

    def test_first_order_observable(self, diff_method, kwargs, tol):
        """Test variance of a first order CV observable"""
        dev = qml.device("default.gaussian", wires=1)

        r = tf.Variable(0.543, dtype=tf.float64)
        phi = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, interface="tf", diff_method=diff_method, **kwargs)
        def circuit(r, phi):
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            return qml.var(qml.X(0))

        with tf.GradientTape() as tape:
            res = circuit(r, phi)

        expected = np.exp(2 * r) * np.sin(phi) ** 2 + np.exp(-2 * r) * np.cos(phi) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        grad = tape.gradient(res, [r, phi])
        expected = [
            2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
            2 * np.sinh(2 * r) * np.sin(2 * phi),
        ]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_second_order_observable(self, diff_method, kwargs, tol):
        """Test variance of a second order CV expectation value"""
        dev = qml.device("default.gaussian", wires=1)

        n = tf.Variable(0.12, dtype=tf.float64)
        a = tf.Variable(0.765, dtype=tf.float64)

        @qnode(dev, interface="tf", diff_method=diff_method, **kwargs)
        def circuit(n, a):
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            return qml.var(qml.NumberOperator(0))

        with tf.GradientTape() as tape:
            res = circuit(n, a)

        expected = n**2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # circuit jacobians
        grad = tape.gradient(res, [n, a])
        expected = [2 * a**2 + 2 * n + 1, 2 * a * (2 * n + 1)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("dev_name,diff_method,mode", qubit_device_and_diff_method)
class TestTapeExpansion:
    """Test that tape expansion within the QNode integrates correctly
    with the TF interface"""

    def test_gradient_expansion(self, dev_name, diff_method, mode, mocker):
        """Test that a *supported* operation with no gradient recipe is
        expanded for both parameter-shift and finite-differences, but not for execution."""
        if diff_method not in ("parameter-shift", "finite-diff"):
            pytest.skip("Only supports gradient transforms")

        dev = qml.device(dev_name, wires=1)

        class PhaseShift(qml.PhaseShift):
            grad_method = None

            def expand(self):
                with qml.tape.QuantumTape() as tape:
                    qml.RY(3 * self.data[0], wires=self.wires)
                return tape

        @qnode(dev, diff_method=diff_method, mode=mode, max_diff=2, interface="tf")
        def circuit(x):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            return qml.expval(qml.PauliX(0))

        spy = mocker.spy(circuit.device, "batch_execute")
        x = tf.Variable(0.5, dtype=tf.float64)

        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                loss = circuit(x)

            tape = spy.call_args[0][0][0]

            spy = mocker.spy(circuit.gradient_fn, "transform_fn")
            res = t1.gradient(loss, x)

        input_tape = spy.call_args[0][0]
        assert len(input_tape.operations) == 2
        assert input_tape.operations[1].name == "RY"
        assert input_tape.operations[1].data[0] == 3 * x

        shifted_tape1, shifted_tape2 = spy.spy_return[0]

        assert len(shifted_tape1.operations) == 2
        assert shifted_tape1.operations[1].name == "RY"

        assert len(shifted_tape2.operations) == 2
        assert shifted_tape2.operations[1].name == "RY"

        assert np.allclose(res, -3 * np.sin(3 * x))

        if diff_method == "parameter-shift":
            # test second order derivatives
            res = t2.gradient(res, x)
            assert np.allclose(res, -9 * np.cos(3 * x))

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_gradient_expansion_trainable_only(self, dev_name, diff_method, mode, max_diff, mocker):
        """Test that a *supported* operation with no gradient recipe is only
        expanded for parameter-shift and finite-differences when it is trainable."""
        if diff_method not in ("parameter-shift", "finite-diff"):
            pytest.skip("Only supports gradient transforms")

        dev = qml.device(dev_name, wires=1)

        class PhaseShift(qml.PhaseShift):
            grad_method = None

            def expand(self):
                with qml.tape.QuantumTape() as tape:
                    qml.RY(3 * self.data[0], wires=self.wires)
                return tape

        @qnode(dev, diff_method=diff_method, mode=mode, max_diff=max_diff, interface="tf")
        def circuit(x, y):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            PhaseShift(2 * y, wires=0)
            return qml.expval(qml.PauliX(0))

        spy = mocker.spy(circuit.device, "batch_execute")
        x = tf.Variable(0.5, dtype=tf.float64)
        y = tf.constant(0.7, dtype=tf.float64)

        with tf.GradientTape() as t:
            res = circuit(x, y)

        spy = mocker.spy(circuit.gradient_fn, "transform_fn")
        res = t.gradient(res, [x, y])

        input_tape = spy.call_args[0][0]
        assert len(input_tape.operations) == 3
        assert input_tape.operations[1].name == "RY"
        assert input_tape.operations[1].data[0] == 3 * x
        assert input_tape.operations[2].name == "PhaseShift"
        assert input_tape.operations[2].grad_method is None

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_expansion_analytic(self, dev_name, diff_method, mode, max_diff):
        """Test that if there are non-commuting groups and the number of shots is None
        the first and second order gradients are correctly evaluated"""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not yet support Hamiltonians")

        dev = qml.device(dev_name, wires=3, shots=None)
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(dev, diff_method=diff_method, mode=mode, max_diff=max_diff, interface="tf")
        def circuit(data, weights, coeffs):
            weights = tf.reshape(weights, [1, -1])
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        d = tf.constant([0.1, 0.2], dtype=tf.float64)
        w = tf.Variable([0.654, -0.734], dtype=tf.float64)
        c = tf.Variable([-0.6543, 0.24, 0.54], dtype=tf.float64)

        # test output
        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape() as t1:
                res = circuit(d, w, c)

            expected = c[2] * np.cos(d[1] + w[1]) - c[1] * np.sin(d[0] + w[0]) * np.sin(d[1] + w[1])
            assert np.allclose(res, expected)

            # test gradients
            grad = t1.gradient(res, [d, w, c])

        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[1], expected_w)
        assert np.allclose(grad[2], expected_c)

        # test second-order derivatives
        if diff_method in ("parameter-shift", "backprop") and max_diff == 2:

            grad2_c = t2.jacobian(grad[2], c)
            assert grad2_c is None or np.allclose(grad2_c, 0)

            grad2_w_c = t2.jacobian(grad[1], c)
            expected = [0, -np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]), 0], [
                0,
                -np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]),
                -np.sin(d[1] + w[1]),
            ]
            assert np.allclose(grad2_w_c, expected)

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_expansion_finite_shots(
        self, dev_name, diff_method, mode, max_diff, mocker
    ):
        """Test that the Hamiltonian is expanded if there
        are non-commuting groups and the number of shots is finite
        and the first and second order gradients are correctly evaluated"""
        if diff_method in ("adjoint", "backprop", "finite-diff"):
            pytest.skip("The adjoint and backprop methods do not yet support sampling")

        dev = qml.device(dev_name, wires=3, shots=50000)
        spy = mocker.spy(qml.transforms, "hamiltonian_expand")
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(dev, diff_method=diff_method, mode=mode, max_diff=max_diff, interface="tf")
        def circuit(data, weights, coeffs):
            weights = tf.reshape(weights, [1, -1])
            qml.templates.AngleEmbedding(data, wires=[0, 1])
            qml.templates.BasicEntanglerLayers(weights, wires=[0, 1])
            H = qml.Hamiltonian(coeffs, obs)
            H.compute_grouping()
            return qml.expval(H)

        d = tf.constant([0.1, 0.2], dtype=tf.float64)
        w = tf.Variable([0.654, -0.734], dtype=tf.float64)
        c = tf.Variable([-0.6543, 0.24, 0.54], dtype=tf.float64)

        # test output
        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape() as t1:
                res = circuit(d, w, c)

            expected = c[2] * np.cos(d[1] + w[1]) - c[1] * np.sin(d[0] + w[0]) * np.sin(d[1] + w[1])
            assert np.allclose(res, expected, atol=0.1)
            spy.assert_called()

            # test gradients
            grad = t1.gradient(res, [d, w, c])

        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[1], expected_w, atol=0.1)
        assert np.allclose(grad[2], expected_c, atol=0.1)

        # test second-order derivatives
        if diff_method == "parameter-shift" and max_diff == 2:
            grad2_c = t2.jacobian(grad[2], c)
            assert grad2_c is None

            grad2_w_c = t2.jacobian(grad[1], c)
            expected = [0, -np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]), 0], [
                0,
                -np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]),
                -np.sin(d[1] + w[1]),
            ]
            assert np.allclose(grad2_w_c, expected, atol=0.1)


class TestSample:
    """Tests for the sample integration"""

    def test_sample_dimension(self):
        """Test sampling works as expected"""
        dev = qml.device("default.qubit", wires=2, shots=10)

        @qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return [qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))]

        res = circuit()

        assert res.shape == (2, 10)
        assert isinstance(res, tf.Tensor)

    def test_sampling_expval(self):
        """Test sampling works as expected if combined with expectation values"""
        dev = qml.device("default.qubit", wires=2, shots=10)

        @qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        res = circuit()

        assert len(res) == 2
        assert isinstance(res, tuple)
        assert res[0].shape == (10,)
        assert isinstance(res[0], tf.Tensor)
        assert isinstance(res[1], tf.Tensor)

    def test_sample_combination(self, tol):
        """Test the output of combining expval, var and sample"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3, shots=n_sample)

        @qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit()

        assert len(result) == 3
        assert np.array_equal(result[0].shape, (n_sample,))
        assert isinstance(result[1], tf.Tensor)
        assert isinstance(result[2], tf.Tensor)
        assert result[0].dtype is tf.int64

    def test_single_wire_sample(self, tol):
        """Test the return type and shape of sampling a single wire"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=1, shots=n_sample)

        @qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, tf.Tensor)
        assert np.array_equal(result.shape, (n_sample,))

    def test_multi_wire_sample_regular_shape(self, tol):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3, shots=n_sample)

        @qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, tf.Tensor)
        assert np.array_equal(result.shape, (3, n_sample))
        assert result.dtype == tf.int64


@pytest.mark.parametrize(
    "decorator, interface", [(tf.function, "tf"), (lambda x: x, "tf-autograph")]
)
class TestAutograph:
    """Tests for Autograph mode. This class is parametrized over the combination:

    1. interface="tf" with the QNode decoratored with @tf.function, and
    2. interface="tf-autograph" with no QNode decorator.

    Option (1) checks that if the user enables autograph functionality
    in TensorFlow, the new `tf-autograph` interface is automatically applied.

    Option (2) ensures that the tf-autograph interface can be manually applied,
    even if in eager execution mode.
    """

    def test_autograph_gradients(self, decorator, interface, tol):
        """Test that a parameter-shift QNode can be compiled
        using @tf.function, and differentiated"""
        dev = qml.device("default.qubit", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(dev, diff_method="parameter-shift", interface=interface)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0]), qml.probs(wires=[1])

        with tf.GradientTape() as tape:
            p0, p1 = circuit(x, y)
            loss = p0[0] + p1[1]

        expected = tf.cos(x / 2) ** 2 + (1 - tf.cos(x) * tf.cos(y)) / 2
        assert np.allclose(loss, expected, atol=tol, rtol=0)

        grad = tape.gradient(loss, [x, y])
        expected = [-tf.sin(x) * tf.sin(y / 2) ** 2, tf.cos(x) * tf.sin(y) / 2]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_autograph_jacobian(self, decorator, interface, tol):
        """Test that a parameter-shift vector-valued QNode can be compiled
        using @tf.function, and differentiated"""
        dev = qml.device("default.qubit", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(dev, diff_method="parameter-shift", max_diff=1, interface=interface)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0]), qml.probs(wires=[1])

        with tf.GradientTape() as tape:
            res = circuit(x, y)

        expected = np.array(
            [
                [tf.cos(x / 2) ** 2, tf.sin(x / 2) ** 2],
                [(1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [x, y])
        expected = np.array(
            [
                [
                    [-tf.sin(x) / 2, tf.sin(x) / 2],
                    [-tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                ],
                [
                    [0, 0],
                    [-tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
                ],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("mode", ["forward", "backward"])
    def test_autograph_adjoint(self, mode, decorator, interface, tol):
        """Test that a parameter-shift vQNode can be compiled
        using @tf.function, and differentiated to second order"""
        dev = qml.device("default.qubit", wires=1)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(dev, diff_method="adjoint", interface=interface, mode=mode)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = tf.Variable([1.0, 2.0], dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(x)
        g = tape.gradient(res, x)
        a, b = x * 1.0

        expected_res = tf.cos(a) * tf.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-tf.sin(a) * tf.cos(b), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

    def test_autograph_hessian(self, decorator, interface, tol):
        """Test that a parameter-shift vQNode can be compiled
        using @tf.function, and differentiated to second order"""
        dev = qml.device("default.qubit", wires=1)
        a = tf.Variable(0.543, dtype=tf.float64)
        b = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(dev, diff_method="parameter-shift", max_diff=2, interface=interface)
        def circuit(x, y):
            qml.RY(x, wires=0)
            qml.RX(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                res = circuit(a, b)
            g = tape2.gradient(res, [a, b])
            g = tf.stack(g)

        hess = tf.stack(tape1.gradient(g, [a, b]))

        expected_res = tf.cos(a) * tf.cos(b)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-tf.sin(a) * tf.cos(b), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

        expected_hess = [
            [-tf.cos(a) * tf.cos(b) + tf.sin(a) * tf.sin(b)],
            [tf.sin(a) * tf.sin(b) - tf.cos(a) * tf.cos(b)],
        ]
        assert np.allclose(hess, expected_hess, atol=tol, rtol=0)

    def test_autograph_state(self, decorator, interface, tol):
        """Test that a parameter-shift QNode returning a state can be compiled
        using @tf.function"""
        dev = qml.device("default.qubit", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(dev, diff_method="parameter-shift", interface=interface)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.state()

        with tf.GradientTape() as tape:
            state = circuit(x, y)
            probs = tf.abs(state) ** 2
            loss = probs[0]

        expected = tf.cos(x / 2) ** 2 * tf.cos(y / 2) ** 2
        assert np.allclose(loss, expected, atol=tol, rtol=0)

    def test_autograph_dimension(self, decorator, interface, tol):
        """Test sampling works as expected"""
        dev = qml.device("default.qubit", wires=2, shots=10)

        @decorator
        @qnode(dev, diff_method="parameter-shift", interface=interface)
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return [qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))]

        res = circuit()

        assert res.shape == (2, 10)
        assert isinstance(res, tf.Tensor)
