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
# pylint: disable=too-many-arguments,too-few-public-methods
import pytest
import numpy as np

import pennylane as qml
from pennylane import qnode

pytestmark = pytest.mark.tf
tf = pytest.importorskip("tensorflow")


qubit_device_and_diff_method = [
    ["default.qubit.legacy", "finite-diff", False],
    ["default.qubit.legacy", "parameter-shift", False],
    ["default.qubit.legacy", "backprop", True],
    ["default.qubit.legacy", "adjoint", True],
    ["default.qubit.legacy", "adjoint", False],
    ["default.qubit.legacy", "spsa", False],
    ["default.qubit.legacy", "hadamard", False],
]

TOL_FOR_SPSA = 1.0
SEED_FOR_SPSA = 32651
H_FOR_SPSA = 0.01

interface_and_qubit_device_and_diff_method = [
    ["auto"] + inner_list for inner_list in qubit_device_and_diff_method
] + [["tf"] + inner_list for inner_list in qubit_device_and_diff_method]


@pytest.mark.parametrize(
    "interface,dev_name,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestQNode:
    """Test that using the QNode with TensorFlow integrates with the PennyLane stack"""

    def test_execution_with_interface(self, dev_name, diff_method, grad_on_execution, interface):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
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

    def test_interface_swap(self, dev_name, diff_method, grad_on_execution, tol, interface):
        """Test that the TF interface can be applied to a QNode
        with a pre-existing interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface="autograd", diff_method=diff_method, grad_on_execution=grad_on_execution
        )
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

    def test_drawing(self, dev_name, diff_method, grad_on_execution, interface):
        """Test circuit drawing when using the TF interface"""

        x = tf.Variable(0.1, dtype=tf.float64)
        y = tf.Variable([0.2, 0.3], dtype=tf.float64)
        z = tf.Variable(0.4, dtype=tf.float64)

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        result = qml.draw(circuit)(p1=x, p3=z)
        expected = "0: ──RX(0.10)──RX(0.40)─╭●─┤  State\n1: ──RY(0.06)───────────╰X─┤  State"
        assert result == expected

    def test_jacobian(self, dev_name, diff_method, grad_on_execution, tol, interface):
        """Test jacobian calculation"""
        kwargs = dict(
            diff_method=diff_method, grad_on_execution=grad_on_execution, interface=interface
        )

        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        @qnode(dev, **kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        with tf.GradientTape() as tape:
            res = circuit(a, b)
            res = tf.stack(res)

        assert circuit.qtape.trainable_params == [0, 1]

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [a, b])
        expected = [[-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_dtype(self, dev_name, diff_method, grad_on_execution, interface):
        """Test calculating the jacobian with a different datatype"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        a = tf.Variable(0.1, dtype=tf.float32)
        b = tf.Variable(0.2, dtype=tf.float32)

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires, r_dtype=np.float32)

        @qnode(
            dev, diff_method=diff_method, grad_on_execution=grad_on_execution, interface=interface
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        with tf.GradientTape() as tape:
            res = circuit(a, b)
            res = tf.stack(res)

        assert circuit.qtape.trainable_params == [0, 1]

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)
        assert res.dtype is tf.float32

        res = tape.jacobian(res, [a, b])
        assert [r.dtype is tf.float32 for r in res]

    def test_jacobian_options(self, dev_name, diff_method, grad_on_execution, interface):
        """Test setting finite-difference jacobian options"""
        if diff_method not in {"finite-diff", "spsa"}:
            pytest.skip("Test only works with finite diff and spsa.")

        a = tf.Variable([0.1, 0.2])

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev,
            interface=interface,
            h=1e-8,
            approx_order=2,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(a)

        tape.jacobian(res, a)

    def test_changing_trainability(self, dev_name, diff_method, grad_on_execution, tol, interface):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method in ["backprop", "adjoint", "spsa"]:
            pytest.skip("Test does not support backprop, adjoint or spsa method")

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        num_wires = 2

        diff_kwargs = {}
        if diff_method == "hadamard":
            num_wires = 3
        elif diff_method == "finite-diff":
            diff_kwargs = {"approx_order": 2, "strategy": "center"}

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            **diff_kwargs,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        with tf.GradientTape() as tape:
            res = circuit(a, b)
            res = tf.stack(res)

        # the tape has reported both gate arguments as trainable
        assert circuit.qtape.trainable_params == [0, 1]

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac = tape.jacobian(res, [a, b])
        expected = [
            [-tf.sin(a), tf.sin(a) * tf.sin(b)],
            [0, -tf.cos(a) * tf.cos(b)],
        ]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        # make the second QNode argument a constant
        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.constant(0.8, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, b)
            res = tf.stack(res)

        # the tape has reported only the first argument as trainable
        assert circuit.qtape.trainable_params == [0]

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac = tape.jacobian(res, a)
        expected = [-tf.sin(a), tf.sin(a) * tf.sin(b)]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

    def test_classical_processing(self, dev_name, diff_method, grad_on_execution, interface):
        """Test classical processing within the quantum tape"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.constant(0.2, dtype=tf.float64)
        c = tf.Variable(0.3, dtype=tf.float64)

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, diff_method=diff_method, grad_on_execution=grad_on_execution, interface=interface
        )
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

    def test_no_trainable_parameters(self, dev_name, diff_method, grad_on_execution, interface):
        """Test evaluation if there are no trainable parameters"""
        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, diff_method=diff_method, grad_on_execution=grad_on_execution, interface=interface
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = 0.1
        b = tf.constant(0.2, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, b)
            res = tf.stack(res)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == []

        assert res.shape == (2,)
        assert isinstance(res, tf.Tensor)

        # can't take the gradient with respect to "a" since it's a Python scalar
        grad = tape.jacobian(res, b)
        assert grad is None

    @pytest.mark.parametrize("U", [tf.constant([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])])
    def test_matrix_parameter(self, dev_name, diff_method, grad_on_execution, U, tol, interface):
        """Test that the TF interface works correctly
        with a matrix parameter"""
        a = tf.Variable(0.1, dtype=tf.float64)

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, diff_method=diff_method, grad_on_execution=grad_on_execution, interface=interface
        )
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

    def test_differentiable_expand(self, dev_name, diff_method, grad_on_execution, tol, interface):
        """Test that operation and nested tapes expansion
        is differentiable"""
        kwargs = dict(
            diff_method=diff_method, grad_on_execution=grad_on_execution, interface=interface
        )
        if diff_method == "spsa":
            spsa_kwargs = dict(sampler_rng=np.random.default_rng(SEED_FOR_SPSA), num_directions=10)
            kwargs = {**kwargs, **spsa_kwargs}
            tol = TOL_FOR_SPSA

        class U3(qml.U3):
            def decomposition(self):
                theta, phi, lam = self.data
                wires = self.wires
                return [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        a = np.array(0.1)
        p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)

        @qnode(dev, **kwargs)
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        with tf.GradientTape() as tape:
            res = circuit(a, p)

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

    def test_changing_shots(self, mocker, tol, interface):
        """Test that changing shots works on execution"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=None)
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
        circuit(weights, shots=100)  # pylint: disable=unexpected-keyword-arg
        spy.assert_called_once()
        assert spy.spy_return.shape == (100,)

        # device state has been unaffected
        assert dev.shots is None
        res = circuit(weights)
        assert np.allclose(res, -np.cos(a) * np.sin(b), atol=tol, rtol=0)
        spy.assert_called_once()

    def test_gradient_integration(self, interface):
        """Test that temporarily setting the shots works
        for gradient computations"""
        # pylint: disable=unexpected-keyword-arg
        dev = qml.device("default.qubit.legacy", wires=2, shots=None)
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

    def test_multiple_gradient_integration(self, tol, interface):
        """Test that temporarily setting the shots works
        for gradient computations, even if the QNode has been re-evaluated
        with a different number of shots in the meantime."""
        dev = qml.device("default.qubit.legacy", wires=2, shots=None)
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

        res2 = circuit(weights, shots=[(1, 1000)])  # pylint: disable=unexpected-keyword-arg
        assert qml.math.shape(res2) == (1000,)

        grad = tape.gradient(res1, weights)
        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_update_diff_method(self, mocker, interface):
        """Test that temporarily setting the shots updates the diff method"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=100)
        weights = tf.Variable([0.543, -0.654], dtype=tf.float64)

        spy = mocker.spy(qml, "execute")

        @qnode(dev, interface=interface)
        def circuit(weights):
            qml.RY(weights[0], wires=0)
            qml.RX(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        # since we are using finite shots, parameter-shift will
        # be chosen
        circuit(weights)
        assert circuit.gradient_fn is qml.gradients.param_shift

        circuit(weights)
        assert spy.call_args[1]["gradient_fn"] is qml.gradients.param_shift
        assert circuit.gradient_fn is qml.gradients.param_shift

        # if we set the shots to None, backprop can now be used
        circuit(weights, shots=None)  # pylint: disable=unexpected-keyword-arg
        assert spy.call_args[1]["gradient_fn"] == "backprop"
        assert circuit.gradient_fn == "backprop"

        circuit(weights)
        assert circuit.gradient_fn is qml.gradients.param_shift
        assert spy.call_args[1]["gradient_fn"] is qml.gradients.param_shift


@pytest.mark.parametrize("interface", ["auto", "tf"])
class TestAdjoint:
    """Specific integration tests for the adjoint method"""

    def test_reuse_state(self, mocker, interface):
        """Tests that the TF interface reuses the device state for adjoint differentiation"""
        dev = qml.device("default.qubit.legacy", wires=2)

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

    def test_reuse_state_multiple_evals(self, mocker, tol, interface):
        """Tests that the TF interface reuses the device state for adjoint differentiation,
        even where there are intermediate evaluations."""
        dev = qml.device("default.qubit.legacy", wires=2)

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
        circuit(tf.math.tan(x), tf.math.cosh(y))

        # the adjoint method will continue to compute the correct derivative
        grad = tape.gradient(res1, x)
        assert np.allclose(grad, -np.sin(x_val), atol=tol, rtol=0)
        assert dev.num_executions == 2
        spy.assert_called_with(mocker.ANY, use_device_state=mocker.ANY)


@pytest.mark.parametrize(
    "interface,dev_name,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestQubitIntegration:
    """Tests that ensure various qubit circuits integrate correctly"""

    def test_probability_differentiation(
        self, dev_name, diff_method, grad_on_execution, tol, interface
    ):
        """Tests correct output shape and evaluation for a tape
        with multiple probs outputs"""
        kwargs = dict(
            diff_method=diff_method, grad_on_execution=grad_on_execution, interface=interface
        )
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")
        elif diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, **kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0]), qml.probs(wires=[1])

        with tf.GradientTape() as tape:
            res = circuit(x, y)
            res = tf.stack(res)

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

    def test_ragged_differentiation(self, dev_name, diff_method, grad_on_execution, tol, interface):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        kwargs = dict(
            diff_method=diff_method, grad_on_execution=grad_on_execution, interface=interface
        )
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")
        elif diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, **kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        with tf.GradientTape() as tape:
            res = circuit(x, y)
            res = tf.experimental.numpy.hstack(res)

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

    def test_second_derivative(self, dev_name, diff_method, grad_on_execution, tol, interface):
        """Test second derivative calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop", "hadamard"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
        )
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

    def test_hessian(self, dev_name, diff_method, grad_on_execution, tol, interface):
        """Test hessian calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop", "hadamard"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
        )
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

    def test_hessian_vector_valued(self, dev_name, diff_method, grad_on_execution, tol, interface):
        """Test hessian calculation of a vector valued QNode"""
        if diff_method not in {"parameter-shift", "backprop", "hadamard"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
        )
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
        self, dev_name, diff_method, grad_on_execution, tol, interface
    ):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        if diff_method not in {"parameter-shift", "backprop", "hadamard"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
        )
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

    def test_hessian_ragged(self, dev_name, diff_method, grad_on_execution, tol, interface):
        """Test hessian calculation of a ragged QNode"""
        if diff_method not in {"parameter-shift", "backprop", "hadamard"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 4

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
        )
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
                res = tf.experimental.numpy.hstack(res)
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

    def test_state(self, dev_name, diff_method, grad_on_execution, tol, interface):
        """Test that the state can be returned and differentiated"""
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support states")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(
            dev, diff_method=diff_method, interface=interface, grad_on_execution=grad_on_execution
        )
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

    @pytest.mark.parametrize("state", [[1], [0, 1]])  # Basis state and state vector
    def test_projector(self, state, dev_name, diff_method, grad_on_execution, tol, interface):
        """Test that the variance of a projector is correctly returned"""
        kwargs = dict(
            diff_method=diff_method, grad_on_execution=grad_on_execution, interface=interface
        )
        if diff_method == "adjoint":
            pytest.skip("Adjoint does not support projectors")
        elif diff_method == "hadamard":
            pytest.skip("Variance not implemented yet.")
        elif diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        dev = qml.device(dev_name, wires=2)
        P = tf.constant(state)

        x, y = 0.765, -0.654
        weights = tf.Variable([x, y], dtype=tf.float64)

        @qnode(dev, **kwargs)
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
    [
        ["finite-diff", {}],
        ["spsa", {"num_directions": 100, "h": 0.05}],
        ("parameter-shift", {}),
        ("parameter-shift", {"force_order2": True}),
    ],
)
class TestCV:
    """Tests for CV integration"""

    def test_first_order_observable(self, diff_method, kwargs, tol):
        """Test variance of a first order CV observable"""
        dev = qml.device("default.gaussian", wires=1)
        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        r = tf.Variable(0.543, dtype=tf.float64)
        phi = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, **kwargs)
        def circuit(r, phi):
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            return qml.var(qml.QuadX(0))

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
        if diff_method == "spsa":
            kwargs["sampler_rng"] = np.random.default_rng(SEED_FOR_SPSA)
            tol = TOL_FOR_SPSA

        n = tf.Variable(0.12, dtype=tf.float64)
        a = tf.Variable(0.765, dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, **kwargs)
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


@pytest.mark.parametrize(
    "interface,dev_name,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestTapeExpansion:
    """Test that tape expansion within the QNode integrates correctly
    with the TF interface"""

    def test_gradient_expansion(self, dev_name, diff_method, grad_on_execution, interface):
        """Test that a *supported* operation with no gradient recipe is
        expanded for both parameter-shift and finite-differences, but not for execution."""
        if diff_method not in ("parameter-shift", "finite-diff", "spsa", "hadamard"):
            pytest.skip("Only supports gradient transforms")
        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        class PhaseShift(qml.PhaseShift):
            grad_method = None

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
        )
        def circuit(x):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            return qml.expval(qml.PauliX(0))

        x = tf.Variable(0.5, dtype=tf.float64)

        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                loss = circuit(x)

            res = t1.gradient(loss, x)

        assert np.allclose(res, -3 * np.sin(3 * x))

        if diff_method == "parameter-shift":
            # test second order derivatives
            res = t2.gradient(res, x)
            assert np.allclose(res, -9 * np.cos(3 * x))

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_gradient_expansion_trainable_only(
        self, dev_name, diff_method, grad_on_execution, max_diff, interface
    ):
        """Test that a *supported* operation with no gradient recipe is only
        expanded for parameter-shift and finite-differences when it is trainable."""
        if diff_method not in ("parameter-shift", "finite-diff", "spsa", "hadamard"):
            pytest.skip("Only supports gradient transforms")

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        class PhaseShift(qml.PhaseShift):
            grad_method = None

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            interface=interface,
        )
        def circuit(x, y):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            PhaseShift(2 * y, wires=0)
            return qml.expval(qml.PauliX(0))

        x = tf.Variable(0.5, dtype=tf.float64)
        y = tf.constant(0.7, dtype=tf.float64)

        with tf.GradientTape() as t:
            res = circuit(x, y)

        t.gradient(res, [x, y])

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_expansion_analytic(
        self, dev_name, diff_method, grad_on_execution, max_diff, tol, interface
    ):
        """Test that if there are non-commuting groups and the number of shots is None
        the first and second order gradients are correctly evaluated"""
        kwargs = dict(
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            interface=interface,
        )
        if diff_method in ["adjoint", "hadamard"]:
            pytest.skip("The adjoint/hadamard method does not yet support Hamiltonians")
        elif diff_method == "spsa":
            tol = TOL_FOR_SPSA
            spsa_kwargs = dict(sampler_rng=np.random.default_rng(SEED_FOR_SPSA), num_directions=10)
            kwargs = {**kwargs, **spsa_kwargs}

        dev = qml.device(dev_name, wires=3, shots=None)
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(dev, **kwargs)
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
            assert np.allclose(res, expected, atol=tol)

            # test gradients
            grad = t1.gradient(res, [d, w, c])

        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[1], expected_w, atol=tol)
        assert np.allclose(grad[2], expected_c, atol=tol)

        # test second-order derivatives
        if diff_method in ("parameter-shift", "backprop") and max_diff == 2:
            grad2_c = t2.jacobian(grad[2], c)
            assert grad2_c is None or np.allclose(grad2_c, 0, atol=tol)

            grad2_w_c = t2.jacobian(grad[1], c)
            expected = [0, -np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]), 0], [
                0,
                -np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]),
                -np.sin(d[1] + w[1]),
            ]
            assert np.allclose(grad2_w_c, expected, atol=tol)

    @pytest.mark.parametrize("max_diff", [1, 2])
    def test_hamiltonian_expansion_finite_shots(
        self, dev_name, diff_method, grad_on_execution, max_diff, mocker, interface
    ):
        """Test that the Hamiltonian is expanded if there
        are non-commuting groups and the number of shots is finite
        and the first and second order gradients are correctly evaluated"""
        gradient_kwargs = {}
        tol = 0.3
        if diff_method in ("adjoint", "backprop", "hadamard"):
            pytest.skip("The adjoint and backprop methods do not yet support sampling")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "sampler_rng": np.random.default_rng(SEED_FOR_SPSA),
                "num_directions": 20,
            }
            tol = TOL_FOR_SPSA
        elif diff_method == "finite-diff":
            gradient_kwargs = {"h": 0.05}

        dev = qml.device(dev_name, wires=3, shots=50000)
        spy = mocker.spy(qml.transforms, "hamiltonian_expand")
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            interface=interface,
            **gradient_kwargs,
        )
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
            assert np.allclose(res, expected, atol=tol)
            spy.assert_called()

            # test gradients
            grad = t1.gradient(res, [d, w, c])

        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[1], expected_w, atol=tol)
        assert np.allclose(grad[2], expected_c, atol=tol)

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
            assert np.allclose(grad2_w_c, expected, atol=tol)


class TestSample:
    """Tests for the sample integration"""

    def test_sample_dimension(self):
        """Test sampling works as expected"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=10)

        @qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        res = circuit()

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].shape == (10,)
        assert res[1].shape == (10,)
        assert isinstance(res[0], tf.Tensor)
        assert isinstance(res[1], tf.Tensor)

    def test_sampling_expval(self):
        """Test sampling works as expected if combined with expectation values"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=10)

        @qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        res = circuit()

        assert len(res) == 2
        assert isinstance(res, tuple)
        assert res[0].shape == (10,)
        assert res[1].shape == ()
        assert isinstance(res[0], tf.Tensor)
        assert isinstance(res[1], tf.Tensor)

    def test_sample_combination(self):
        """Test the output of combining expval, var and sample"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=3, shots=n_sample)

        @qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit()

        assert len(result) == 3
        assert result[0].shape == (n_sample,)
        assert result[1].shape == ()
        assert result[2].shape == ()
        assert isinstance(result[0], tf.Tensor)
        assert isinstance(result[1], tf.Tensor)
        assert isinstance(result[2], tf.Tensor)
        assert result[0].dtype is tf.int64
        assert result[1].dtype is tf.float64
        assert result[2].dtype is tf.float64

    def test_single_wire_sample(self):
        """Test the return type and shape of sampling a single wire"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=1, shots=n_sample)

        @qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, tf.Tensor)
        assert np.array_equal(result.shape, (n_sample,))

    def test_multi_wire_sample_regular_shape(self):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=3, shots=n_sample)

        @qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()
        result = tf.stack(result)

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, tf.Tensor)
        assert np.array_equal(result.shape, (3, n_sample))
        assert result.dtype == tf.int64

    def test_counts(self):
        """Test counts works as expected for TF"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=100)

        @qnode(dev, interface="tf")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.counts(qml.PauliZ(0))

        res = circuit()

        assert isinstance(res, dict)
        assert list(res.keys()) == [-1, 1]
        assert isinstance(res[-1], tf.Tensor)
        assert isinstance(res[1], tf.Tensor)
        assert res[-1].shape == ()
        assert res[1].shape == ()


@pytest.mark.parametrize(
    "decorator, interface",
    [(tf.function, "auto"), (tf.function, "tf"), (lambda x: x, "tf-autograph")],
)
class TestAutograph:
    """Tests for Autograph mode. This class is parametrized over the combination:

    1. interface=interface with the QNode decoratored with @tf.function, and
    2. interface="tf-autograph" with no QNode decorator.

    Option (1) checks that if the user enables autograph functionality
    in TensorFlow, the new `tf-autograph` interface is automatically applied.

    Option (2) ensures that the tf-autograph interface can be manually applied,
    even if in eager execution mode.
    """

    def test_autograph_gradients(self, decorator, interface, tol):
        """Test that a parameter-shift QNode can be compiled
        using @tf.function, and differentiated"""
        dev = qml.device("default.qubit.legacy", wires=2)
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
        dev = qml.device("default.qubit.legacy", wires=2)
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
            res = tf.stack(res)

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

    @pytest.mark.parametrize("grad_on_execution", [True, False])
    def test_autograph_adjoint_single_param(self, grad_on_execution, decorator, interface, tol):
        """Test that a parameter-shift QNode can be compiled
        using @tf.function, and differentiated to second order"""
        dev = qml.device("default.qubit.legacy", wires=1)

        @decorator
        @qnode(dev, diff_method="adjoint", interface=interface, grad_on_execution=grad_on_execution)
        def circuit(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = tf.Variable(1.0, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(x)
        g = tape.gradient(res, x)

        expected_res = tf.cos(x)
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = -tf.sin(x)
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

    @pytest.mark.parametrize("grad_on_execution", [True, False])
    def test_autograph_adjoint_multi_params(self, grad_on_execution, decorator, interface, tol):
        """Test that a parameter-shift QNode can be compiled
        using @tf.function, and differentiated to second order"""
        dev = qml.device("default.qubit.legacy", wires=1)

        @decorator
        @qnode(dev, diff_method="adjoint", interface=interface, grad_on_execution=grad_on_execution)
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

    def test_autograph_ragged_differentiation(self, decorator, interface, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit.legacy", wires=2)

        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(dev, diff_method="parameter-shift", max_diff=1, interface=interface)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        with tf.GradientTape() as tape:
            res = circuit(x, y)
            res = tf.experimental.numpy.hstack(res)

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

    def test_autograph_hessian(self, decorator, interface, tol):
        """Test that a parameter-shift QNode can be compiled
        using @tf.function, and differentiated to second order"""
        dev = qml.device("default.qubit.legacy", wires=1)
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
        dev = qml.device("default.qubit.legacy", wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        # TODO: fix this for diff_method=None
        @decorator
        @qnode(dev, diff_method="parameter-shift", interface=interface)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.state()

        with tf.GradientTape():
            state = circuit(x, y)
            probs = tf.abs(state) ** 2
            loss = probs[0]

        expected = tf.cos(x / 2) ** 2 * tf.cos(y / 2) ** 2
        assert np.allclose(loss, expected, atol=tol, rtol=0)

    def test_autograph_dimension(self, decorator, interface):
        """Test sampling works as expected"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=10)

        @decorator
        @qnode(dev, diff_method="parameter-shift", interface=interface)
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        res = circuit()
        res = tf.stack(res)

        assert res.shape == (2, 10)
        assert isinstance(res, tf.Tensor)


@pytest.mark.parametrize(
    "interface,dev_name,diff_method,grad_on_execution", interface_and_qubit_device_and_diff_method
)
class TestReturn:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types."""

    def test_grad_single_measurement_param(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """For one measurement and one param, the gradient is a float."""
        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(0.1)

        with tf.GradientTape() as tape:
            res = circuit(a)

        grad = tape.gradient(res, a)

        assert isinstance(grad, tf.Tensor)
        assert grad.shape == ()

    def test_grad_single_measurement_multiple_param(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(0.1)
        b = tf.Variable(0.2)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        grad = tape.gradient(res, (a, b))

        assert isinstance(grad, tuple)
        assert len(grad) == 2
        assert grad[0].shape == ()
        assert grad[1].shape == ()

    def test_grad_single_measurement_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""

        num_wires = 1

        if diff_method == "hadamard":
            num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable([0.1, 0.2])

        with tf.GradientTape() as tape:
            res = circuit(a)

        grad = tape.gradient(res, a)

        assert isinstance(grad, tf.Tensor)
        assert len(grad) == 2
        assert grad.shape == (2,)

    def test_jacobian_single_measurement_param_probs(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable(0.1)

        with tf.GradientTape() as tape:
            res = circuit(a)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable(0.1)
        b = tf.Variable(0.2)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        jac = tape.jacobian(res, (a, b))

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], tf.Tensor)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], tf.Tensor)
        assert jac[1].shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable([0.1, 0.2])

        with tf.GradientTape() as tape:
            res = circuit(a)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (4, 2)

    def test_jacobian_multiple_measurement_single_param(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """The jacobian of multiple measurements with a single params return an array."""
        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable(0.1)

        with tf.GradientTape() as tape:
            res = circuit(a)
            res = tf.experimental.numpy.hstack(res)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (5,)

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 3

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable(0.1)
        b = tf.Variable(0.2)

        with tf.GradientTape() as tape:
            res = circuit(a, b)
            res = tf.experimental.numpy.hstack(res)

        jac = tape.jacobian(res, (a, b))

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tf.Tensor)
        assert jac[0].shape == (5,)

        assert isinstance(jac[1], tf.Tensor)
        assert jac[1].shape == (5,)

    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because of probabilities.")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 4

        dev = qml.device(dev_name, wires=num_wires)

        @qnode(
            dev, interface=interface, diff_method=diff_method, grad_on_execution=grad_on_execution
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable([0.1, 0.2])

        with tf.GradientTape() as tape:
            res = circuit(a)
            res = tf.experimental.numpy.hstack(res)

        jac = tape.jacobian(res, a)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (5, 2)

    def test_hessian_expval_multiple_params(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 4

        dev = qml.device(dev_name, wires=num_wires)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = tf.Variable(0.1, dtype=tf.float64)
        par_1 = tf.Variable(0.2, dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                res = circuit(par_0, par_1)

            grad = tape2.gradient(res, (par_0, par_1))
            grad = tf.stack(grad)

        hess = tape1.jacobian(grad, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tf.Tensor)
        assert hess[0].shape == (2,)

        assert isinstance(hess[1], tf.Tensor)
        assert hess[1].shape == (2,)

    def test_hessian_expval_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """The hessian of single measurement with a multiple params array return a single array."""

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        num_wires = 2

        if diff_method == "hadamard":
            num_wires = 4

        dev = qml.device(dev_name, wires=num_wires)

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                res = circuit(params)

            grad = tape2.gradient(res, params)

        hess = tape1.jacobian(grad, params)

        assert isinstance(hess, tf.Tensor)
        assert hess.shape == (2, 2)

    def test_hessian_var_multiple_params(self, dev_name, diff_method, grad_on_execution, interface):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        if diff_method == "hadamard":
            pytest.skip("Test does not support hadamard because of variance.")

        par_0 = tf.Variable(0.1, dtype=tf.float64)
        par_1 = tf.Variable(0.2, dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                res = circuit(par_0, par_1)

            grad = tape2.gradient(res, (par_0, par_1))
            grad = tf.stack(grad)

        hess = tape1.jacobian(grad, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tf.Tensor)
        assert hess[0].shape == (2,)

        assert isinstance(hess[1], tf.Tensor)
        assert hess[1].shape == (2,)

    def test_hessian_var_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        if diff_method == "hadamard":
            pytest.skip("Test does not support hadamard because of variance.")

        dev = qml.device(dev_name, wires=2)

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1))

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                res = circuit(params)

            grad = tape2.gradient(res, params)

        hess = tape1.jacobian(grad, params)

        assert isinstance(hess, tf.Tensor)
        assert hess.shape == (2, 2)

    def test_hessian_probs_expval_multiple_params(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        num_wires = 2

        if diff_method == "hadamard":
            pytest.skip("Test does not support hadamard because multiple measurements.")

        dev = qml.device(dev_name, wires=num_wires)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        par_0 = tf.Variable(0.1, dtype=tf.float64)
        par_1 = tf.Variable(0.2, dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(par_0, par_1)
                res = tf.experimental.numpy.hstack(res)

            grad = tape2.jacobian(res, (par_0, par_1), experimental_use_pfor=False)
            grad = tf.concat(grad, 0)

        hess = tape1.jacobian(grad, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tf.Tensor)
        assert hess[0].shape == (6,)

        assert isinstance(hess[1], tf.Tensor)
        assert hess[1].shape == (6,)

    def test_hessian_probs_expval_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        if diff_method == "hadamard":
            pytest.skip("Test does not support hadamard because multiple measurements.")

        num_wires = 2

        dev = qml.device(dev_name, wires=num_wires)

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(params)
                res = tf.experimental.numpy.hstack(res)

            grad = tape2.jacobian(res, params, experimental_use_pfor=False)

        hess = tape1.jacobian(grad, params)

        assert isinstance(hess, tf.Tensor)
        assert hess.shape == (3, 2, 2)

    def test_hessian_probs_var_multiple_params(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
        dev = qml.device(dev_name, wires=2)

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        if diff_method == "hadamard":
            pytest.skip("Test does not support hadamard because of variance.")

        par_0 = tf.Variable(0.1, dtype=tf.float64)
        par_1 = tf.Variable(0.2, dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(par_0, par_1)
                res = tf.experimental.numpy.hstack(res)

            grad = tape2.jacobian(res, (par_0, par_1), experimental_use_pfor=False)
            grad = tf.concat(grad, 0)

        hess = tape1.jacobian(grad, (par_0, par_1))

        assert isinstance(hess, tuple)
        assert len(hess) == 2

        assert isinstance(hess[0], tf.Tensor)
        assert hess[0].shape == (6,)

        assert isinstance(hess[1], tf.Tensor)
        assert hess[1].shape == (6,)

    def test_hessian_probs_var_multiple_param_array(
        self, dev_name, diff_method, grad_on_execution, interface
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        if diff_method == "hadamard":
            pytest.skip("Test does not support hadamard because of variance.")

        dev = qml.device(dev_name, wires=2)

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x):
            qml.RX(x[0], wires=[0])
            qml.RY(x[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliZ(0) @ qml.PauliX(1)), qml.probs(wires=[0])

        with tf.GradientTape() as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(params)
                res = tf.experimental.numpy.hstack(res)

            grad = tape2.jacobian(res, params, experimental_use_pfor=False)

        hess = tape1.jacobian(grad, params)

        assert isinstance(hess, tf.Tensor)
        assert hess.shape == (3, 2, 2)


@pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
def test_no_ops(dev_name):
    """Test that the return value of the QNode matches in the interface
    even if there are no ops"""

    dev = qml.device(dev_name, wires=1)

    @qml.qnode(dev, interface="tf")
    def circuit():
        qml.Hadamard(wires=0)
        return qml.state()

    res = circuit()
    assert isinstance(res, tf.Tensor)
