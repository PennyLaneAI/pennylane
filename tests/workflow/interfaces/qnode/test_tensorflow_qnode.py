# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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

# pylint: disable=too-many-arguments,too-few-public-methods,comparison-with-callable, use-implicit-booleaness-not-comparison
import pytest

import pennylane as qml
from pennylane import qnode
from pennylane.devices import DefaultQubit

pytestmark = pytest.mark.tf
tf = pytest.importorskip("tensorflow")

# device, diff_method, grad_on_execution, device_vjp
qubit_device_and_diff_method = [
    [DefaultQubit(), "finite-diff", False, False],
    [DefaultQubit(), "parameter-shift", False, False],
    [DefaultQubit(), "backprop", True, False],
    [DefaultQubit(), "adjoint", True, False],
    [DefaultQubit(), "adjoint", False, False],
    [DefaultQubit(), "adjoint", False, True],
    [DefaultQubit(), "spsa", False, False],
    [DefaultQubit(), "hadamard", False, False],
    [qml.device("lightning.qubit", wires=4), "adjoint", False, True],
    [qml.device("lightning.qubit", wires=4), "adjoint", False, False],
    [qml.device("lightning.qubit", wires=4), "adjoint", True, True],
    [qml.device("lightning.qubit", wires=4), "adjoint", True, False],
    [qml.device("reference.qubit"), "parameter-shift", False, False],
]

TOL_FOR_SPSA = 1.0
H_FOR_SPSA = 0.01

interface_and_qubit_device_and_diff_method = [
    ["auto"] + inner_list for inner_list in qubit_device_and_diff_method
] + [["tf"] + inner_list for inner_list in qubit_device_and_diff_method]


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution,device_vjp",
    interface_and_qubit_device_and_diff_method,
)
class TestQNode:
    """Test that using the QNode with TensorFlow integrates with the PennyLane stack"""

    def test_execution_with_interface(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(0.1)
        circuit(a)

        with tf.GradientTape() as tape:
            res = circuit(a)

        assert circuit.interface == interface

        # with the interface, the tape returns tensorflow tensors
        assert isinstance(res, tf.Tensor)
        assert res.shape == ()

        # gradients should work
        grad = tape.gradient(res, a)
        assert isinstance(grad, tf.Tensor)
        assert grad.shape == ()

    def test_interface_swap(self, dev, diff_method, grad_on_execution, device_vjp, tol, interface):
        """Test that the TF interface can be applied to a QNode
        with a pre-existing interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        @qnode(
            dev,
            interface="autograd",
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
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

    def test_drawing(self, dev, diff_method, grad_on_execution, device_vjp, interface):
        """Test circuit drawing when using the TF interface"""

        x = tf.Variable(0.1, dtype=tf.float64)
        y = tf.Variable([0.2, 0.3], dtype=tf.float64)
        z = tf.Variable(0.4, dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        result = qml.draw(circuit)(p1=x, p3=z)
        expected = "0: ──RX(0.10)──RX(0.40)─╭●─┤  <Z>\n1: ──RY(0.06)───────────╰X─┤  <Z>"
        assert result == expected

    def test_jacobian(self, dev, diff_method, grad_on_execution, device_vjp, tol, interface, seed):
        """Test jacobian calculation"""
        gradient_kwargs = {}
        kwargs = {
            "diff_method": diff_method,
            "grad_on_execution": grad_on_execution,
            "interface": interface,
            "device_vjp": device_vjp,
        }
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a, b)
            res = tf.stack(res)

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [a, b], experimental_use_pfor=not device_vjp)
        expected = [[-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_jacobian_options(self, dev, diff_method, grad_on_execution, device_vjp, interface):
        """Test setting finite-difference jacobian options"""
        if diff_method not in {"finite-diff", "spsa"}:
            pytest.skip("Test only works with finite diff and spsa.")

        a = tf.Variable([0.1, 0.2])

        gradient_kwargs = {"approx_order": 2, "h": 1e-8}

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a)

        tape.jacobian(res, a, experimental_use_pfor=not device_vjp)

    def test_changing_trainability(
        self, dev, diff_method, grad_on_execution, device_vjp, tol, interface
    ):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method in ["backprop", "adjoint", "spsa"]:
            pytest.skip("Test does not support backprop, adjoint or spsa method")

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        diff_kwargs = {}
        if diff_method == "finite-diff":
            diff_kwargs = {"approx_order": 2, "strategy": "center"}

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
            gradient_kwargs=diff_kwargs,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a, b)
            res = tf.stack(res)

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac = tape.jacobian(res, [a, b], experimental_use_pfor=not device_vjp)
        expected = [
            [-tf.sin(a), tf.sin(a) * tf.sin(b)],
            [0, -tf.cos(a) * tf.cos(b)],
        ]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        # make the second QNode argument a constant
        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.constant(0.8, dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a, b)
            res = tf.stack(res)

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        jac = tape.jacobian(res, a, experimental_use_pfor=not device_vjp)
        expected = [-tf.sin(a), tf.sin(a) * tf.sin(b)]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

    def test_classical_processing(self, dev, diff_method, grad_on_execution, device_vjp, interface):
        """Test classical processing within the quantum tape"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.constant(0.2, dtype=tf.float64)
        c = tf.Variable(0.3, dtype=tf.float64)

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
        )
        def circuit(x, y, z):
            qml.RY(x * z, wires=0)
            qml.RZ(y, wires=0)
            qml.RX(z + z**2 + tf.sin(a), wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a, b, c)

        res = tape.jacobian(res, [a, b, c], experimental_use_pfor=not device_vjp)

        assert isinstance(res[0], tf.Tensor)
        assert res[1] is None
        assert isinstance(res[2], tf.Tensor)

    def test_no_trainable_parameters(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """Test evaluation if there are no trainable parameters"""

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        a = 0.1
        b = tf.constant(0.2, dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a, b)
            res = tf.stack(res)

        assert res.shape == (2,)
        assert isinstance(res, tf.Tensor)

        # can't take the gradient with respect to "a" since it's a Python scalar
        grad = tape.jacobian(res, b, experimental_use_pfor=not device_vjp)
        assert grad is None

    @pytest.mark.parametrize("U", [tf.constant([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])])
    def test_matrix_parameter(
        self, dev, diff_method, grad_on_execution, device_vjp, U, tol, interface
    ):
        """Test that the TF interface works correctly
        with a matrix parameter"""
        a = tf.Variable(0.1, dtype=tf.float64)

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            interface=interface,
            device_vjp=device_vjp,
        )
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(U, a)

        assert np.allclose(res, -tf.cos(a), atol=tol, rtol=0)

        res = tape.jacobian(res, a, experimental_use_pfor=not device_vjp)
        assert np.allclose(res, tf.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(
        self, dev, diff_method, grad_on_execution, device_vjp, tol, interface, seed
    ):
        """Test that operation and nested tapes expansion
        is differentiable"""
        gradient_kwargs = {}
        kwargs = {
            "diff_method": diff_method,
            "grad_on_execution": grad_on_execution,
            "interface": interface,
            "device_vjp": device_vjp,
        }
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        class U3(qml.U3):
            def decomposition(self):
                theta, phi, lam = self.data
                wires = self.wires
                return [
                    qml.Rot(lam, theta, -lam, wires=wires),
                    qml.PhaseShift(phi + lam, wires=wires),
                ]

        a = np.array(0.1)
        p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a, p)

        expected = tf.cos(a) * tf.cos(p[1]) * tf.sin(p[0]) + tf.sin(a) * (
            tf.cos(p[2]) * tf.sin(p[1]) + tf.cos(p[0]) * tf.cos(p[1]) * tf.sin(p[2])
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, p, experimental_use_pfor=not device_vjp)
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

    def test_changing_shots(self, interface):
        """Test that changing shots works on execution"""
        dev = DefaultQubit()
        a, b = [0.543, -0.654]
        weights = tf.Variable([a, b], dtype=tf.float64)

        @qnode(dev, interface=interface, diff_method=qml.gradients.param_shift)
        def circuit(weights):
            qml.RY(weights[0], wires=0)
            qml.RX(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=(0, 1))

        # execute with device default shots (None)
        with pytest.raises(qml.DeviceError):
            circuit(weights)

        # execute with shots=100
        res = circuit(weights, shots=100)  # pylint: disable=unexpected-keyword-arg
        assert res.shape == (100, 2)

    def test_gradient_integration(self, interface):
        """Test that temporarily setting the shots works
        for gradient computations"""
        # pylint: disable=unexpected-keyword-arg
        dev = DefaultQubit()
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

        assert len(res) == 3

        jacobian = tape.jacobian(res, weights)
        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(np.mean(jacobian, axis=0), expected, atol=0.1, rtol=0)

    def test_multiple_gradient_integration(self, tol, interface):
        """Test that temporarily setting the shots works
        for gradient computations, even if the QNode has been re-evaluated
        with a different number of shots in the meantime."""
        dev = DefaultQubit()
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

        assert qml.math.shape(res1) == ()

        res2 = circuit(weights, shots=[(1, 1000)])  # pylint: disable=unexpected-keyword-arg
        assert qml.math.shape(res2) == (1000,)

        grad = tape.gradient(res1, weights)
        expected = [np.sin(a) * np.sin(b), -np.cos(a) * np.cos(b)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_update_diff_method(self, interface):
        """Test that temporarily setting the shots updates the diff method"""
        dev = DefaultQubit()
        weights = tf.Variable([0.543, -0.654], dtype=tf.float64)

        @qnode(dev, interface=interface)
        def circuit(weights):
            qml.RY(weights[0], wires=0)
            qml.RX(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(1))

        with dev.tracker:
            with tf.GradientTape() as tape:
                res = circuit(weights, shots=100)
            tape.gradient(res, weights)
        # since we are using finite shots, use parameter shift
        assert dev.tracker.totals["executions"] == 5

        # if we use the default shots value of None, backprop can now be used
        with dev.tracker:
            with tf.GradientTape() as tape:
                res = circuit(weights)
            tape.gradient(res, weights)
        assert dev.tracker.totals["executions"] == 1


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution,device_vjp",
    interface_and_qubit_device_and_diff_method,
)
class TestQubitIntegration:
    """Tests that ensure various qubit circuits integrate correctly"""

    def test_probability_differentiation(
        self, dev, diff_method, grad_on_execution, device_vjp, tol, interface, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with multiple probs outputs"""

        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("state adjoint diff not supported with lightning")

        kwargs = {
            "diff_method": diff_method,
            "grad_on_execution": grad_on_execution,
            "interface": interface,
            "device_vjp": device_vjp,
        }
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0]), qml.probs(wires=[1])

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(x, y)
            res = tf.stack(res)

        expected = np.array(
            [
                [tf.cos(x / 2) ** 2, tf.sin(x / 2) ** 2],
                [(1 + tf.cos(x) * tf.cos(y)) / 2, (1 - tf.cos(x) * tf.cos(y)) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [x, y], experimental_use_pfor=not device_vjp)
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

    def test_ragged_differentiation(
        self, dev, diff_method, grad_on_execution, device_vjp, tol, interface, seed
    ):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("state adjoint diff not supported with lightning")

        kwargs = {
            "diff_method": diff_method,
            "grad_on_execution": grad_on_execution,
            "interface": interface,
            "device_vjp": device_vjp,
        }
        gradient_kwargs = {}
        if diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        with tf.GradientTape(persistent=device_vjp) as tape:
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

        res = tape.jacobian(res, [x, y], experimental_use_pfor=not device_vjp)
        expected = np.array(
            [
                [-tf.sin(x), -tf.sin(x) * tf.cos(y) / 2, tf.cos(y) * tf.sin(x) / 2],
                [0, -tf.cos(x) * tf.sin(y) / 2, tf.cos(x) * tf.sin(y) / 2],
            ]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_second_derivative(
        self, dev, diff_method, grad_on_execution, device_vjp, tol, interface
    ):
        """Test second derivative calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop", "hadamard"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
            device_vjp=device_vjp,
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

    def test_hessian(self, dev, diff_method, grad_on_execution, device_vjp, tol, interface):
        """Test hessian calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop", "hadamard"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
            device_vjp=device_vjp,
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

    def test_hessian_vector_valued(
        self, dev, diff_method, grad_on_execution, device_vjp, tol, interface
    ):
        """Test hessian calculation of a vector valued QNode"""
        if diff_method not in {"parameter-shift", "backprop", "hadamard"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, tol, interface
    ):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        if diff_method not in {"parameter-shift", "backprop", "hadamard"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
            device_vjp=device_vjp,
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

    def test_hessian_ragged(self, dev, diff_method, grad_on_execution, device_vjp, tol, interface):
        """Test hessian calculation of a ragged QNode"""
        if diff_method not in {"parameter-shift", "backprop", "hadamard"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
            device_vjp=device_vjp,
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

    def test_state(self, dev, diff_method, grad_on_execution, device_vjp, tol, interface):
        """Test that the state can be returned and differentiated"""

        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("state adjoint diff not supported with lightning")

        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.state()

        def cost_fn(x, y):
            res = circuit(x, y)
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
    @pytest.mark.parametrize("dtype", ("int32", "int64"))
    def test_projector(
        self, state, dev, diff_method, grad_on_execution, device_vjp, tol, interface, dtype, seed
    ):
        """Test that the variance of a projector is correctly returned"""
        kwargs = {
            "diff_method": diff_method,
            "grad_on_execution": grad_on_execution,
            "interface": interface,
            "device_vjp": device_vjp,
        }
        gradient_kwargs = {}
        if diff_method == "adjoint":
            pytest.skip("adjoint supports either all expvals or all diagonal measurements.")
        if diff_method == "hadamard":
            pytest.skip("Variance not implemented yet.")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA
        if dev.name == "reference.qubit":
            pytest.xfail("diagonalize_measurements do not support projectors (sc-72911)")

        P = tf.constant(state, dtype=dtype)

        x, y = 0.765, -0.654
        weights = tf.Variable([x, y], dtype=tf.float64)

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
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

    def test_postselection_differentiation(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """Test that when postselecting with default.qubit, differentiation works correctly."""

        if diff_method in ["adjoint", "spsa", "hadamard"]:
            pytest.skip("Diff method does not support postselection.")

        if dev.name == "reference.qubit":
            pytest.skip("reference.qubit does not support postselection.")

        @qml.qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(phi, theta):
            qml.RX(phi, wires=0)
            qml.CNOT([0, 1])
            qml.measure(wires=0, postselect=1)
            qml.RX(theta, wires=1)
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(
            dev,
            diff_method=diff_method,
            interface=interface,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def expected_circuit(theta):
            qml.PauliX(1)
            qml.RX(theta, wires=1)
            return qml.expval(qml.PauliZ(1))

        phi = tf.Variable(1.23)
        theta = tf.Variable(4.56)

        assert np.allclose(circuit(phi, theta), expected_circuit(theta))

        with tf.GradientTape() as res_tape:
            res = circuit(phi, theta)
        gradient = res_tape.gradient(res, [phi, theta])

        with tf.GradientTape() as expected_tape:
            expected = expected_circuit(theta)
        exp_theta_grad = expected_tape.gradient(expected, theta)

        assert np.allclose(gradient, [0.0, exp_theta_grad])


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution,device_vjp",
    interface_and_qubit_device_and_diff_method,
)
class TestTapeExpansion:
    """Test that tape expansion within the QNode integrates correctly
    with the TF interface"""

    def test_gradient_expansion(self, dev, diff_method, grad_on_execution, device_vjp, interface):
        """Test that a *supported* operation with no gradient recipe is
        expanded for both parameter-shift and finite-differences, but not for execution."""
        if diff_method not in ("parameter-shift", "finite-diff", "spsa", "hadamard"):
            pytest.skip("Only supports gradient transforms")

        class PhaseShift(qml.PhaseShift):
            grad_method = None
            has_generator = False

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=2,
            interface=interface,
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, max_diff, interface
    ):
        """Test that a *supported* operation with no gradient recipe is only
        expanded for parameter-shift and finite-differences when it is trainable."""
        if diff_method not in ("parameter-shift", "finite-diff", "spsa", "hadamard"):
            pytest.skip("Only supports gradient transforms")

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
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, max_diff, tol, interface, seed
    ):
        """Test that if there are non-commuting groups and the number of shots is None
        the first and second order gradients are correctly evaluated"""
        kwargs = {
            "diff_method": diff_method,
            "grad_on_execution": grad_on_execution,
            "max_diff": max_diff,
            "interface": interface,
            "device_vjp": device_vjp,
        }
        gradient_kwargs = {}
        if diff_method in ["adjoint", "hadamard"]:
            pytest.skip("The adjoint/hadamard method does not yet support Hamiltonians")
        elif diff_method == "spsa":
            gradient_kwargs["sampler_rng"] = np.random.default_rng(seed)
            gradient_kwargs["num_directions"] = 20
            tol = TOL_FOR_SPSA

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(dev, **kwargs, gradient_kwargs=gradient_kwargs)
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
    def test_hamiltonian_finite_shots(
        self, dev, diff_method, grad_on_execution, device_vjp, max_diff, interface, seed
    ):
        """Test that the Hamiltonian is correctly measured if there
        are non-commuting groups and the number of shots is finite
        and the first and second order gradients are correctly evaluated"""
        gradient_kwargs = {}
        tol = 0.1
        if diff_method in ("adjoint", "backprop", "hadamard"):
            pytest.skip("The adjoint and backprop methods do not yet support sampling")
        elif diff_method == "spsa":
            gradient_kwargs = {
                "h": H_FOR_SPSA,
                "sampler_rng": np.random.default_rng(seed),
                "num_directions": 20,
            }
            tol = TOL_FOR_SPSA
        elif diff_method == "finite-diff":
            gradient_kwargs = {"h": 0.05}

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]

        @qnode(
            dev,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            max_diff=max_diff,
            interface=interface,
            device_vjp=device_vjp,
            gradient_kwargs=gradient_kwargs,
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
        with tf.GradientTape(persistent=True) as _t2:
            with tf.GradientTape() as t1:
                res = circuit(d, w, c, shots=50000)  # pylint:disable=unexpected-keyword-arg

            expected = c[2] * np.cos(d[1] + w[1]) - c[1] * np.sin(d[0] + w[0]) * np.sin(d[1] + w[1])
            assert np.allclose(res, expected, atol=tol)

            # test gradients
            grad = t1.gradient(res, [d, w, c])

        if diff_method in ["finite-diff", "spsa"]:
            pytest.skip(f"{diff_method} not compatible")

        expected_w = [
            -c[1] * np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]),
            -c[1] * np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]) - c[2] * np.sin(d[1] + w[1]),
        ]
        expected_c = [0, -np.sin(d[0] + w[0]) * np.sin(d[1] + w[1]), np.cos(d[1] + w[1])]
        assert np.allclose(grad[1], expected_w, atol=tol)
        assert np.allclose(grad[2], expected_c, atol=tol)

        # test second-order derivatives
        # TODO: figure out why grad2_c is np.zeros((3,3)) instead of None
        # if diff_method == "parameter-shift" and max_diff == 2:
        #     grad2_c = _t2.jacobian(grad[2], c)
        #     print(grad2_c, grad[2], c)
        #     assert grad2_c is None

        #     grad2_w_c = _t2.jacobian(grad[1], c)
        #     expected = [0, -np.cos(d[0] + w[0]) * np.sin(d[1] + w[1]), 0], [
        #         0,
        #         -np.cos(d[1] + w[1]) * np.sin(d[0] + w[0]),
        #         -np.sin(d[1] + w[1]),
        #     ]
        #     assert np.allclose(grad2_w_c, expected, atol=tol)


class TestSample:
    """Tests for the sample integration"""

    # pylint:disable=unexpected-keyword-arg

    def test_sample_dimension(self):
        """Test sampling works as expected"""

        @qnode(DefaultQubit(), diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        res = circuit(shots=10)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert res[0].shape == (10,)
        assert res[1].shape == (10,)
        assert isinstance(res[0], tf.Tensor)
        assert isinstance(res[1], tf.Tensor)

    def test_sampling_expval(self):
        """Test sampling works as expected if combined with expectation values"""

        @qnode(DefaultQubit(), diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        res = circuit(shots=10)

        assert len(res) == 2
        assert isinstance(res, tuple)
        assert res[0].shape == (10,)
        assert res[1].shape == ()
        assert isinstance(res[0], tf.Tensor)
        assert isinstance(res[1], tf.Tensor)

    def test_sample_combination(self):
        """Test the output of combining expval, var and sample"""

        @qnode(DefaultQubit(), diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit(shots=10)

        assert len(result) == 3
        assert result[0].shape == (10,)
        assert result[1].shape == ()
        assert result[2].shape == ()
        assert isinstance(result[0], tf.Tensor)
        assert isinstance(result[1], tf.Tensor)
        assert isinstance(result[2], tf.Tensor)
        assert result[0].dtype is tf.float64  # pylint:disable=no-member
        assert result[1].dtype is tf.float64  # pylint:disable=no-member
        assert result[2].dtype is tf.float64  # pylint:disable=no-member

    def test_single_wire_sample(self):
        """Test the return type and shape of sampling a single wire"""

        @qnode(DefaultQubit(), diff_method="parameter-shift", interface="tf")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0))

        result = circuit(shots=10)

        assert isinstance(result, tf.Tensor)
        assert np.array_equal(result.shape, (10,))

    def test_multi_wire_sample_regular_shape(self):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""

        @qnode(DefaultQubit(), diff_method="parameter-shift", interface="tf")
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit(shots=10)
        result = tf.stack(result)

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, tf.Tensor)
        assert np.array_equal(result.shape, (3, 10))
        assert result.dtype == tf.float64

    def test_counts(self):
        """Test counts works as expected for TF"""

        # pylint:disable=unsubscriptable-object,no-member
        @qnode(DefaultQubit(), interface="tf")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.counts(qml.PauliZ(0))

        res = circuit(shots=100)

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
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(DefaultQubit(), diff_method="parameter-shift", interface=interface)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0]), qml.probs(wires=[1])

        with tf.GradientTape() as tape:
            p0, p1 = circuit(x, y)
            loss = p0[0] + p1[1]  # pylint:disable=unsubscriptable-object

        expected = tf.cos(x / 2) ** 2 + (1 - tf.cos(x) * tf.cos(y)) / 2
        assert np.allclose(loss, expected, atol=tol, rtol=0)

        grad = tape.gradient(loss, [x, y])
        expected = [-tf.sin(x) * tf.sin(y / 2) ** 2, tf.cos(x) * tf.sin(y) / 2]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_autograph_jacobian(self, decorator, interface, tol):
        """Test that a parameter-shift vector-valued QNode can be compiled
        using @tf.function, and differentiated"""
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(DefaultQubit(), diff_method="parameter-shift", max_diff=1, interface=interface)
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

        @decorator
        @qnode(
            DefaultQubit(),
            diff_method="adjoint",
            interface=interface,
            grad_on_execution=grad_on_execution,
        )
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

        @decorator
        @qnode(
            DefaultQubit(),
            diff_method="adjoint",
            interface=interface,
            grad_on_execution=grad_on_execution,
        )
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

    @pytest.mark.xfail
    @pytest.mark.parametrize("grad_on_execution", [True, False])
    def test_autograph_adjoint_multi_out(self, grad_on_execution, decorator, interface, tol):
        """Test that a parameter-shift QNode can be compiled
        using @tf.function, and differentiated to second order"""

        @decorator
        @qnode(
            DefaultQubit(),
            diff_method="adjoint",
            interface=interface,
            grad_on_execution=grad_on_execution,
        )
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(0))

        x = tf.Variable(0.5, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = qml.math.hstack(circuit(x))
        g = tape.jacobian(res, x)
        a = x * 1.0

        expected_res = [tf.cos(a), tf.sin(a)]
        assert np.allclose(res, expected_res, atol=tol, rtol=0)

        expected_g = [-tf.sin(a), tf.cos(a)]
        assert np.allclose(g, expected_g, atol=tol, rtol=0)

    def test_autograph_ragged_differentiation(self, decorator, interface, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(DefaultQubit(), diff_method="parameter-shift", max_diff=1, interface=interface)
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
        a = tf.Variable(0.543, dtype=tf.float64)
        b = tf.Variable(-0.654, dtype=tf.float64)

        @decorator
        @qnode(DefaultQubit(), diff_method="parameter-shift", max_diff=2, interface=interface)
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
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        # TODO: fix this for diff_method=None
        @decorator
        @qnode(DefaultQubit(), diff_method="parameter-shift", interface=interface)
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

        @decorator
        @qnode(DefaultQubit(), diff_method="parameter-shift", interface=interface)
        def circuit(**_):
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        res = circuit(shots=10)  # pylint:disable=unexpected-keyword-arg
        res = tf.stack(res)

        assert res.shape == (2, 10)
        assert isinstance(res, tf.Tensor)


@pytest.mark.parametrize(
    "interface,dev,diff_method,grad_on_execution,device_vjp",
    interface_and_qubit_device_and_diff_method,
)
class TestReturn:
    """Class to test the shape of the Grad/Jacobian/Hessian with different return types."""

    def test_grad_single_measurement_param(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """For one measurement and one param, the gradient is a float."""

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """For one measurement and multiple param, the gradient is a tuple of arrays."""

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """For one measurement and multiple param as a single array params, the gradient is an array."""

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned with the correct
        dimension"""

        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("state adjoint diff not supported with lightning")

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable(0.1, dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a)

        jac = tape.jacobian(res, a, experimental_use_pfor=not device_vjp)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """For a multi dimensional measurement (probs), check that a single tuple is returned containing arrays with
        the correct dimension"""

        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("state adjoint diff not supported with lightning")

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a, b)

        jac = tape.jacobian(res, (a, b), experimental_use_pfor=not device_vjp)

        assert isinstance(jac, tuple)

        assert isinstance(jac[0], tf.Tensor)
        assert jac[0].shape == (4,)

        assert isinstance(jac[1], tf.Tensor)
        assert jac[1].shape == (4,)

    def test_jacobian_single_measurement_probs_multiple_param_single_array(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """For a multi dimensional measurement (probs), check that a single array is returned."""

        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("state adjoint diff not supported with lightning")

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.probs(wires=[0, 1])

        a = tf.Variable([0.1, 0.2], dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a)

        jac = tape.jacobian(res, a, experimental_use_pfor=not device_vjp)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (4, 2)

    def test_jacobian_multiple_measurement_single_param(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The jacobian of multiple measurements with a single params return an array."""

        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("state adjoint diff not supported with lightning")

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable(0.1, dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a)
            res = tf.experimental.numpy.hstack(res)

        jac = tape.jacobian(res, a, experimental_use_pfor=not device_vjp)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (5,)

    def test_jacobian_multiple_measurement_multiple_param(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The jacobian of multiple measurements with a multiple params return a tuple of arrays."""

        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("state adjoint diff not supported with lightning")

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a, b)
            res = tf.experimental.numpy.hstack(res)

        jac = tape.jacobian(res, (a, b), experimental_use_pfor=not device_vjp)

        assert isinstance(jac, tuple)
        assert len(jac) == 2

        assert isinstance(jac[0], tf.Tensor)
        assert jac[0].shape == (5,)

        assert isinstance(jac[1], tf.Tensor)
        assert jac[1].shape == (5,)

    def test_jacobian_multiple_measurement_multiple_param_array(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The jacobian of multiple measurements with a multiple params array return a single array."""

        if "lightning" in getattr(dev, "name", "").lower():
            pytest.xfail("state adjoint diff not supported with lightning")

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
        )
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        a = tf.Variable([0.1, 0.2], dtype=tf.float64)

        with tf.GradientTape(persistent=device_vjp) as tape:
            res = circuit(a)
            res = tf.experimental.numpy.hstack(res)

        jac = tape.jacobian(res, a, experimental_use_pfor=not device_vjp)

        assert isinstance(jac, tf.Tensor)
        assert jac.shape == (5, 2)

    def test_hessian_expval_multiple_params(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
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
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The hessian of single measurement with a multiple params array return a single array."""

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
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

    def test_hessian_var_multiple_params(
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The hessian of single a measurement with multiple params return a tuple of arrays."""
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
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The hessian of single measurement with a multiple params array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        if diff_method == "hadamard":
            pytest.skip("Test does not support hadamard because of variance.")

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
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
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""

        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")

        if diff_method == "hadamard":
            pytest.skip("Test does not support hadamard because multiple measurements.")

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The hessian of multiple measurements with multiple params return a tuple of arrays."""
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
            device_vjp=device_vjp,
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
        self, dev, diff_method, grad_on_execution, device_vjp, interface
    ):
        """The hessian of multiple measurements with a multiple param array return a single array."""
        if diff_method == "adjoint":
            pytest.skip("Test does not supports adjoint because second order diff.")
        if diff_method == "hadamard":
            pytest.skip("Test does not support hadamard because of variance.")

        params = tf.Variable([0.1, 0.2], dtype=tf.float64)

        @qnode(
            dev,
            interface=interface,
            diff_method=diff_method,
            max_diff=2,
            grad_on_execution=grad_on_execution,
            device_vjp=device_vjp,
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


def test_no_ops():
    """Test that the return value of the QNode matches in the interface
    even if there are no ops"""

    @qml.qnode(DefaultQubit(), interface="tf")
    def circuit():
        qml.Hadamard(wires=0)
        return qml.state()

    res = circuit()
    assert isinstance(res, tf.Tensor)


def test_error_device_vjp_jacobian():
    """Test a ValueError is raised if a jacobian is attempted to be computed with device_vjp=True."""

    dev = qml.device("default.qubit")

    @qml.qnode(dev, diff_method="adjoint", device_vjp=True)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(0))

    x = tf.Variable(0.1)

    with tf.GradientTape() as tape:
        y = qml.math.hstack(circuit(x))

    with pytest.raises(ValueError):
        tape.jacobian(y, x)


def test_error_device_vjp_state_float32():
    """Test a ValueError is raised is state differentiation is attemped with float32 parameters."""

    dev = qml.device("default.qubit")

    @qml.qnode(dev, diff_method="adjoint", device_vjp=True)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.probs(wires=0)

    x = tf.Variable(0.1, dtype=tf.float32)
    with pytest.raises(ValueError, match="tensorflow with adjoint differentiation of the state"):
        with tf.GradientTape():
            circuit(x)
