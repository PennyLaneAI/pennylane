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
"""Unit tests for the tf interface"""
import pytest

tf = pytest.importorskip("tensorflow", minversion="2.1")

import numpy as np

import pennylane as qml
from pennylane import qnode, QNode
from pennylane.tape import JacobianTape


@pytest.mark.parametrize(
    "dev_name,diff_method",
    [
        ["default.qubit", "finite-diff"],
        ["default.qubit", "parameter-shift"],
        ["default.qubit", "backprop"],
        ["default.qubit", "adjoint"],
    ],
)
class TestQNode:
    """Tests the tensorflow interface used with a QNode."""

    def test_import_error(self, dev_name, diff_method, mocker):
        """Test that an exception is caught on import error"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        mock = mocker.patch("pennylane.interfaces.tf.TFInterface.apply")
        mock.side_effect = ImportError()

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device(dev_name, wires=2)
        qn = QNode(func, dev, interface="tf", diff_method=diff_method)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="TensorFlow not found. Please install the latest version of TensorFlow to enable the 'tf' interface",
        ):
            qn(0.1, 0.1)

    def test_execution_no_interface(self, dev_name, diff_method):
        """Test execution works without an interface, and that trainable parameters
        are correctly inferred within a gradient tape."""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(0.1)

        with tf.GradientTape() as tape:
            res = circuit(a)

        assert circuit.qtape.interface == "autograd"

        # without the interface, the tape simply returns an array of results
        assert isinstance(res, np.ndarray)
        assert res.shape == tuple()

        # without the interface, the tape is unable to deduce
        # trainable parameters
        assert circuit.qtape.trainable_params == {0}

        # gradients should cause an error
        with pytest.raises(AttributeError, match="has no attribute '_id'"):
            assert tape.gradient(res, a) is None

    def test_execution_with_interface(self, dev_name, diff_method):
        """Test execution works with the interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(a):
            qml.RY(a, wires=0)
            qml.RX(0.2, wires=0)
            return qml.expval(qml.PauliZ(0))

        a = tf.Variable(0.1)
        circuit(a)

        # if executing outside a gradient tape, the number of trainable parameters
        # cannot be determined by TensorFlow
        assert circuit.qtape.trainable_params == set()

        with tf.GradientTape() as tape:
            res = circuit(a)

        assert circuit.qtape.interface == "tf"

        # with the interface, the tape returns tensorflow tensors
        assert isinstance(res, tf.Tensor)
        assert res.shape == tuple()

        # the tape is able to deduce trainable parameters
        assert circuit.qtape.trainable_params == {0}

        # gradients should work
        grad = tape.gradient(res, a)
        assert isinstance(grad, tf.Tensor)
        assert grad.shape == tuple()

    def test_interface_swap(self, dev_name, diff_method, tol):
        """Test that the TF interface can be applied to a QNode
        with a pre-existing interface"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, interface="autograd", diff_method=diff_method)
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
        circuit.to_tf()

        a = tf.Variable(0.1, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res2 = circuit(a)

        grad2 = tape.gradient(res2, a)
        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(grad1, grad2, atol=tol, rtol=0)

    def test_drawing(self, dev_name, diff_method):
        """Test circuit drawing when using the TF interface"""

        x = tf.Variable(0.1, dtype=tf.float64)
        y = tf.Variable([0.2, 0.3], dtype=tf.float64)
        z = tf.Variable(0.4, dtype=tf.float64)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        circuit(p1=x, p3=z)

        result = circuit.draw()
        expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──╭┤ State 
 1: ──RY(0.06)───────────╰X──╰┤ State 
"""

        assert result == expected

    def test_jacobian(self, dev_name, diff_method, mocker, tol):
        """Test jacobian calculation"""
        spy = mocker.spy(JacobianTape, "jacobian")
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        assert circuit.qtape.trainable_params == {0, 1}

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = tape.jacobian(res, [a, b])
        expected = [[-tf.sin(a), tf.sin(a) * tf.sin(b)], [0, -tf.cos(a) * tf.cos(b)]]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        if diff_method == "finite-diff":
            spy.assert_called()
        elif diff_method == "backprop":
            spy.assert_not_called()

    def test_jacobian_dtype(self, dev_name, diff_method, tol):
        """Test calculating the jacobian with a different datatype"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        a = tf.Variable(0.1, dtype=tf.float32)
        b = tf.Variable(0.2, dtype=tf.float32)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, diff_method=diff_method)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))]

        circuit.to_tf(dtype=tf.float32)
        assert circuit.dtype is tf.float32

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        assert circuit.qtape.interface == "tf"
        assert circuit.qtape.trainable_params == {0, 1}

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2,)
        assert res.dtype is tf.float32

        res = tape.jacobian(res, [a, b])
        assert [r.dtype is tf.float32 for r in res]

    def test_jacobian_options(self, dev_name, diff_method, mocker, tol):
        """Test setting finite-difference jacobian options"""
        if diff_method != "finite-diff":
            pytest.skip("Test only works with finite diff")

        spy = mocker.spy(JacobianTape, "numeric_pd")

        a = tf.Variable([0.1, 0.2])

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface="tf", h=1e-8, order=2, diff_method=diff_method)
        def circuit(a):
            qml.RY(a[0], wires=0)
            qml.RX(a[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(a)

        tape.jacobian(res, a)

        for args in spy.call_args_list:
            assert args[1]["order"] == 2
            assert args[1]["h"] == 1e-8

    def test_changing_trainability(self, dev_name, diff_method, mocker, tol):
        """Test changing the trainability of parameters changes the
        number of differentiation requests made"""
        if diff_method == "backprop":
            pytest.skip("Test does not support backprop")

        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.Variable(0.2, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="tf", diff_method="finite-diff")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        # the tape has reported both gate arguments as trainable
        assert circuit.qtape.trainable_params == {0, 1}

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy = mocker.spy(JacobianTape, "numeric_pd")

        jac = tape.jacobian(res, [a, b])
        expected = [
            [-tf.sin(a), tf.sin(a) * tf.sin(b)],
            [0, -tf.cos(a) * tf.cos(b)],
        ]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        # JacobianTape.numeric_pd has been called for each argument
        assert len(spy.call_args_list) == 2

        # make the second QNode argument a constant
        a = tf.Variable(0.54, dtype=tf.float64)
        b = tf.constant(0.8, dtype=tf.float64)

        with tf.GradientTape() as tape:
            res = circuit(a, b)

        # the tape has reported only the first argument as trainable
        assert circuit.qtape.trainable_params == {0}

        expected = [tf.cos(a), -tf.cos(a) * tf.sin(b)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        spy.call_args_list = []
        jac = tape.jacobian(res, a)
        expected = [-tf.sin(a), tf.sin(a) * tf.sin(b)]
        assert np.allclose(jac, expected, atol=tol, rtol=0)

        # JacobianTape.numeric_pd has been called only once
        assert len(spy.call_args_list) == 1

    def test_classical_processing(self, dev_name, diff_method, tol):
        """Test classical processing within the quantum tape"""
        a = tf.Variable(0.1, dtype=tf.float64)
        b = tf.constant(0.2, dtype=tf.float64)
        c = tf.Variable(0.3, dtype=tf.float64)

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(x, y, z):
            qml.RY(x * z, wires=0)
            qml.RZ(y, wires=0)
            qml.RX(z + z ** 2 + tf.sin(a), wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(a, b, c)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {0, 2}
            assert circuit.qtape.get_parameters() == [a * c, c + c ** 2 + tf.sin(a)]

        res = tape.jacobian(res, [a, b, c])

        assert isinstance(res[0], tf.Tensor)
        assert res[1] is None
        assert isinstance(res[2], tf.Tensor)

    def test_no_trainable_parameters(self, dev_name, diff_method, tol):
        """Test evaluation if there are no trainable parameters"""
        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="tf")
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
            assert circuit.qtape.trainable_params == set()

        assert res.shape == (2,)
        assert isinstance(res, tf.Tensor)

    @pytest.mark.parametrize("U", [tf.constant([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])])
    def test_matrix_parameter(self, dev_name, diff_method, U, tol):
        """Test that the TF interface works correctly
        with a matrix parameter"""
        a = tf.Variable(0.1, dtype=tf.float64)

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(U, a):
            qml.QubitUnitary(U, wires=0)
            qml.RY(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        with tf.GradientTape() as tape:
            res = circuit(U, a)

        if diff_method == "finite-diff":
            assert circuit.qtape.trainable_params == {1}

        assert np.allclose(res, -tf.cos(a), atol=tol, rtol=0)

        res = tape.jacobian(res, a)
        assert np.allclose(res, tf.sin(a), atol=tol, rtol=0)

    def test_differentiable_expand(self, dev_name, diff_method, tol):
        """Test that operation and nested tapes expansion
        is differentiable"""

        class U3(qml.U3):
            def expand(self):
                theta, phi, lam = self.data
                wires = self.wires

                with JacobianTape() as tape:
                    qml.Rot(lam, theta, -lam, wires=wires)
                    qml.PhaseShift(phi + lam, wires=wires)

                return tape

        dev = qml.device(dev_name, wires=1)
        a = np.array(0.1)
        p = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(a, p):
            qml.RX(a, wires=0)
            U3(p[0], p[1], p[2], wires=0)
            return qml.expval(qml.PauliX(0))

        with tf.GradientTape() as tape:
            res = circuit(a, p)

        assert circuit.qtape.trainable_params == {1, 2, 3, 4}
        assert [i.name for i in circuit.qtape.operations] == ["RX", "Rot", "PhaseShift"]
        assert np.all(circuit.qtape.get_parameters() == [p[2], p[0], -p[2], p[1] + p[2]])

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

    def test_probability_differentiation(self, dev_name, diff_method, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple probs outputs"""

        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        dev = qml.device(dev_name, wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface="tf")
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

    def test_ragged_differentiation(self, dev_name, diff_method, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        if diff_method == "adjoint":
            pytest.skip("The adjoint method does not currently support returning probabilities")

        dev = qml.device(dev_name, wires=2)
        x = tf.Variable(0.543, dtype=tf.float64)
        y = tf.Variable(-0.654, dtype=tf.float64)

        @qnode(dev, diff_method=diff_method, interface="tf")
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

    def test_sampling(self, dev_name, diff_method):
        """Test sampling works as expected"""
        if diff_method == "backprop":
            pytest.skip("Sampling not possible with backprop differentiation.")

        dev = qml.device(dev_name, wires=2, shots=10)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            return [qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))]

        with tf.GradientTape() as tape:
            res = circuit()

        assert res.shape == (2, 10)
        assert isinstance(res, tf.Tensor)

    def test_second_derivative(self, dev_name, diff_method, mocker, tol):
        """Test second derivative calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="tf")
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

        spy = mocker.spy(JacobianTape, "hessian")
        g2 = tape1.gradient(res2, x)

        if diff_method == "parameter-shift":
            spy.assert_called_once()
        elif diff_method == "backprop":
            spy.assert_not_called()

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

    def test_hessian(self, dev_name, diff_method, mocker, tol):
        """Test hessian calculation of a scalar valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        x = tf.Variable([1.0, 2.0], dtype=tf.float64)

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                res = circuit(x)
            g = tape2.gradient(res, x)

        spy = mocker.spy(JacobianTape, "hessian")
        hess = tape1.jacobian(g, x)

        if diff_method == "parameter-shift":
            spy.assert_called_once()
        elif diff_method == "backprop":
            spy.assert_not_called()

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

    def test_hessian_vector_valued(self, dev_name, diff_method, mocker, tol):
        """Test hessian calculation of a vector valued QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            return qml.probs(wires=0)

        x = tf.Variable([1.0, 2.0], dtype=tf.float64)

        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(x)

            spy = mocker.spy(JacobianTape, "hessian")
            g = tape2.jacobian(res, x, experimental_use_pfor=False)

        hess = tape1.jacobian(g, x, experimental_use_pfor=False)

        if diff_method == "parameter-shift":
            spy.assert_called_once()
        elif diff_method == "backprop":
            spy.assert_not_called()

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

    def test_hessian_vector_valued_postprocessing(self, dev_name, diff_method, mocker, tol):
        """Test hessian calculation of a vector valued QNode with post-processing"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=1)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(0))]

        x = tf.Variable([0.76, -0.87], dtype=tf.float64)

        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = tf.tensordot(x, circuit(x), axes=[0, 0])

            spy = mocker.spy(JacobianTape, "hessian")
            g = tape2.jacobian(res, x, experimental_use_pfor=False)

        hess = tape1.jacobian(g, x, experimental_use_pfor=False)

        if diff_method == "parameter-shift":
            spy.assert_called_once()
        elif diff_method == "backprop":
            spy.assert_not_called()

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

    def test_hessian_ragged(self, dev_name, diff_method, mocker, tol):
        """Test hessian calculation of a ragged QNode"""
        if diff_method not in {"parameter-shift", "backprop"}:
            pytest.skip("Test only supports parameter-shift or backprop")

        dev = qml.device(dev_name, wires=2)

        @qnode(dev, diff_method=diff_method, interface="tf")
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=0)
            qml.RY(x[0], wires=1)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=1)

        x = tf.Variable([1.0, 2.0], dtype=tf.float64)
        res = circuit(x)

        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape2:
                res = circuit(x)

            spy = mocker.spy(JacobianTape, "hessian")
            g = tape2.jacobian(res, x, experimental_use_pfor=False)

        hess = tape1.jacobian(g, x, experimental_use_pfor=False)

        if diff_method == "parameter-shift":
            spy.assert_called_once()
        elif diff_method == "backprop":
            spy.assert_not_called()

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


class Test_adjoint:
    def test_adjoint_default_save_state(self, mocker):
        """Tests that the state will be saved by default"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="adjoint", interface="tf")
        def circ(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=(0, 1))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        expected_grad = lambda x: np.array([-np.sin(x[0]), np.cos(x[1])])

        spy = mocker.spy(dev, "adjoint_jacobian")

        x1 = tf.Variable([0.1, 0.2])
        x2 = tf.Variable([0.3, 0.4])

        with tf.GradientTape() as tape1:
            res1 = circ(x1)

        with tf.GradientTape() as tape2:
            res2 = circ(x2)

        grad1 = tape1.gradient(res1, x1)
        grad2 = tape2.gradient(res2, x2)

        assert np.allclose(grad1, expected_grad(x1))
        assert np.allclose(grad2, expected_grad(x2))

        assert circ.device.num_executions == 2
        spy.assert_called_with(mocker.ANY, starting_state=mocker.ANY)

    def test_adjoint_save_state(self, mocker):
        """Tests that the tf interface reuses device state when prompted by `cache_state=True`.
        Also makes sure executing a second circuit before backward pass does not interfere
        with answer.
        """

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="adjoint", interface="tf", adjoint_cache=True)
        def circ(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=(0, 1))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        expected_grad = lambda x: np.array([-np.sin(x[0]), np.cos(x[1])])

        spy = mocker.spy(dev, "adjoint_jacobian")

        x1 = tf.Variable([0.1, 0.2])
        x2 = tf.Variable([0.3, 0.4])

        with tf.GradientTape() as tape1:
            res1 = circ(x1)

        with tf.GradientTape() as tape2:
            res2 = circ(x2)

        grad1 = tape1.gradient(res1, x1)
        grad2 = tape2.gradient(res2, x2)

        assert np.allclose(grad1, expected_grad(x1))
        assert np.allclose(grad2, expected_grad(x2))

        assert circ.device.num_executions == 2
        spy.assert_called_with(mocker.ANY, starting_state=mocker.ANY)

        assert circ.qtape.jacobian_options["adjoint_cache"] == True

    def test_adjoint_no_save_state(self, mocker):
        """Tests that with `adjoint_cache=False`, the state is not cached"""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, diff_method="adjoint", interface="tf", adjoint_cache=False)
        def circ(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(dev, "adjoint_jacobian")

        x = tf.Variable(0.1)
        with tf.GradientTape() as tape:
            res = circ(x)

        grad = tape.gradient(res, x)

        assert circ.device.num_executions == 2
        spy.assert_called_with(mocker.ANY)

        assert circ.qtape.jacobian_options.get("adjoint_cache", False) == False


def qtransform(qnode, a, framework=tf):
    """Transforms every RY(y) gate in a circuit to RX(-a*cos(y))"""

    def construct(self, args, kwargs):
        """New quantum tape construct method, that performs
        the transform on the tape in a define-by-run manner"""

        # the following global variable is defined simply for testing
        # purposes, so that we can easily extract the transformed operations
        # for verification.
        global t_op

        t_op = []

        QNode.construct(self, args, kwargs)

        new_ops = []
        for o in self.qtape.operations:
            # here, we loop through all tape operations, and make
            # the transformation if a RY gate is encountered.
            if isinstance(o, qml.RY):
                t_op.append(qml.RX(-a * framework.cos(o.data[0]), wires=o.wires))
                new_ops.append(t_op[-1])
            else:
                new_ops.append(o)

        self.qtape._ops = new_ops
        self.qtape._update()

    import copy

    new_qnode = copy.deepcopy(qnode)
    new_qnode.construct = construct.__get__(new_qnode, QNode)
    return new_qnode


@pytest.mark.parametrize(
    "dev_name,diff_method",
    [("default.qubit", "finite-diff"), ("default.qubit.tf", "backprop")],
)
def test_transform(dev_name, diff_method, tol):
    """Test an example transform"""

    dev = qml.device(dev_name, wires=1)

    @qnode(dev, interface="tf", diff_method=diff_method)
    def circuit(weights):
        # the following global variables are defined simply for testing
        # purposes, so that we can easily extract the operations for verification.
        global op1, op2
        op1 = qml.RY(weights[0], wires=0)
        op2 = qml.RX(weights[1], wires=0)
        return qml.expval(qml.PauliZ(wires=0))

    weights = tf.Variable([0.32, 0.543], dtype=tf.float64)
    a = tf.Variable(0.5, dtype=tf.float64)

    with tf.GradientTape(persistent=True) as tape:
        # transform the circuit QNode with trainable weight 'a'
        new_qnode = qtransform(circuit, a)
        # evaluate the transformed QNode
        res = new_qnode(weights)
        # evaluate the original QNode with pre-processed parameters
        res2 = circuit(tf.sin(weights))
        # the loss is the sum of the two QNode evaluations
        loss = res + res2

    # verify that the transformed QNode has the expected operations
    assert circuit.qtape.operations == [op1, op2]
    assert new_qnode.qtape.operations[0] == t_op[0]
    assert new_qnode.qtape.operations[1].name == op2.name
    assert new_qnode.qtape.operations[1].wires == op2.wires

    # check that the incident gate arguments of both QNode tapes are correct
    assert np.all(circuit.qtape.get_parameters() == tf.sin(weights))
    assert np.all(new_qnode.qtape.get_parameters() == [-a * tf.cos(weights[0]), weights[1]])

    # verify that the gradient has the correct shape
    grad = tape.gradient(loss, [weights, a])
    assert len(grad) == 2
    assert grad[0].shape == weights.shape
    assert grad[1].shape == a.shape

    # compare against the expected values
    assert np.allclose(loss, 1.8244501889992706, atol=tol, rtol=0)
    assert np.allclose(grad[0], [-0.26610258, -0.47053553], atol=tol, rtol=0)
    assert np.allclose(grad[1], 0.06486032, atol=tol, rtol=0)
